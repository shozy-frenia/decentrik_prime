"""Fused LM-head statistics and gradient over the 151,936-token vocabulary, one program per token.

Base Soup materialises the [4, T, 151936] logits several times (fp16 GEMM output, accelerate's
fp32 copy, the fp32 upcast inside the loss, log-softmax for the backward, TRL's chunked entropy,
the argmax for accuracy, per-row cross-entropy for the rewind log): gigabytes of memory and a
long tail of memory-bound kernels. Here the logits exist only for one chunk of CHUNK tokens at a
time (fp16, written by cuBLAS), and this kernel reads each logits row and produces everything
Soup logs plus the gradient:
    pass 1: row max and first-index argmax (torch.argmax tie rule)       -> accuracy
    pass 2: s = sum exp(x - m), t = sum exp(x - m) * (x - m)
            lse = m + log s,  entropy = log s - t / s,  loss = lse - x[target]
    pass 3 (rows with a target, only in gradient chunks): in place, fp16(scale * (softmax - onehot))
The caller multiplies that chunk by the embedding matrix (cuBLAS) to get d(hidden) during the
forward pass, so the [CHUNK, V] buffer is reused for the next chunk and never stored for the
backward (the classic fused-linear-cross-entropy trick, cf. Liger-Kernel's FusedLinearCrossEntropy
and Apple's Cut Cross-Entropy; code ours).
stats row = (loss, entropy, correct, lse) in fp32.
"""

import triton
import triton.language as tl

HEAD_BLOCK_V = 2048
HEAD_WARPS = 8


@triton.jit(do_not_specialize=["row0", "n_rows", "vocab", "do_grad"])
def lm_head_rows_kernel(logits_ptr, tgt_ptr, stats_ptr, row0, n_rows, vocab, do_grad, grad_scale,
                        BLOCK_V: tl.constexpr):
    r = tl.program_id(0)
    row = row0 + r
    base = logits_ptr + r.to(tl.int64) * vocab
    t = tl.load(tgt_ptr + row)
    lanes = tl.arange(0, BLOCK_V)
    # pass 1: max and the first index reaching it
    best = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_i = tl.zeros([BLOCK_V], tl.int32)
    for v0 in range(0, vocab, BLOCK_V):
        cols = v0 + lanes
        x = tl.load(base + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
        upd = x > best
        best = tl.where(upd, x, best)
        best_i = tl.where(upd, cols, best_i)
    m = tl.max(best, axis=0)
    amax = tl.min(tl.where(best == m, best_i, 2147483647), axis=0)
    # pass 2: softmax statistics
    acc_s = tl.zeros([BLOCK_V], tl.float32)
    acc_t = tl.zeros([BLOCK_V], tl.float32)
    for v0 in range(0, vocab, BLOCK_V):
        cols = v0 + lanes
        cm = cols < vocab
        x = tl.load(base + cols, mask=cm, other=float("-inf")).to(tl.float32)
        z = x - m
        e = tl.exp(z)
        acc_s += e
        acc_t += tl.where(cm, e * z, 0.0)
    s = tl.sum(acc_s, axis=0)
    log_s = tl.log(s)
    lse = m + log_s
    ent = log_s - tl.sum(acc_t, axis=0) / s
    valid = t >= 0
    xt = tl.load(base + tl.maximum(t, 0)).to(tl.float32)
    loss = tl.where(valid, lse - xt, 0.0)
    hit = tl.where(valid & (amax == t), 1.0, 0.0)
    out = stats_ptr + row * 4
    tl.store(out, loss)
    tl.store(out + 1, ent)
    tl.store(out + 2, hit)
    tl.store(out + 3, lse)
    # pass 3: gradient of the summed cross-entropy w.r.t. the logits, scaled, in place
    if do_grad != 0:
        if t >= 0:
            for v0 in range(0, vocab, BLOCK_V):
                cols = v0 + lanes
                cm = cols < vocab
                x = tl.load(base + cols, mask=cm, other=0.0).to(tl.float32)
                g = tl.exp(x - lse) - tl.where(cols == t, 1.0, 0.0)
                tl.store(base + cols, (g * grad_scale).to(tl.float16), mask=cm)


def head_rows(logits, tgt, stats, row0, n_rows, do_grad, grad_scale):
    vocab = logits.shape[1]
    lm_head_rows_kernel[(n_rows,)](logits, tgt, stats, int(row0), int(n_rows), int(vocab), int(do_grad),
                                   float(grad_scale), BLOCK_V=HEAD_BLOCK_V, num_warps=HEAD_WARPS)


def fused_head(yf, emb, tgt_sorted, perm, inv, n_lab, chunk, grad_scale):
    """LM head over the packed tokens without full logits.

    yf [N, D] fp16 final-norm output, emb [V, D] fp16 tied embedding, tgt_sorted [N] int32 targets
    (-1 = none) in the order `perm` (tokens with a target first, n_lab of them), inv its inverse.
    Returns (loss_sum fp32 scalar, stats [N, 4] in sorted order, dY [N, D] fp16 = grad_scale *
    d(sum CE)/d(yf) in token order). Logits exist for `chunk` tokens at a time; supervised chunks
    turn their logits into the scaled softmax gradient in place and one GEMM gives dY."""
    import torch

    n, d = yf.shape
    v = emb.shape[0]
    ys = yf.index_select(0, perm)
    stats = torch.empty((n, 4), dtype=torch.float32, device=yf.device)
    dys = torch.zeros((n, d), dtype=torch.float16, device=yf.device)
    chunk = max(16, int(chunk))
    buf = torch.empty((min(chunk, n), v), dtype=torch.float16, device=yf.device)
    spans = [(s0, min(s0 + chunk, n_lab), 1) for s0 in range(0, n_lab, chunk)]
    spans += [(s0, min(s0 + chunk, n), 0) for s0 in range(n_lab, n, chunk)]
    for s0, s1, grad in spans:
        c = s1 - s0
        logits = buf[:c]
        torch.mm(ys[s0:s1], emb.t(), out=logits)
        head_rows(logits, tgt_sorted, stats, s0, c, grad, grad_scale)
        if grad:
            torch.mm(logits, emb, out=dys[s0:s1])
    del buf, ys
    dy = dys.index_select(0, inv)
    return stats[:, 0].sum(), stats, dy


def warmup_specs():
    import torch
    return [(lm_head_rows_kernel, (torch.float16, torch.int32, torch.float32, 1, 1, 1, 1, 1.0),
             dict(BLOCK_V=HEAD_BLOCK_V, num_warps=HEAD_WARPS))]
