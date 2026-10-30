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


def warmup_specs():
    import torch
    return [(lm_head_rows_kernel, (torch.float16, torch.int32, torch.float32, 1, 1, 1, 1, 1.0),
             dict(BLOCK_V=HEAD_BLOCK_V, num_warps=HEAD_WARPS))]
