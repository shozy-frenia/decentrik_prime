"""LM head over the 151,936-token vocabulary without full logits: loss, entropy, argmax accuracy,
per-row loss and d(hidden), computed during the forward.

Base Soup materialises the [4, T, 151936] logits several times (fp16 GEMM output, accelerate's
fp32 copy, the fp32 upcast inside the loss, log-softmax for the backward, TRL's chunked entropy,
the argmax for accuracy, per-row cross-entropy for the rewind log): gigabytes of memory and a long
tail of memory-bound kernels. Two implementations live here; both give the same numbers.

1. `streamed_head` (default, v6): a one-pass online softmax over VOCABULARY chunks, i.e.
   flash attention with the hidden states as queries and the embedding as both keys and values.
   For each chunk of CHUNK embedding rows:
       logits_c = Y @ E_c^T                       one cuBLAS GEMM, [N, CHUNK] fp16 (as in the base)
       head_chunk_kernel (one program per row):
         pass 1: chunk max and first-index argmax (torch.argmax tie rule)
         pass 2: s_c = sum exp(x - m_c), t_c = sum exp(x - m_c)(x - m_c), merged into the row's
                 running (max, sum, entropy sum, argmax, target logit); for rows with a target
                 also P = 1024 * exp(x - m_run) written in place (target column set to 0), and
                 the row's fp32 gradient accumulator rescaled by exp(m_old - m_run)
       acc += P @ E_c                             one cuBLAS GEMM, fp16 in, fp32 accumulated in place
   head_finalize_kernel then gives lse = m + log s, entropy = log s - t/s, loss = lse - x_t, the
   accuracy hit, and d(hidden) = acc / (1024 s) + (p_t - 1) * E_t, written straight into token
   order. The target term is added in fp32 from the gathered row E_t instead of through the fp16
   P matrix: for a confident prediction (p_t -> 1) the gradient is a small difference of two
   nearly equal vectors and must not go through an fp16 rounding.
   Only N x CHUNK logits ever exist (7 MB for 900 tokens), and the embedding is read once per step
   in chunks, so it can stay in pinned host memory and stream through a ring of GPU buffers
   (`EmbeddingStream`), one chunk ahead of the GEMMs, instead of occupying 467 MB of GPU memory.
2. `fused_head` (v5): token chunks of 256 rows against the whole vocabulary (`lm_head_rows_kernel`).

Ideas: FlashAttention's online softmax (Dao et al., 2022/2023), the fused linear-cross-entropy of
Liger-Kernel and Apple's Cut Cross-Entropy (arXiv:2411.09009); the vocabulary-streaming
formulation, the in-pass gradient accumulation and the code are ours.
stats row = (loss, entropy, correct, lse) in fp32 in both implementations.
"""

import torch
import triton
import triton.language as tl

HEAD_BLOCK_V = 2048
HEAD_WARPS = 8
CHUNK_BLOCK_V = 2048
CHUNK_WARPS = 8
FIN_BLOCK_D = 2048
FIN_WARPS = 4
STATE_W = 8          # per-row running state: m, s, t, argmax, x_target, (pad)


# --------------------------------------------------------------------------- v6: vocabulary chunks
@triton.jit(do_not_specialize=["n_rows", "n_lab", "v0"])
def head_chunk_kernel(logits_ptr, tgt_ptr, state_ptr, acc_ptr, n_rows, n_lab, vc, v0, p_scale,
                      D: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr):
    """One row of one vocabulary chunk [v0, v0 + vc). `vc` is left to Triton's integer
    specialisation (every chunk width is a multiple of 16), so row starts are known to be aligned
    and the fp16 loads vectorise."""
    r = tl.program_id(0)
    base = logits_ptr + r.to(tl.int64) * vc
    lanes = tl.arange(0, BLOCK_V)
    t = tl.load(tgt_ptr + r)
    tloc = t - v0
    in_chunk = (tloc >= 0) & (tloc < vc)
    # pass 1: chunk max, the first index reaching it, and the target's logit. The target logit is
    # picked out of the pass-1 loads (not re-read with a scalar load later): pass 2 overwrites the
    # row in place, and the reductions below order every pass-1 read before any pass-2 write.
    best = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_i = tl.zeros([BLOCK_V], tl.int32)
    xt_v = tl.zeros([BLOCK_V], tl.float32)
    for c0 in range(0, vc, BLOCK_V):
        cols = c0 + lanes
        x = tl.load(base + cols, mask=cols < vc, other=float("-inf")).to(tl.float32)
        upd = x > best
        best = tl.where(upd, x, best)
        best_i = tl.where(upd, cols, best_i)
        xt_v = tl.where(cols == tloc, x, xt_v)
    mc = tl.max(best, axis=0)
    ac = tl.min(tl.where(best == mc, best_i, 2147483647), axis=0)
    xt_c = tl.sum(xt_v, axis=0)        # exactly x[tloc] (one non-zero lane) when in this chunk
    # merge with the row's running state
    st = state_ptr + r * 8
    m = tl.load(st)
    s = tl.load(st + 1)
    tt = tl.load(st + 2)
    am = tl.load(st + 3)
    xt = tl.load(st + 4)
    m_new = tl.maximum(m, mc)
    a_old = tl.exp(m - m_new)          # 0 on the first chunk (m = -inf)
    a_new = tl.exp(mc - m_new)
    sup = r < n_lab
    # pass 2: chunk sums relative to the chunk max; P for rows with a target
    acc_s = tl.zeros([BLOCK_V], tl.float32)
    acc_t = tl.zeros([BLOCK_V], tl.float32)
    pmul = a_new * p_scale
    for c0 in range(0, vc, BLOCK_V):
        cols = c0 + lanes
        cm = cols < vc
        x = tl.load(base + cols, mask=cm, other=float("-inf")).to(tl.float32)
        z = x - mc
        e = tl.exp(z)
        acc_s += e
        acc_t += tl.where(cm, e * z, 0.0)
        if sup:
            p = tl.where(cols == tloc, 0.0, e * pmul)
            tl.store(base + cols, p.to(tl.float16), mask=cm)
    sc = tl.sum(acc_s, axis=0)
    tc = tl.sum(acc_t, axis=0)
    s_new = s * a_old + sc * a_new
    t_old = tl.where(s > 0.0, a_old * (tt + s * (m - m_new)), 0.0)
    t_new = t_old + a_new * (tc + sc * (mc - m_new))
    am_new = tl.where(mc > m, (ac + v0).to(tl.float32), am)   # ties keep the earlier index
    xt_new = tl.where(in_chunk, xt_c, xt)
    tl.store(st, m_new)
    tl.store(st + 1, s_new)
    tl.store(st + 2, t_new)
    tl.store(st + 3, am_new)
    tl.store(st + 4, xt_new)
    # rescale this row's gradient accumulator to the new running max
    if sup:
        if a_old < 1.0:
            dd = tl.arange(0, BLOCK_D)
            dm = dd < D
            arow = acc_ptr + r.to(tl.int64) * D + dd
            a = tl.load(arow, mask=dm, other=0.0)
            tl.store(arow, a * a_old, mask=dm)


@triton.jit(do_not_specialize=["n_rows", "n_lab"])
def head_finalize_kernel(state_ptr, tgt_ptr, perm_ptr, acc_ptr, et_ptr, stats_ptr, dy_ptr,
                         n_rows, n_lab, out_scale, inv_p_scale,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    """Row r of the sorted order: stats (loss, entropy, hit, lse) and the d(hidden) row of token
    perm[r] (zero for rows without a target), scaled by out_scale, in fp16."""
    r = tl.program_id(0)
    st = state_ptr + r * 8
    m = tl.load(st)
    s = tl.load(st + 1)
    tt = tl.load(st + 2)
    am = tl.load(st + 3)
    xt = tl.load(st + 4)
    log_s = tl.log(s)
    lse = m + log_s
    ent = log_s - tt / s
    t = tl.load(tgt_ptr + r)
    valid = t >= 0
    loss = tl.where(valid, lse - xt, 0.0)
    hit = tl.where(valid & (am == t.to(tl.float32)), 1.0, 0.0)
    out = stats_ptr + r * 4
    tl.store(out, loss)
    tl.store(out + 1, ent)
    tl.store(out + 2, hit)
    tl.store(out + 3, lse)
    dst = tl.load(perm_ptr + r).to(tl.int64)
    dd = tl.arange(0, BLOCK_D)
    dm = dd < D
    if r < n_lab:
        pt = tl.exp(xt - lse)
        a = tl.load(acc_ptr + r.to(tl.int64) * D + dd, mask=dm, other=0.0)
        e_t = tl.load(et_ptr + r.to(tl.int64) * D + dd, mask=dm, other=0.0).to(tl.float32)
        g = a * (inv_p_scale / s) + (pt - 1.0) * e_t
        tl.store(dy_ptr + dst * D + dd, (g * out_scale).to(tl.float16), mask=dm)
    else:
        tl.store(dy_ptr + dst * D + dd, tl.zeros([BLOCK_D], tl.float16), mask=dm)


def _accumulate(acc, p16, e16):
    """acc (fp32) += p16 @ e16 (fp16 inputs, fp32 accumulation, in place): torch's addmm with an
    fp32 out_dtype (cublasGemmEx with an fp32 C). Older torch: fp16 product, fp32 add."""
    op = getattr(getattr(torch.ops.aten, "addmm", None), "dtype_out", None)
    if op is not None and _ACC_OK[0]:
        try:
            op(acc, p16, e16, torch.float32, beta=1, alpha=1, out=acc)
            return
        except (RuntimeError, TypeError):
            _ACC_OK[0] = False
    acc.add_(torch.mm(p16, e16).float())


_ACC_OK = [True]


class ResidentEmbedding:
    """The embedding on the GPU: chunk c is a view of it."""

    def __init__(self, emb, chunk):
        self.emb, self.chunk = emb, int(chunk)
        self.vocab = emb.shape[0]
        self.n_chunks = -(-self.vocab // self.chunk)

    def start(self):
        pass

    def get(self, c):
        v0 = c * self.chunk
        return self.emb[v0:min(v0 + self.chunk, self.vocab)]

    def release(self, c):
        pass

    def rows(self, idx_dev, idx_cpu=None):
        return self.emb.index_select(0, idx_dev)


class EmbeddingStream:
    """The embedding in pinned host memory, streamed through `slots` GPU buffers of `chunk` rows.

    start() queues the H2D copies of the first chunks on the copy stream, gated on the main
    stream reaching that point (the CPU runs layers ahead of the GPU; the copies should overlap the
    last layer, not compete with the weight copies of earlier ones) and on the main stream having
    released each slot in the previous step's head; get(c) makes the main stream wait for chunk c;
    release(c) records that the main stream has issued the last GEMM reading chunk c's slot and
    queues the copy of chunk c + slots into it. One FIFO stream, so chunks arrive in order.
    Nothing else reads these buffers.
    """

    def __init__(self, host_emb, chunk, slots, device):
        assert host_emb.is_pinned(), "the streamed embedding must be in pinned memory"
        self.host, self.chunk, self.k = host_emb, int(chunk), max(2, int(slots))
        self.vocab, self.dim = host_emb.shape
        self.n_chunks = -(-self.vocab // self.chunk)
        self.device = device
        self.bufs = [torch.empty((self.chunk, self.dim), dtype=host_emb.dtype, device=device)
                     for _ in range(self.k)]
        self.loaded = [None] * self.k
        self.freed = [None] * self.k
        self.queued = set()
        self.stream = torch.cuda.Stream(device=device)
        self.copies = 0

    def _issue(self, c, gate=None):
        if c >= self.n_chunks or c in self.queued:
            return
        k = c % self.k
        v0 = c * self.chunk
        vc = min(self.chunk, self.vocab - v0)
        with torch.cuda.stream(self.stream):
            if gate is not None:
                self.stream.wait_event(gate)
            if self.freed[k] is not None:
                self.stream.wait_event(self.freed[k])
            self.bufs[k][:vc].copy_(self.host[v0:v0 + vc], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.loaded[k] = done
        self.queued.add(c)
        self.copies += 1

    def start(self):
        gate = torch.cuda.Event()
        gate.record(torch.cuda.current_stream(self.device))
        for c in range(min(self.k, self.n_chunks)):
            self._issue(c, gate)

    def get(self, c):
        self._issue(c)
        k = c % self.k
        torch.cuda.current_stream(self.device).wait_event(self.loaded[k])
        vc = min(self.chunk, self.vocab - c * self.chunk)
        return self.bufs[k][:vc]

    def release(self, c):
        k = c % self.k
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        self.freed[k] = ev
        self.queued.discard(c)
        self._issue(c + self.k)
        if c == self.n_chunks - 1:
            self.queued.clear()

    def rows(self, idx_dev, idx_cpu):
        """Rows of the embedding for the given (CPU) indices: gathered on the CPU from pinned
        memory into a pinned staging tensor, copied to the GPU asynchronously."""
        stage = torch.empty((idx_cpu.numel(), self.dim), dtype=self.host.dtype, pin_memory=True)
        torch.index_select(self.host, 0, idx_cpu.to(torch.int64), out=stage)
        return stage.to(self.device, non_blocking=True)


def streamed_head(yf, source, tgt_sorted, perm, n_lab, e_target, grad_scale, dummy32):
    """LM head of the packed batch (see the module docstring).

    yf [N, D] fp16 final-norm output in token order; source: ResidentEmbedding or EmbeddingStream;
    tgt_sorted [N] int32 targets (-1 = none) in the order `perm` (tokens with a target first,
    n_lab of them); e_target [n_lab, D] fp16 embedding rows of those targets.
    Returns (loss_sum fp32 scalar, stats [N, 4] in sorted order, dY [N, D] fp16 = grad_scale *
    d(sum CE)/d(yf) in token order)."""
    n, d = yf.shape
    dev = yf.device
    ys = yf.index_select(0, perm)
    state = torch.zeros((n, STATE_W), dtype=torch.float32, device=dev)
    state[:, 0] = float("-inf")
    acc = torch.zeros((n_lab, d), dtype=torch.float32, device=dev) if n_lab > 0 else dummy32
    buf = torch.empty((n * source.chunk,), dtype=torch.float16, device=dev)
    for c in range(source.n_chunks):
        e_c = source.get(c)
        vc = e_c.shape[0]
        logits = buf[:n * vc].view(n, vc)
        torch.mm(ys, e_c.t(), out=logits)
        head_chunk_kernel[(n,)](logits, tgt_sorted, state, acc, n, n_lab, vc, c * source.chunk,
                                float(grad_scale), D=d, BLOCK_V=CHUNK_BLOCK_V,
                                BLOCK_D=triton.next_power_of_2(d), num_warps=CHUNK_WARPS)
        if n_lab > 0:
            _accumulate(acc, logits[:n_lab], e_c)
        source.release(c)
    del buf, ys
    stats = torch.empty((n, 4), dtype=torch.float32, device=dev)
    dy = torch.empty((n, d), dtype=torch.float16, device=dev)
    et = e_target if n_lab > 0 else torch.zeros(16, dtype=torch.float16, device=dev)
    head_finalize_kernel[(n,)](state, tgt_sorted, perm, acc, et, stats, dy, n, n_lab,
                               float(grad_scale), 1.0 / float(grad_scale), D=d,
                               BLOCK_D=triton.next_power_of_2(d), num_warps=FIN_WARPS)
    return stats[:, 0].sum(), stats, dy


# --------------------------------------------------------------------------- v5: token chunks
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
    # pass 2: softmax statistics, and the target's logit picked out of the same loads (pass 3
    # overwrites the row in place; the reductions order these reads before its writes)
    acc_s = tl.zeros([BLOCK_V], tl.float32)
    acc_t = tl.zeros([BLOCK_V], tl.float32)
    xt_v = tl.zeros([BLOCK_V], tl.float32)
    for v0 in range(0, vocab, BLOCK_V):
        cols = v0 + lanes
        cm = cols < vocab
        x = tl.load(base + cols, mask=cm, other=float("-inf")).to(tl.float32)
        z = x - m
        e = tl.exp(z)
        acc_s += e
        acc_t += tl.where(cm, e * z, 0.0)
        xt_v = tl.where(cols == t, x, xt_v)
    s = tl.sum(acc_s, axis=0)
    log_s = tl.log(s)
    lse = m + log_s
    ent = log_s - tl.sum(acc_t, axis=0) / s
    valid = t >= 0
    xt = tl.sum(xt_v, axis=0)
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
    """v5 LM head: token chunks of `chunk` rows against the whole (GPU-resident) vocabulary.

    yf [N, D] fp16 final-norm output, emb [V, D] fp16 tied embedding, tgt_sorted [N] int32 targets
    (-1 = none) in the order `perm` (tokens with a target first, n_lab of them), inv its inverse.
    Returns (loss_sum fp32 scalar, stats [N, 4] in sorted order, dY [N, D] fp16 = grad_scale *
    d(sum CE)/d(yf) in token order)."""
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


def warmup_specs(hidden=1536, chunk=4096, impl="chunks"):
    f16, f32, i32 = torch.float16, torch.float32, torch.int32
    bd = triton.next_power_of_2(hidden)
    if impl == "rows":
        return [(lm_head_rows_kernel, (f16, i32, f32, 1, 1, 1, 1, 1.0),
                 dict(BLOCK_V=HEAD_BLOCK_V, num_warps=HEAD_WARPS))]
    return [
        # vc is specialised by Triton (multiple of 16): warm up with a representative width
        (head_chunk_kernel, (f16, i32, f32, f32, 1, 1, int(chunk), 1, 1024.0),
         dict(D=hidden, BLOCK_V=CHUNK_BLOCK_V, BLOCK_D=bd, num_warps=CHUNK_WARPS)),
        (head_finalize_kernel, (f32, i32, i32, f32, f16, f32, f16, 1, 1, 1024.0, 1.0 / 1024.0),
         dict(D=hidden, BLOCK_D=bd, num_warps=FIN_WARPS)),
    ]
