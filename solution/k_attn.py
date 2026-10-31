"""Variable-length causal GQA flash attention for the padding-free packed batch, forward + backward.

The batch is packed: the real tokens of the four rows are concatenated (no padding), the queries,
keys and values of a token live in one row of the fused QKV buffer [N, (12+2+2)*128] fp16 written
by the QKV epilogue. A tiny int32 table maps each program to (sequence, first row of its block):
blocks never straddle two sequences, so the block-diagonal causal mask is just `m >= n` inside a
sequence, and no attention is ever computed between rows of different examples or with padding.

Forward (one program per (query block, query head)): online softmax in exp2 domain with fp32
running max/sum (FlashAttention-2 schedule), output fp16 [N, 12*128] and the log2-sum-exp per
(head, token) for the backward pass. GQA: query head h reads key/value head h // 6 directly, no
repeat_kv copy.

Backward:
  delta  : D_i = sum_d O_id dO_id                                  (one program per token)
  dk/dv  : one program per (key block, query head); loops over the query blocks at or after it,
           recomputes P^T from Q, K and the saved lse; writes fp32 per-query-head partials which
           the QKV epilogue sums over each group of 6 (keeps 12x more parallelism than one program
           per key/value head on a 40-SM T4)
  dq     : one program per (query block, query head); loops over the key blocks up to the diagonal
Block sizes fit sm_75 (64 KB shared memory, no cp.async pipelining): forward 64x32, backward 32x32.

Algorithm: FlashAttention-2 (Dao, 2023, https://arxiv.org/abs/2307.08691) and the structure of
Triton's fused-attention tutorial (https://triton-lang.org/main/getting-started/tutorials/
06-fused-attention.html); the varlen block tables, GQA partial reduction and code are ours.
"""

import math

import torch
import triton
import triton.language as tl

FWD_BLOCK_M, FWD_BLOCK_N = 64, 32
BWD_BLOCK = 32
ATTN_WARPS = 4
LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["n_tokens"])
def attn_fwd_kernel(qkv_ptr, out_ptr, lse_ptr, cu_ptr, tab_ptr, n_tokens, qk_scale,
                    NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    GROUP: tl.constexpr = NQ // NKV
    seq = tl.load(tab_ptr + 2 * pid)
    mstart = tl.load(tab_ptr + 2 * pid + 1)
    s0 = tl.load(cu_ptr + seq)
    seqlen = tl.load(cu_ptr + seq + 1) - s0
    kvh = h // GROUP
    offs_m = mstart + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    mrow = offs_m < seqlen
    q = tl.load(qkv_ptr + (s0 + offs_m)[:, None] * ROW + h * HD + offs_d[None, :],
                mask=mrow[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    hi = tl.minimum(mstart + BLOCK_M, seqlen)
    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < seqlen
        kt = tl.load(qkv_ptr + (s0 + offs_n)[None, :] * ROW + (NQ + kvh) * HD + offs_d[:, None],
                     mask=nmask[None, :], other=0.0)
        sc = tl.dot(q, kt) * qk_scale
        keep = (offs_m[:, None] >= offs_n[None, :]) & nmask[None, :]
        sc = tl.where(keep, sc, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        p = tl.exp2(sc - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(qkv_ptr + (s0 + offs_n)[:, None] * ROW + (NQ + NKV + kvh) * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(out_ptr + (s0 + offs_m)[:, None] * (NQ * HD) + h * HD + offs_d[None, :],
             acc.to(tl.float16), mask=mrow[:, None])
    tl.store(lse_ptr + h * n_tokens + s0 + offs_m, m_i + tl.log2(l_i), mask=mrow)


@triton.jit(do_not_specialize=["n_tokens"])
def attn_bwd_delta_kernel(o_ptr, do_ptr, delta_ptr, n_tokens,
                          NQ: tl.constexpr, HD: tl.constexpr, NQ_P2: tl.constexpr):
    row = tl.program_id(0)
    hh = tl.arange(0, NQ_P2)
    d = tl.arange(0, HD)
    hm = hh < NQ
    idx = row * (NQ * HD) + hh[:, None] * HD + d[None, :]
    o = tl.load(o_ptr + idx, mask=hm[:, None], other=0.0).to(tl.float32)
    g = tl.load(do_ptr + idx, mask=hm[:, None], other=0.0).to(tl.float32)
    tl.store(delta_ptr + hh * n_tokens + row, tl.sum(o * g, axis=1), mask=hm)


@triton.jit(do_not_specialize=["n_tokens"])
def attn_bwd_dkdv_kernel(qkv_ptr, do_ptr, lse_ptr, delta_ptr, dkp_ptr, dvp_ptr, cu_ptr, tab_ptr,
                         n_tokens, qk_scale, sm_scale,
                         NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    GROUP: tl.constexpr = NQ // NKV
    seq = tl.load(tab_ptr + 2 * pid)
    nstart = tl.load(tab_ptr + 2 * pid + 1)
    s0 = tl.load(cu_ptr + seq)
    seqlen = tl.load(cu_ptr + seq + 1) - s0
    kvh = h // GROUP
    offs_n = nstart + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HD)
    nmask = offs_n < seqlen
    k = tl.load(qkv_ptr + (s0 + offs_n)[:, None] * ROW + (NQ + kvh) * HD + offs_d[None, :],
                mask=nmask[:, None], other=0.0)
    v = tl.load(qkv_ptr + (s0 + offs_n)[:, None] * ROW + (NQ + NKV + kvh) * HD + offs_d[None, :],
                mask=nmask[:, None], other=0.0)
    dk = tl.zeros([BLOCK, HD], tl.float32)
    dv = tl.zeros([BLOCK, HD], tl.float32)
    for m0 in range(nstart, seqlen, BLOCK):
        offs_m = m0 + tl.arange(0, BLOCK)
        mmask = offs_m < seqlen
        q = tl.load(qkv_ptr + (s0 + offs_m)[:, None] * ROW + h * HD + offs_d[None, :],
                    mask=mmask[:, None], other=0.0)
        do = tl.load(do_ptr + (s0 + offs_m)[:, None] * (NQ * HD) + h * HD + offs_d[None, :],
                     mask=mmask[:, None], other=0.0)
        lse = tl.load(lse_ptr + h * n_tokens + s0 + offs_m, mask=mmask, other=0.0)
        dlt = tl.load(delta_ptr + h * n_tokens + s0 + offs_m, mask=mmask, other=0.0)
        st = tl.dot(k, tl.trans(q)) * qk_scale                         # [BLOCK_N, BLOCK_M]
        keep = (offs_m[None, :] >= offs_n[:, None]) & mmask[None, :]
        pt = tl.where(keep, tl.exp2(st - lse[None, :]), 0.0)
        dv += tl.dot(pt.to(tl.float16), do)
        dpt = tl.dot(v, tl.trans(do))
        dst = pt * (dpt - dlt[None, :])
        dk += tl.dot(dst.to(tl.float16), q)
    dk = dk * sm_scale
    out = (s0 + offs_n)[:, None] * (NQ * HD) + h * HD + offs_d[None, :]
    tl.store(dkp_ptr + out, dk, mask=nmask[:, None])
    tl.store(dvp_ptr + out, dv, mask=nmask[:, None])


@triton.jit(do_not_specialize=["n_tokens"])
def attn_bwd_dq_kernel(qkv_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr, cu_ptr, tab_ptr,
                       n_tokens, qk_scale, sm_scale,
                       NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    GROUP: tl.constexpr = NQ // NKV
    seq = tl.load(tab_ptr + 2 * pid)
    mstart = tl.load(tab_ptr + 2 * pid + 1)
    s0 = tl.load(cu_ptr + seq)
    seqlen = tl.load(cu_ptr + seq + 1) - s0
    kvh = h // GROUP
    offs_m = mstart + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HD)
    mmask = offs_m < seqlen
    q = tl.load(qkv_ptr + (s0 + offs_m)[:, None] * ROW + h * HD + offs_d[None, :],
                mask=mmask[:, None], other=0.0)
    do = tl.load(do_ptr + (s0 + offs_m)[:, None] * (NQ * HD) + h * HD + offs_d[None, :],
                 mask=mmask[:, None], other=0.0)
    lse = tl.load(lse_ptr + h * n_tokens + s0 + offs_m, mask=mmask, other=0.0)
    dlt = tl.load(delta_ptr + h * n_tokens + s0 + offs_m, mask=mmask, other=0.0)
    dq = tl.zeros([BLOCK, HD], tl.float32)
    hi = tl.minimum(mstart + BLOCK, seqlen)
    for n0 in range(0, hi, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        nmask = offs_n < seqlen
        k = tl.load(qkv_ptr + (s0 + offs_n)[:, None] * ROW + (NQ + kvh) * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        v = tl.load(qkv_ptr + (s0 + offs_n)[:, None] * ROW + (NQ + NKV + kvh) * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * qk_scale                          # [BLOCK_M, BLOCK_N]
        keep = (offs_m[:, None] >= offs_n[None, :]) & nmask[None, :] & mmask[:, None]
        p = tl.where(keep, tl.exp2(sc - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dlt[:, None])
        dq += tl.dot(ds.to(tl.float16), k)
    dq = dq * sm_scale
    tl.store(dq_ptr + (s0 + offs_m)[:, None] * (NQ * HD) + h * HD + offs_d[None, :], dq,
             mask=mmask[:, None])


def block_table(lens, block):
    """Flat [seq, start, seq, start, ...] for every block of `block` rows of every sequence."""
    tab = []
    for s, n in enumerate(lens):
        for start in range(0, n, block):
            tab += (s, start)
    return tab


class AttnMeta:
    """Per-batch tables (device int32 views) for the attention kernels."""

    def __init__(self, cu, tab_fwd, tab_bwd, n_fwd, n_bwd, n_tokens):
        self.cu, self.tab_fwd, self.tab_bwd = cu, tab_fwd, tab_bwd
        self.n_fwd, self.n_bwd, self.n_tokens = n_fwd, n_bwd, n_tokens


def attention_forward(qkv, meta, nq, nkv, hd, block_m=FWD_BLOCK_M, block_n=FWD_BLOCK_N,
                      warps=ATTN_WARPS):
    """meta.tab_fwd must be the block table for `block_m`."""
    n = meta.n_tokens
    out = torch.empty((n, nq * hd), dtype=torch.float16, device=qkv.device)
    lse = torch.empty((nq, n), dtype=torch.float32, device=qkv.device)
    sm_scale = 1.0 / math.sqrt(hd)
    attn_fwd_kernel[(meta.n_fwd, nq)](qkv, out, lse, meta.cu, meta.tab_fwd, n, sm_scale * LOG2E,
                                      NQ=nq, NKV=nkv, HD=hd, BLOCK_M=block_m,
                                      BLOCK_N=block_n, num_warps=warps)
    return out, lse


def attention_backward(qkv, out, lse, dout, meta, nq, nkv, hd, block=BWD_BLOCK, warps=ATTN_WARPS):
    """Returns fp32 dq [N, nq*hd] and per-query-head dk, dv partials [N, nq*hd].
    meta.tab_bwd must be the block table for `block`."""
    n = meta.n_tokens
    dev = qkv.device
    sm_scale = 1.0 / math.sqrt(hd)
    delta = torch.empty((nq, n), dtype=torch.float32, device=dev)
    attn_bwd_delta_kernel[(n,)](out, dout, delta, n, NQ=nq, HD=hd,
                                NQ_P2=triton.next_power_of_2(nq), num_warps=ATTN_WARPS)
    dq = torch.empty((n, nq * hd), dtype=torch.float32, device=dev)
    dkp = torch.empty((n, nq * hd), dtype=torch.float32, device=dev)
    dvp = torch.empty((n, nq * hd), dtype=torch.float32, device=dev)
    attn_bwd_dkdv_kernel[(meta.n_bwd, nq)](qkv, dout, lse, delta, dkp, dvp, meta.cu, meta.tab_bwd,
                                           n, sm_scale * LOG2E, sm_scale, NQ=nq, NKV=nkv, HD=hd,
                                           BLOCK=block, num_warps=warps)
    attn_bwd_dq_kernel[(meta.n_bwd, nq)](qkv, dout, lse, delta, dq, meta.cu, meta.tab_bwd,
                                         n, sm_scale * LOG2E, sm_scale, NQ=nq, NKV=nkv, HD=hd,
                                         BLOCK=block, num_warps=warps)
    return dq, dkp, dvp


# --------------------------------------------------------------------------- varlen path
# Why not our Triton kernels above: Triton 3.6 lowers tl.dot to tensor cores (MMA) only on
# compute capability >= 8.0; on the T4 (sm_75) AccelerateMatmul emits "Dot op using MMA for
# compute capability 75 has been deprecated. It falls back to the FMA path", so every tl.dot
# runs on the fp32 FMA units (~8 TFLOP/s peak instead of 65). That is the ~40x we measured.
# The attention therefore uses the tensor-core kernel PyTorch itself ships for sm_75 (the
# memory-efficient attention behind SDPA, the one base Soup runs), called the way PyTorch's own
# nested-tensor SDPA calls it: on the packed tokens with cu_seqlens, so the four rows are four
# independent causal sequences and no padding is computed, zero-filled, scattered or gathered.
_VARLEN_ERR = []


def _call_by_name(op, values):
    """Call an aten overload with keyword arguments taken from its schema (argument lists differ
    between torch versions). Unknown keyword-only arguments keep their defaults; an unknown
    positional argument means a signature we do not know, so we refuse."""
    kwargs = {}
    for arg in op._schema.arguments:
        if arg.name in values:
            kwargs[arg.name] = values[arg.name]
        elif not arg.kwarg_only:
            raise RuntimeError(f"{op} has an unexpected argument {arg.name}")
    return op(**kwargs)


def varlen_attention(qkv, cu, max_len, nq, nkv, hd, need_lse=False):
    """Causal GQA attention of the packed batch through aten::_efficient_attention_forward with
    cu_seqlens (custom_mask_type 1 = causal from the top-left of each sequence). Q is a strided
    view of the fused QKV buffer; K/V are expanded to the query heads (the kernel wants equal head
    counts, like repeat_kv). Returns (out [N, nq*hd] fp16, context for the backward or None)."""
    n = qkv.shape[0]
    g = nq // nkv
    q = qkv[:, :nq * hd].view(1, n, nq, hd)
    k = qkv[:, nq * hd:(nq + nkv) * hd].view(n, nkv, 1, hd).expand(n, nkv, g, hd).reshape(1, n, nq, hd)
    v = qkv[:, (nq + nkv) * hd:].view(n, nkv, 1, hd).expand(n, nkv, g, hd).reshape(1, n, nq, hd)
    res = _call_by_name(torch.ops.aten._efficient_attention_forward.default, dict(
        query=q, key=k, value=v, bias=None, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=int(max_len), max_seqlen_k=int(max_len), dropout_p=0.0, custom_mask_type=1,
        compute_log_sumexp=bool(need_lse), scale=1.0 / math.sqrt(hd)))
    o = res[0]
    out = o.reshape(n, nq * hd)
    if not out.is_contiguous():
        out = out.contiguous()
    ctx = (q, k, v, o, res[1], res[2], res[3]) if need_lse else None
    return out, ctx


def varlen_context(qkv, out, saved, nq, nkv, hd):
    """The backward context of varlen_attention rebuilt from the forward's QKV buffer, output and
    the kernel's (logsumexp, philox seed, philox offset), when those were kept from the forward
    instead of recomputing the attention."""
    n = qkv.shape[0]
    g = nq // nkv
    q = qkv[:, :nq * hd].view(1, n, nq, hd)
    k = qkv[:, nq * hd:(nq + nkv) * hd].view(n, nkv, 1, hd).expand(n, nkv, g, hd).reshape(1, n, nq, hd)
    v = qkv[:, (nq + nkv) * hd:].view(n, nkv, 1, hd).expand(n, nkv, g, hd).reshape(1, n, nq, hd)
    return (q, k, v, out.view(1, n, nq, hd)) + tuple(saved)


def varlen_attention_backward(ctx, dout, cu, max_len, nq, nkv, hd):
    """(dq [N, nq*hd], dk [N, nq*hd], dv [N, nq*hd]) fp16; dk/dv are per-query-head partials
    (6 per KV head) that the QKV epilogue backward sums (GIN = nq // nkv)."""
    q, k, v, o, lse, seed, offset = ctx
    n = dout.shape[0]
    grads = _call_by_name(torch.ops.aten._efficient_attention_backward.default, dict(
        grad_out_=dout.view(1, n, nq, hd), query=q, key=k, value=v, bias=None, out=o,
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=int(max_len), max_seqlen_k=int(max_len),
        logsumexp=lse, dropout_p=0.0, philox_seed=seed, philox_offset=offset, custom_mask_type=1,
        bias_requires_grad=False, scale=1.0 / math.sqrt(hd), shared_storage_dqdkdv=False))
    out = []
    for t in grads[:3]:
        t = t.reshape(n, nq * hd)
        out.append(t if t.is_contiguous() else t.contiguous())
    return tuple(out)


def varlen_selftest(device, nq=12, nkv=2, hd=128, lens=(37, 70, 5, 64)):
    """Compare the varlen path with the padded SDPA path on a small random batch (forward and
    backward). Returns (ok, worst relative error); any exception counts as failure."""
    try:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        n, B, T = sum(lens), len(lens), max(lens)
        qkv = (torch.randn(n, (nq + 2 * nkv) * hd, generator=gen) * 0.7).half().to(device)
        dout = torch.randn(n, nq * hd, generator=gen).half().to(device)
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32,
                          device=device)
        row = torch.cat([torch.full((L,), b) for b, L in enumerate(lens)]).to(device)
        pos = torch.cat([torch.arange(L) for L in lens]).to(device)
        ref_out, rctx = sdpa_attention(qkv, row, pos, B, T, nq, nkv, hd, grad=True)
        rq, rk, rv = sdpa_attention_backward(rctx, dout, row, pos, B, T, nq, nkv, hd)
        out, ctx = varlen_attention(qkv, cu, T, nq, nkv, hd, need_lse=True)
        out2, _ = varlen_attention(qkv, cu, T, nq, nkv, hd, need_lse=False)
        dq, dk, dv = varlen_attention_backward(ctx, dout, cu, T, nq, nkv, hd)
        g = nq // nkv
        dk = dk.float().view(n, nkv, g, hd).sum(2).reshape(n, nkv * hd)
        dv = dv.float().view(n, nkv, g, hd).sum(2).reshape(n, nkv * hd)

        def rel(a, b):
            a, b = a.float(), b.float()
            return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))

        errs = [rel(out, ref_out), rel(out2, ref_out), rel(dq, rq), rel(dk, rk), rel(dv, rv)]
        worst = max(errs)
        return bool(worst < 2e-2 and all(math.isfinite(e) for e in errs)), worst
    except Exception as exc:  # noqa: BLE001
        _VARLEN_ERR.append(f"{type(exc).__name__}: {exc}")
        return False, float("nan")


# --------------------------------------------------------------------------- SDPA path
def sdpa_attention(qkv, b_idx, t_idx, B, T, nq, nkv, hd, grad=False):
    """PyTorch SDPA (memory-efficient kernel on sm_75, what base Soup runs) on zero-padded
    [B, H, T, hd] buffers written straight from the packed tokens: one index_put per tensor, K/V
    expanded to the query heads through a broadcast view (like repeat_kv). Right padding needs
    only the causal mask. With grad=True the call records a small autograd graph rooted at the
    packed q/k/v views, so sdpa_attention_backward gets packed, GQA-reduced gradients.
    Returns (out [N, nq*hd] fp16, context)."""
    n = qkv.shape[0]
    g = nq // nkv
    q_p = qkv[:, :nq * hd].reshape(n, nq, hd).detach().requires_grad_(grad)
    k_p = qkv[:, nq * hd:(nq + nkv) * hd].reshape(n, nkv, hd).detach().requires_grad_(grad)
    v_p = qkv[:, (nq + nkv) * hd:].reshape(n, nkv, hd).detach().requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        qb = qkv.new_zeros((B, nq, T, hd))
        qb.permute(0, 2, 1, 3)[b_idx, t_idx] = q_p
        kb = qkv.new_zeros((B, nq, T, hd))
        kb.view(B, nkv, g, T, hd).permute(0, 3, 1, 2, 4)[b_idx, t_idx] = k_p[:, :, None].expand(n, nkv, g, hd)
        vb = qkv.new_zeros((B, nq, T, hd))
        vb.view(B, nkv, g, T, hd).permute(0, 3, 1, 2, 4)[b_idx, t_idx] = v_p[:, :, None].expand(n, nkv, g, hd)
        o = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, is_causal=True,
                                                             scale=1.0 / math.sqrt(hd))
    out = o.detach().permute(0, 2, 1, 3)[b_idx, t_idx].reshape(n, nq * hd)
    return out, ((o, q_p, k_p, v_p) if grad else None)


def sdpa_attention_backward(ctx, dout, b_idx, t_idx, B, T, nq, nkv, hd):
    """(dq [N, nq*hd], dk [N, nkv*hd], dv [N, nkv*hd]) fp16, packed."""
    o, q_p, k_p, v_p = ctx
    n = dout.shape[0]
    dpad = dout.new_zeros((B, nq, T, hd))
    dpad.permute(0, 2, 1, 3)[b_idx, t_idx] = dout.view(n, nq, hd)
    dq, dk, dv = torch.autograd.grad(o, (q_p, k_p, v_p), dpad)
    return (dq.reshape(n, nq * hd).contiguous(), dk.reshape(n, nkv * hd).contiguous(),
            dv.reshape(n, nkv * hd).contiguous())


def warmup_specs(nq, nkv, hd):
    f32, f16, i32 = torch.float32, torch.float16, torch.int32
    common = dict(NQ=nq, NKV=nkv, HD=hd, num_warps=ATTN_WARPS)
    return [
        (attn_fwd_kernel, (f16, f16, f32, i32, i32, 1, 0.1),
         dict(common, BLOCK_M=FWD_BLOCK_M, BLOCK_N=FWD_BLOCK_N)),
        (attn_bwd_delta_kernel, (f16, f16, f32, 1),
         dict(NQ=nq, HD=hd, NQ_P2=triton.next_power_of_2(nq), num_warps=ATTN_WARPS)),
        (attn_bwd_dkdv_kernel, (f16, f16, f32, f32, f32, f32, i32, i32, 1, 0.1, 0.1),
         dict(common, BLOCK=BWD_BLOCK)),
        (attn_bwd_dq_kernel, (f16, f16, f32, f32, f32, i32, i32, 1, 0.1, 0.1),
         dict(common, BLOCK=BWD_BLOCK)),
    ]
