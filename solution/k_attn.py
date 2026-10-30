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


def attention_forward(qkv, meta, nq, nkv, hd):
    n = meta.n_tokens
    out = torch.empty((n, nq * hd), dtype=torch.float16, device=qkv.device)
    lse = torch.empty((nq, n), dtype=torch.float32, device=qkv.device)
    sm_scale = 1.0 / math.sqrt(hd)
    attn_fwd_kernel[(meta.n_fwd, nq)](qkv, out, lse, meta.cu, meta.tab_fwd, n, sm_scale * LOG2E,
                                      NQ=nq, NKV=nkv, HD=hd, BLOCK_M=FWD_BLOCK_M,
                                      BLOCK_N=FWD_BLOCK_N, num_warps=ATTN_WARPS)
    return out, lse


def attention_backward(qkv, out, lse, dout, meta, nq, nkv, hd):
    """Returns fp32 dq [N, nq*hd] and per-query-head dk, dv partials [N, nq*hd]."""
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
                                           BLOCK=BWD_BLOCK, num_warps=ATTN_WARPS)
    attn_bwd_dq_kernel[(meta.n_bwd, nq)](qkv, dout, lse, delta, dq, meta.cu, meta.tab_bwd,
                                         n, sm_scale * LOG2E, sm_scale, NQ=nq, NKV=nkv, HD=hd,
                                         BLOCK=BWD_BLOCK, num_warps=ATTN_WARPS)
    return dq, dkp, dvp


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
