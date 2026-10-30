"""QKV epilogue: LoRA add + rotary embedding, in place on the fused [N, (12+2+2)*128] QKV buffer.

Forward, one program per token (all 16 heads of the row):
    q32 = f32(q16) + scale * f32(lq16)            PEFT: base output (fp32 view of the fp16 GEMM) + lora
    k32 = f32(k16)
    v32 = f32(v16) + scale * f32(lv16)
    q/k <- rope(q32/k32) in fp32: [x1*c - x2*s, x2*c + x1*s] (= x*cos + rotate_half(x)*sin)
    everything is stored back as fp16 (what SDPA's autocast cast would hand to attention).
Backward reads the attention grads (dq, and dk/dv either already reduced per KV head or as
per-query-head partials that it sums over each GQA group), applies the transposed rotation and
writes fp16 grads of the pre-rotation q/k/v in the QKV buffer layout, ready for the fused QKV
weight-grad GEMM:
    dx1 = d1*c + d2*s,  dx2 = d2*c - d1*s.
Launched with enable_fp_fusion=False so every product/sum rounds like the separate PyTorch ops.
"""

import triton
import triton.language as tl

ROPE_WARPS = 4


@triton.jit(do_not_specialize=["n_rows"])
def qkv_epilogue_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, pos_ptr, cos_ptr, sin_ptr, n_rows, lora_scale,
                            NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                            NQ_P2: tl.constexpr):
    row = tl.program_id(0)
    HALF: tl.constexpr = HD // 2
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    p = tl.load(pos_ptr + row)
    j = tl.arange(0, HALF)
    c = tl.load(cos_ptr + p * HD + j)[None, :]
    s = tl.load(sin_ptr + p * HD + j)[None, :]
    base = qkv_ptr + row * ROW
    # query heads: LoRA add, then rotate
    hq = tl.arange(0, NQ_P2)
    qm = (hq < NQ)[:, None]
    o1 = hq[:, None] * HD + j[None, :]
    q1 = tl.load(base + o1, mask=qm, other=0.0).to(tl.float32)
    q2 = tl.load(base + o1 + HALF, mask=qm, other=0.0).to(tl.float32)
    lrow = lq_ptr + row * (NQ * HD)
    q1 = q1 + tl.load(lrow + o1, mask=qm, other=0.0).to(tl.float32) * lora_scale
    q2 = q2 + tl.load(lrow + o1 + HALF, mask=qm, other=0.0).to(tl.float32) * lora_scale
    tl.store(base + o1, (q1 * c - q2 * s).to(tl.float16), mask=qm)
    tl.store(base + o1 + HALF, (q2 * c + q1 * s).to(tl.float16), mask=qm)
    # key heads: rotate
    hk = tl.arange(0, NKV)
    ok = NQ * HD + hk[:, None] * HD + j[None, :]
    k1 = tl.load(base + ok).to(tl.float32)
    k2 = tl.load(base + ok + HALF).to(tl.float32)
    tl.store(base + ok, (k1 * c - k2 * s).to(tl.float16))
    tl.store(base + ok + HALF, (k2 * c + k1 * s).to(tl.float16))
    # value heads: LoRA add
    dv = tl.arange(0, NKV * HD)
    ov = (NQ + NKV) * HD + dv
    v = tl.load(base + ov).to(tl.float32)
    v = v + tl.load(lv_ptr + row * (NKV * HD) + dv).to(tl.float32) * lora_scale
    tl.store(base + ov, v.to(tl.float16))


@triton.jit(do_not_specialize=["n_rows"])
def qkv_epilogue_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, pos_ptr, cos_ptr, sin_ptr, out_ptr, n_rows,
                            NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                            NQ_P2: tl.constexpr, GIN: tl.constexpr, GIN_P2: tl.constexpr):
    """dk/dv rows hold GIN partials per KV head ([N, NKV*GIN*HD], summed here): GIN = 6 for the
    per-query-head partials of our attention kernel, GIN = 1 for SDPA's reduced grads."""
    row = tl.program_id(0)
    HALF: tl.constexpr = HD // 2
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    p = tl.load(pos_ptr + row)
    j = tl.arange(0, HALF)
    c = tl.load(cos_ptr + p * HD + j)
    s = tl.load(sin_ptr + p * HD + j)
    dst = out_ptr + row * ROW
    # query heads
    src = row * (NQ * HD)
    hq = tl.arange(0, NQ_P2)
    qm = (hq < NQ)[:, None]
    o1 = hq[:, None] * HD + j[None, :]
    d1 = tl.load(dq_ptr + src + o1, mask=qm, other=0.0).to(tl.float32)
    d2 = tl.load(dq_ptr + src + o1 + HALF, mask=qm, other=0.0).to(tl.float32)
    tl.store(dst + o1, (d1 * c[None, :] + d2 * s[None, :]).to(tl.float16), mask=qm)
    tl.store(dst + o1 + HALF, (d2 * c[None, :] - d1 * s[None, :]).to(tl.float16), mask=qm)
    # key heads: sum the partials of each KV head, then rotate back
    skv = row * (NKV * GIN * HD)
    hk = tl.arange(0, NKV)
    g = tl.arange(0, GIN_P2)
    gm = (g < GIN)[None, :, None]
    part = (hk[:, None, None] * GIN + g[None, :, None]) * HD + j[None, None, :]       # [NKV, G, HALF]
    k1 = tl.sum(tl.load(dk_ptr + skv + part, mask=gm, other=0.0).to(tl.float32), axis=1)
    k2 = tl.sum(tl.load(dk_ptr + skv + part + HALF, mask=gm, other=0.0).to(tl.float32), axis=1)
    ok = NQ * HD + hk[:, None] * HD + j[None, :]
    tl.store(dst + ok, (k1 * c[None, :] + k2 * s[None, :]).to(tl.float16))
    tl.store(dst + ok + HALF, (k2 * c[None, :] - k1 * s[None, :]).to(tl.float16))
    # value heads: sum the partials
    dvo = tl.arange(0, HD)
    vpart = (hk[:, None, None] * GIN + g[None, :, None]) * HD + dvo[None, None, :]    # [NKV, G, HD]
    vsum = tl.sum(tl.load(dv_ptr + skv + vpart, mask=gm, other=0.0).to(tl.float32), axis=1)
    ov = (NQ + NKV) * HD + hk[:, None] * HD + dvo[None, :]
    tl.store(dst + ov, vsum.to(tl.float16))


def qkv_epilogue_forward(qkv, lq, lv, pos, cos, sin, lora_scale, nq, nkv, hd):
    n = qkv.shape[0]
    qkv_epilogue_fwd_kernel[(n,)](qkv, lq, lv, pos, cos, sin, n, float(lora_scale),
                                  NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq),
                                  num_warps=ROPE_WARPS, enable_fp_fusion=False)


def qkv_epilogue_backward(dq, dk, dv, pos, cos, sin, out, nq, nkv, hd, gin):
    n = out.shape[0]
    qkv_epilogue_bwd_kernel[(n,)](dq, dk, dv, pos, cos, sin, out, n,
                                  NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq),
                                  GIN=gin, GIN_P2=triton.next_power_of_2(gin),
                                  num_warps=ROPE_WARPS, enable_fp_fusion=False)


def warmup_specs(nq, nkv, hd, gin, grad_dtype):
    import torch
    f32, f16, i32 = torch.float32, torch.float16, torch.int32
    g = grad_dtype
    return [
        (qkv_epilogue_fwd_kernel, (f16, f16, f16, i32, f32, f32, 1, 2.0),
         dict(NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq), num_warps=ROPE_WARPS,
              enable_fp_fusion=False)),
        (qkv_epilogue_bwd_kernel, (g, g, g, i32, f32, f32, f16, 1),
         dict(NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq), GIN=gin,
              GIN_P2=triton.next_power_of_2(gin), num_warps=ROPE_WARPS, enable_fp_fusion=False)),
    ]
