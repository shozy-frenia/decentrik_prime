"""QKV epilogue: LoRA add + rotary embedding, in place on the fused [N, (12+2+2)*128] QKV buffer.

Forward, one program per token (all 16 heads of the row):
    q32 = f32(q16) + scale * f32(lq16)            PEFT: base output (fp32 view of the fp16 GEMM) + lora
    k32 = f32(k16)
    v32 = f32(v16) + scale * f32(lv16)
    q/k <- rope(q32/k32) in fp32: [x1*c - x2*s, x2*c + x1*s] (= x*cos + rotate_half(x)*sin)
    everything is stored back as fp16 (what SDPA's autocast cast would hand to attention).
Backward reads the fp32 attention grads (dq, and the per-q-head partials of dk/dv that it also
sums over each GQA group), applies the transposed rotation and writes fp16 grads of the
pre-rotation q/k/v into the same buffer layout, ready for the fused QKV weight-grad GEMM:
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
def qkv_epilogue_bwd_kernel(dq_ptr, dkp_ptr, dvp_ptr, pos_ptr, cos_ptr, sin_ptr, out_ptr, n_rows,
                            NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                            NQ_P2: tl.constexpr, GROUP_P2: tl.constexpr):
    row = tl.program_id(0)
    HALF: tl.constexpr = HD // 2
    ROW: tl.constexpr = (NQ + 2 * NKV) * HD
    GROUP: tl.constexpr = NQ // NKV
    p = tl.load(pos_ptr + row)
    j = tl.arange(0, HALF)
    c = tl.load(cos_ptr + p * HD + j)
    s = tl.load(sin_ptr + p * HD + j)
    dst = out_ptr + row * ROW
    src = row * (NQ * HD)
    # query heads
    hq = tl.arange(0, NQ_P2)
    qm = (hq < NQ)[:, None]
    o1 = hq[:, None] * HD + j[None, :]
    d1 = tl.load(dq_ptr + src + o1, mask=qm, other=0.0)
    d2 = tl.load(dq_ptr + src + o1 + HALF, mask=qm, other=0.0)
    tl.store(dst + o1, (d1 * c[None, :] + d2 * s[None, :]).to(tl.float16), mask=qm)
    tl.store(dst + o1 + HALF, (d2 * c[None, :] - d1 * s[None, :]).to(tl.float16), mask=qm)
    # key heads: sum the per-query-head partials of each group, then rotate back
    hk = tl.arange(0, NKV)
    g = tl.arange(0, GROUP_P2)
    gm = (g < GROUP)[None, :, None]
    part = (hk[:, None, None] * GROUP + g[None, :, None]) * HD + j[None, None, :]   # [NKV, G, HALF]
    k1 = tl.sum(tl.load(dkp_ptr + src + part, mask=gm, other=0.0), axis=1)          # [NKV, HALF]
    k2 = tl.sum(tl.load(dkp_ptr + src + part + HALF, mask=gm, other=0.0), axis=1)
    ok = NQ * HD + hk[:, None] * HD + j[None, :]
    tl.store(dst + ok, (k1 * c[None, :] + k2 * s[None, :]).to(tl.float16))
    tl.store(dst + ok + HALF, (k2 * c[None, :] - k1 * s[None, :]).to(tl.float16))
    # value heads: sum the partials
    dvo = tl.arange(0, HD)
    vpart = (hk[:, None, None] * GROUP + g[None, :, None]) * HD + dvo[None, None, :]  # [NKV, G, HD]
    vsum = tl.sum(tl.load(dvp_ptr + src + vpart, mask=gm, other=0.0), axis=1)        # [NKV, HD]
    ov = (NQ + NKV) * HD + hk[:, None] * HD + dvo[None, :]
    tl.store(dst + ov, vsum.to(tl.float16))


def qkv_epilogue_forward(qkv, lq, lv, pos, cos, sin, lora_scale, nq, nkv, hd):
    n = qkv.shape[0]
    qkv_epilogue_fwd_kernel[(n,)](qkv, lq, lv, pos, cos, sin, n, float(lora_scale),
                                  NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq),
                                  num_warps=ROPE_WARPS, enable_fp_fusion=False)


def qkv_epilogue_backward(dq, dkp, dvp, pos, cos, sin, out, nq, nkv, hd):
    n = out.shape[0]
    qkv_epilogue_bwd_kernel[(n,)](dq, dkp, dvp, pos, cos, sin, out, n,
                                  NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq),
                                  GROUP_P2=triton.next_power_of_2(nq // nkv),
                                  num_warps=ROPE_WARPS, enable_fp_fusion=False)


def warmup_specs(nq, nkv, hd):
    import torch
    f32, f16, i32 = torch.float32, torch.float16, torch.int32
    return [
        (qkv_epilogue_fwd_kernel, (f16, f16, f16, i32, f32, f32, 1, 2.0),
         dict(NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq), num_warps=ROPE_WARPS,
              enable_fp_fusion=False)),
        (qkv_epilogue_bwd_kernel, (f32, f32, f32, i32, f32, f32, f16, 1),
         dict(NQ=nq, NKV=nkv, HD=hd, NQ_P2=triton.next_power_of_2(nq),
              GROUP_P2=triton.next_power_of_2(nq // nkv), num_warps=ROPE_WARPS,
              enable_fp_fusion=False)),
    ]
