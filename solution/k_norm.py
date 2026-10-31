"""Fused residual add + RMSNorm (+ the two LoRA dropout masks), forward and backward.

Forward, one program per token row (hidden 1536 held in registers):
    x      = x_prev (fp32) [+ delta (fp16 GEMM output)]      residual stream, kept in fp32 as in base
    rstd   = rsqrt(mean(x^2) + eps)
    y      = w * (x * rstd)                                   Qwen2RMSNorm in fp32 (weight fp32)
    y16    = fp16(y)                                          what bitsandbytes casts to for the GEMMs
    xq16   = fp16(dropout_q(y)), xv16 = fp16(dropout_v(y))    PEFT's lora_dropout on the fp32 input,
                                                              then autocast's fp16 cast for lora_A
The dropout masks come from Philox (tl.rand) keyed by (seed, token*1536+col), so the backward pass
regenerates exactly the same masks instead of storing them.

Backward, one program per row:
    dy   = fp16 grad of y16 (from the GEMM) * gscale
           + [mask_q] * keep_scale * lq16 + [mask_v] * keep_scale * lv16   (LoRA-A input grads)
    dx   = rstd * (w*dy - xhat * mean(w*dy*xhat)),  xhat = x * rstd
    out  = dres + dx                                                       (residual branch)
`out` may alias `dres` (in-place accumulation of the residual gradient).
Flags are runtime ints (one compiled variant per kernel).
"""

import triton
import triton.language as tl

HIDDEN_BLOCK = 2048     # next power of two above 1536
NORM_WARPS = 8


@triton.jit(do_not_specialize=["n_rows", "seed_q", "seed_v", "add_delta", "write_x", "dropout"])
def rms_fwd_kernel(x_ptr, delta_ptr, xout_ptr, w_ptr, y_ptr, rstd_ptr, xq_ptr, xv_ptr,
                   n_rows, eps, p_drop, keep_scale, seed_q, seed_v, add_delta, write_x, dropout,
                   D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cm = cols < D
    off = row * D + cols
    x = tl.load(x_ptr + off, mask=cm, other=0.0)
    if add_delta != 0:
        x = x + tl.load(delta_ptr + off, mask=cm, other=0.0).to(tl.float32)
    if write_x != 0:
        tl.store(xout_ptr + off, x, mask=cm)
    ms = tl.sum(x * x, axis=0) * (1.0 / D)
    r = tl.rsqrt(ms + eps)
    tl.store(rstd_ptr + row, r)
    w = tl.load(w_ptr + cols, mask=cm, other=0.0)
    y = w * (x * r)
    tl.store(y_ptr + off, y.to(tl.float16), mask=cm)
    if dropout != 0:
        keep_q = tl.rand(seed_q, off) >= p_drop
        tl.store(xq_ptr + off, tl.where(keep_q, y * keep_scale, 0.0).to(tl.float16), mask=cm)
        keep_v = tl.rand(seed_v, off) >= p_drop
        tl.store(xv_ptr + off, tl.where(keep_v, y * keep_scale, 0.0).to(tl.float16), mask=cm)


@triton.jit(do_not_specialize=["n_rows", "seed_q", "seed_v", "use_lora", "has_res", "write16"])
def rms_bwd_kernel(dy_ptr, gscale_ptr, x_ptr, w_ptr, rstd_ptr, dres_ptr, out_ptr, lq_ptr, lv_ptr,
                   out16_ptr, n_rows, p_drop, keep_scale, seed_q, seed_v, use_lora, has_res, write16,
                   D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cm = cols < D
    off = row * D + cols
    dy = tl.load(dy_ptr + off, mask=cm, other=0.0).to(tl.float32) * tl.load(gscale_ptr)
    if use_lora != 0:
        keep_q = tl.rand(seed_q, off) >= p_drop
        lq = tl.load(lq_ptr + off, mask=cm, other=0.0).to(tl.float32)
        dy += tl.where(keep_q, lq * keep_scale, 0.0)
        keep_v = tl.rand(seed_v, off) >= p_drop
        lv = tl.load(lv_ptr + off, mask=cm, other=0.0).to(tl.float32)
        dy += tl.where(keep_v, lv * keep_scale, 0.0)
    x = tl.load(x_ptr + off, mask=cm, other=0.0)
    r = tl.load(rstd_ptr + row)
    w = tl.load(w_ptr + cols, mask=cm, other=0.0)
    xh = x * r
    g = dy * w
    c = tl.sum(g * xh, axis=0) * (1.0 / D)
    dx = (g - xh * c) * r
    if has_res != 0:
        dx += tl.load(dres_ptr + off, mask=cm, other=0.0)
    tl.store(out_ptr + off, dx, mask=cm)
    if write16 != 0:
        # the fp16 copy the next GEMM consumes (autocast's cast of the residual grad)
        tl.store(out16_ptr + off, dx.to(tl.float16), mask=cm)


def rms_forward(x, w, y, rstd, eps, *, delta=None, xout=None, xq=None, xv=None,
                p_drop=0.0, keep_scale=1.0, seed_q=0, seed_v=0, dummy16=None, dummy32=None):
    n, d = x.shape
    dropout = xq is not None
    rms_fwd_kernel[(n,)](
        x, delta if delta is not None else dummy16, xout if xout is not None else dummy32, w, y, rstd,
        xq if dropout else dummy16, xv if dropout else dummy16,
        n, float(eps), float(p_drop), float(keep_scale), int(seed_q), int(seed_v),
        int(delta is not None), int(xout is not None), int(dropout),
        D=d, BLOCK=HIDDEN_BLOCK, num_warps=NORM_WARPS)


def rms_backward(dy16, gscale, x, w, rstd, out, *, dres=None, lq=None, lv=None, p_drop=0.0,
                 keep_scale=1.0, seed_q=0, seed_v=0, out16=None, dummy16=None, dummy32=None):
    n, d = x.shape
    use_lora = lq is not None
    rms_bwd_kernel[(n,)](
        dy16, gscale, x, w, rstd, dres if dres is not None else dummy32, out,
        lq if use_lora else dummy16, lv if use_lora else dummy16,
        out16 if out16 is not None else dummy16,
        n, float(p_drop), float(keep_scale), int(seed_q), int(seed_v), int(use_lora),
        int(dres is not None), int(out16 is not None), D=d, BLOCK=HIDDEN_BLOCK, num_warps=NORM_WARPS)


def warmup_specs(hidden):
    import torch
    f32, f16 = torch.float32, torch.float16
    return [
        (rms_fwd_kernel, (f32, f16, f32, f32, f16, f32, f16, f16, 1, 1e-6, 0.05, 1.05, 1, 1, 1, 1, 1),
         dict(D=hidden, BLOCK=HIDDEN_BLOCK, num_warps=NORM_WARPS)),
        (rms_bwd_kernel, (f16, f32, f32, f32, f32, f32, f32, f16, f16, f16, 1, 0.05, 1.05, 1, 1, 1, 1, 1),
         dict(D=hidden, BLOCK=HIDDEN_BLOCK, num_warps=NORM_WARPS)),
    ]
