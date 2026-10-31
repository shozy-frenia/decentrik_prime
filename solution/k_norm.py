# solution/k_norm.py
import torch
import triton
import triton.language as tl


@triton.jit
def rms_fwd_kernel(
    x_ptr,               # [M, N] fp16 input (or residual input if res_ptr is provided)
    w_ptr,               # [N] fp16 weight
    y_ptr,               # [M, N] fp16 normalized output
    rstd_ptr,            # [M] fp32 optional output rstd (0 / None if not saved)
    stride_x,
    stride_y,
    N: tl.constexpr,     # 1536
    eps: tl.constexpr,   # 1e-6
    SAVE_RSTD: tl.constexpr = True,
    BLOCK_N: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Single load of x and w into registers (no second global read pass)
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) * (1.0 / N)
    rstd = tl.rsqrt(var + eps)
    if SAVE_RSTD:
        tl.store(rstd_ptr + row, rstd)

    y = (x * rstd) * w
    tl.store(y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)


@triton.jit
def rms_bwd_kernel(
    dy_ptr,              # [M, N] fp16 incoming gradient
    x_ptr,               # [M, N] fp16 original unnormalized input x
    w_ptr,               # [N] fp16 RMSNorm weight (frozen in QLoRA)
    rstd_ptr,            # [M] fp32 saved rstd from forward (if HAS_RSTD)
    dres_ptr,            # [M, N] fp16 optional residual gradient to accumulate (+= dx)
    dx_ptr,              # [M, N] fp16 output gradient (can alias dres_ptr or dy_ptr)
    stride_dy,
    stride_x,
    stride_dres,
    stride_dx,
    N: tl.constexpr,     # 1536
    eps: tl.constexpr,
    HAS_RSTD: tl.constexpr = True,
    HAS_DRES: tl.constexpr = False,
    BLOCK_N: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if HAS_RSTD:
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
    else:
        var = tl.sum(x * x, axis=0) * (1.0 / N)
        rstd = tl.rsqrt(var + eps)

    x_hat = x * rstd
    dy_w = dy * w
    mean_dyw_xhat = tl.sum(dy_w * x_hat, axis=0) * (1.0 / N)
    dx = rstd * (dy_w - x_hat * mean_dyw_xhat)

    if HAS_DRES:
        dres = tl.load(dres_ptr + row * stride_dres + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dx + dres

    tl.store(dx_ptr + row * stride_dx + cols, dx.to(tl.float16), mask=mask)
