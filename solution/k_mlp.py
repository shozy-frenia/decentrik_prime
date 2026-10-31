# solution/k_mlp.py
import triton
import triton.language as tl


@triton.jit
def gated_silu_fwd_kernel(
    gu_ptr,              # [T, 2 * I] fp16 where gate = gu[:, :I], up = gu[:, I:]
    h_ptr,               # [T, I] fp16 output
    stride_gu_t,
    stride_h_t,
    I: tl.constexpr,     # 8960
    BLOCK_C: tl.constexpr = 1024,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < I

    g_ptr = gu_ptr + t * stride_gu_t + cols
    u_ptr = g_ptr + I

    g = tl.load(g_ptr, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    u = tl.load(u_ptr, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    sig_g = tl.sigmoid(g)
    h = (g * sig_g) * u
    tl.store(h_ptr + t * stride_h_t + cols, h.to(tl.float16), mask=mask)


@triton.jit
def gated_silu_bwd_kernel(
    dh_ptr,              # [T, I] fp16 incoming grad
    gu_ptr,              # [T, 2 * I] fp16 saved [gate, up] from forward (overwritten in-place with [dg, du])
    stride_dh_t,
    stride_gu_t,
    I: tl.constexpr,     # 8960
    BLOCK_C: tl.constexpr = 1024,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < I

    g_ptr = gu_ptr + t * stride_gu_t + cols
    u_ptr = g_ptr + I

    dh = tl.load(dh_ptr + t * stride_dh_t + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    g = tl.load(g_ptr, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    u = tl.load(u_ptr, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    sig = tl.sigmoid(g)
    silu_g = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))

    dg = (dh * u) * dsilu
    du = dh * silu_g

    tl.store(g_ptr, dg.to(tl.float16), mask=mask, eviction_policy="evict_first")
    tl.store(u_ptr, du.to(tl.float16), mask=mask, eviction_policy="evict_first")
