"""Gated SiLU of the Qwen2 MLP on the fused gate|up GEMM output, forward and in-place backward.

The fused GEMM writes gu = [gate | up] as one [N, 2*8960] fp16 row per token. The forward reads
both halves of a row tile and writes act = fp16(silu(gate) * up) with fp32 math (bitsandbytes
returns gate/up as fp32 views of fp16 values, Qwen2MLP multiplies in fp32, the down projection
casts back to fp16). The backward overwrites gu in place with [d gate | d up], so the gradient of
the fused weight GEMM reads one contiguous buffer and no extra [N, 17920] tensor is allocated:
    d up   = dact * silu(gate)
    d gate = dact * up * sig * (1 + gate * (1 - sig))          (torch's silu_backward)
2-D grid: (token row, column tile), no integer division per element.
"""

import triton
import triton.language as tl

MLP_BLOCK = 1024
MLP_WARPS = 4


@triton.jit(do_not_specialize=["n_rows"])
def gated_silu_fwd_kernel(gu_ptr, act_ptr, n_rows, INTER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = col < INTER
    src = gu_ptr + row.to(tl.int64) * (2 * INTER)
    gate = tl.load(src + col, mask=cm, other=0.0).to(tl.float32)
    up = tl.load(src + INTER + col, mask=cm, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(act_ptr + row.to(tl.int64) * INTER + col, (silu * up).to(tl.float16), mask=cm)


@triton.jit(do_not_specialize=["n_rows"])
def gated_silu_bwd_kernel(gu_ptr, dact_ptr, n_rows, INTER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = col < INTER
    src = gu_ptr + row.to(tl.int64) * (2 * INTER)
    gate = tl.load(src + col, mask=cm, other=0.0).to(tl.float32)
    up = tl.load(src + INTER + col, mask=cm, other=0.0).to(tl.float32)
    da = tl.load(dact_ptr + row.to(tl.int64) * INTER + col, mask=cm, other=0.0).to(tl.float32)
    e = 1.0 + tl.exp(-gate)
    sig = 1.0 / e
    silu = gate / e
    d_up = da * silu
    d_gate = (da * up) * sig * (1.0 + gate * (1.0 - sig))
    tl.store(src + col, d_gate.to(tl.float16), mask=cm)
    tl.store(src + INTER + col, d_up.to(tl.float16), mask=cm)


def gated_silu_forward(gu, act):
    n, two_i = gu.shape
    inter = two_i // 2
    gated_silu_fwd_kernel[(n, triton.cdiv(inter, MLP_BLOCK))](gu, act, n, INTER=inter, BLOCK=MLP_BLOCK,
                                                             num_warps=MLP_WARPS)
    return act


def gated_silu_backward(gu, dact):
    n, two_i = gu.shape
    inter = two_i // 2
    gated_silu_bwd_kernel[(n, triton.cdiv(inter, MLP_BLOCK))](gu, dact, n, INTER=inter, BLOCK=MLP_BLOCK,
                                                             num_warps=MLP_WARPS)
    return gu


def warmup_specs(inter):
    import torch
    f16 = torch.float16
    return [
        (gated_silu_fwd_kernel, (f16, f16, 1), dict(INTER=inter, BLOCK=MLP_BLOCK, num_warps=MLP_WARPS)),
        (gated_silu_bwd_kernel, (f16, f16, 1), dict(INTER=inter, BLOCK=MLP_BLOCK, num_warps=MLP_WARPS)),
    ]
