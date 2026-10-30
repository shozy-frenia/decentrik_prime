"""Micro-benchmarks on a T4 for the design choices of the fast path (random data, real shapes).

    python dev/bench_kernels.py          # ~2-3 minutes, prints a table and a JSON line

A. cuBLAS fp16 GEMM throughput for every GEMM shape of a step (the floor of the step time)
B. our elementwise kernels per layer
C. attention per layer (forward + recompute forward + backward): padded SDPA vs the varlen
   memory-efficient kernel, and our Triton kernels (tl.dot runs on FMA units below sm_80)
D. fused LM head at several chunk sizes (time, logits buffer size)
E. NF4 dequant of one layer, alone and concurrently with the gate|up GEMM on a side stream
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEV = "cuda"
f16, f32 = torch.float16, torch.float32
RES = {}


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def gemms(n):
    D, I, V = 1536, 8960, 151936
    x = torch.randn(n, D, device=DEV, dtype=f16)
    out = {}
    shapes = {"qkv_fwd": (n, D, 2048), "o_fwd": (n, D, D), "gu_fwd": (n, D, 2 * I), "down_fwd": (n, I, D),
              "qkv_bwd": (n, 2048, D), "gu_bwd": (n, 2 * I, D), "down_bwd": (n, D, I), "head_fwd_256": (256, D, V),
              "head_bwd_256": (256, V, D), "head_fwd_512": (512, D, V)}
    for name, (m, k, nn) in shapes.items():
        a = torch.randn(m, k, device=DEV, dtype=f16)
        w = torch.randn(nn, k, device=DEV, dtype=f16) * 0.02
        ms = bench(lambda: torch.mm(a, w.t()), reps=10)
        out[name] = (round(ms, 3), round(2 * m * k * nn / ms / 1e9, 1))      # ms, TFLOP/s
    del x
    return out


def layer_kernels(n):
    from solution.k_mlp import gated_silu_backward, gated_silu_forward
    from solution.k_norm import rms_backward, rms_forward
    from solution.k_rope import qkv_epilogue_backward, qkv_epilogue_forward

    D, I = 1536, 8960
    d16, d32 = torch.zeros(16, dtype=f16, device=DEV), torch.zeros(16, device=DEV)
    x = torch.randn(n, D, device=DEV)
    w = torch.ones(D, device=DEV)
    y, xq, xv = (torch.empty(n, D, dtype=f16, device=DEV) for _ in range(3))
    xo = torch.empty(n, D, device=DEV)
    rstd = torch.empty(n, device=DEV)
    delta = torch.randn(n, D, device=DEV).half()
    out = {}
    out["rms_fwd_dropout"] = bench(lambda: rms_forward(x, w, y, rstd, 1e-6, delta=delta, xout=xo, xq=xq, xv=xv,
                                                       p_drop=0.05, keep_scale=1 / 0.95, seed_q=1, seed_v=2,
                                                       dummy16=d16, dummy32=d32))
    dh = torch.randn(n, D, device=DEV)
    dh16 = torch.empty(n, D, dtype=f16, device=DEV)
    one = torch.ones(1, device=DEV)
    out["rms_bwd_lora"] = bench(lambda: rms_backward(delta, one, x, w, rstd, dh, dres=dh, lq=delta, lv=delta,
                                                     p_drop=0.05, keep_scale=1 / 0.95, seed_q=1, seed_v=2,
                                                     out16=dh16, dummy16=d16, dummy32=d32))
    qkv = torch.randn(n, 2048, device=DEV).half()
    lq = torch.randn(n, 1536, device=DEV).half()
    lv = torch.randn(n, 256, device=DEV).half()
    pos = (torch.arange(n, device=DEV) % 512).int()
    cos = torch.randn(1024, 128, device=DEV)
    out["qkv_epilogue_fwd"] = bench(lambda: qkv_epilogue_forward(qkv, lq, lv, pos, cos, cos, 2.0, 12, 2, 128))
    dq, dk, dv = lq.clone(), lv.clone(), lv.clone()
    out["qkv_epilogue_bwd"] = bench(lambda: qkv_epilogue_backward(dq, dk, dv, pos, cos, cos, qkv, 12, 2, 128, 1))
    gu = torch.randn(n, 2 * I, device=DEV).half()
    act = torch.empty(n, I, dtype=f16, device=DEV)
    out["gated_silu_fwd"] = bench(lambda: gated_silu_forward(gu, act))
    out["gated_silu_bwd"] = bench(lambda: gated_silu_backward(gu, act))
    a16 = torch.randn(16, D, device=DEV).half()
    out["lora_tiny_gemm"] = bench(lambda: torch.mm(xq, a16.t()))
    return {k: round(v, 4) for k, v in out.items()}


def attention(lens, configs):
    from solution.k_attn import (AttnMeta, attention_backward, attention_forward, block_table,
                                 sdpa_attention, sdpa_attention_backward, varlen_attention,
                                 varlen_attention_backward)

    H, KV, HD = 12, 2, 128
    n, B, T = sum(lens), len(lens), max(lens)
    qkv = (torch.randn(n, (H + 2 * KV) * HD, device=DEV) * 0.5).half()
    row = torch.cat([torch.full((L,), b) for b, L in enumerate(lens)]).to(DEV)
    pos = torch.cat([torch.arange(L) for L in lens]).to(DEV)
    dout = torch.randn(n, H * HD, device=DEV).half()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    res = {"tokens": n, "padded_tokens": B * T}
    out_ref, _ = sdpa_attention(qkv, row, pos, B, T, H, KV, HD, grad=False)
    # one layer of a training step = forward + (recompute forward + backward)
    res["padded_sdpa_fwd"] = round(bench(lambda: sdpa_attention(qkv, row, pos, B, T, H, KV, HD, grad=False)), 3)

    def sdpa_fb():
        o, ctx = sdpa_attention(qkv, row, pos, B, T, H, KV, HD, grad=True)
        sdpa_attention_backward(ctx, dout, row, pos, B, T, H, KV, HD)

    res["padded_sdpa_layer_total"] = round(res["padded_sdpa_fwd"] + bench(sdpa_fb, reps=10), 3)
    try:
        out_v, _ = varlen_attention(qkv, cu, T, H, KV, HD)
        res["varlen_max_err_vs_padded"] = round(float((out_v.float() - out_ref.float()).abs().max()), 5)
        res["varlen_fwd"] = round(bench(lambda: varlen_attention(qkv, cu, T, H, KV, HD)), 3)

        def var_fb():
            o, ctx = varlen_attention(qkv, cu, T, H, KV, HD, need_lse=True)
            varlen_attention_backward(ctx, dout, cu, T, H, KV, HD)

        res["varlen_layer_total"] = round(res["varlen_fwd"] + bench(var_fb, reps=10), 3)
    except Exception as exc:  # noqa: BLE001
        res["varlen_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    for bm, bn, warps, bb, bwarps in configs:
        tag = f"triton_fwd{bm}x{bn}w{warps}_bwd{bb}w{bwarps}"
        try:
            tf = block_table(lens, bm)
            tb = block_table(lens, bb)
            meta = AttnMeta(cu, torch.tensor(tf, dtype=torch.int32, device=DEV),
                            torch.tensor(tb, dtype=torch.int32, device=DEV), len(tf) // 2, len(tb) // 2, n)
            t0 = time.perf_counter()
            out, lse = attention_forward(qkv, meta, H, KV, HD, bm, bn, warps)
            attention_backward(qkv, out, lse, dout, meta, H, KV, HD, bb, bwarps)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            err = float((out.float() - out_ref.float()).abs().max())
            fwd = bench(lambda: attention_forward(qkv, meta, H, KV, HD, bm, bn, warps), reps=3, warm=1)
            bwd = bench(lambda: attention_backward(qkv, out, lse, dout, meta, H, KV, HD, bb, bwarps),
                        reps=2, warm=1)
            res[tag] = {"fwd_ms": round(fwd, 3), "bwd_ms": round(bwd, 3), "layer_total": round(2 * fwd + bwd, 3),
                        "compile_s": round(compile_s, 1), "max_err_vs_sdpa": round(err, 4)}
        except Exception as exc:  # noqa: BLE001
            res[tag] = f"{type(exc).__name__}: {str(exc)[:200]}"
    print("attention", res, flush=True)
    return res


def triton_dot_path():
    """Does Triton use tensor cores for tl.dot on this GPU? (Triton 3.6: only from sm_80.)"""
    import triton
    import triton.language as tl

    @triton.jit
    def mm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))

    a = torch.randn(64, 64, device=DEV).half()
    b = torch.randn(64, 64, device=DEV).half()
    c = torch.empty(64, 64, device=DEV)
    k = mm_kernel[(1,)](a, b, c, 64, 64, 64)
    ptx = k.asm.get("ptx", "")
    return {"capability": list(torch.cuda.get_device_capability()),
            "ptx_has_mma": "mma.sync" in ptx, "ptx_fma_count": ptx.count("fma.rn.f32"),
            "max_err": round(float((c - a.float() @ b.float()).abs().max()), 4)}


def head(n, n_lab):
    from solution.k_head import fused_head

    D, V = 1536, 151936
    emb = (torch.randn(V, D, device=DEV) * 0.02).half()
    yf = torch.randn(n, D, device=DEV).half()
    tgt = torch.randint(0, V, (n,), device=DEV, dtype=torch.int32)
    tgt[n_lab:] = -1
    perm = torch.arange(n, device=DEV, dtype=torch.int32)
    res = {}
    for chunk in (128, 256, 512, 1024):
        ms = bench(lambda: fused_head(yf, emb, tgt, perm, perm, n_lab, chunk, 1024.0), reps=5, warm=2)
        res[f"chunk{chunk}"] = {"ms": round(ms, 2), "logits_buffer_mib": round(min(chunk, n) * V * 2 / 2**20)}
    return res


def dequant_overlap(n):
    from solution.k_nf4 import NF4_CODE, dequant_tensors

    nq = 46792704 // 64
    packed = torch.randint(0, 256, (nq * 32,), dtype=torch.uint8, device=DEV)
    qabs = torch.randint(0, 256, (nq,), dtype=torch.uint8, device=DEV)
    s2 = torch.rand(nq // 256, device=DEV) * 0.01
    o2 = torch.full((nq // 256,), 0.02, device=DEV)
    code2 = torch.linspace(-1, 1, 256, device=DEV)
    nf4 = torch.tensor(NF4_CODE, device=DEV)
    ws = torch.empty(nq * 64, dtype=f16, device=DEV)
    ws2 = torch.empty(nq * 64, dtype=f16, device=DEV)
    a = torch.randn(n, 1536, device=DEV).half()
    wgu = ws[:17920 * 1536].view(17920, 1536)
    side = torch.cuda.Stream()
    res = {}
    res["dequant_layer_ms"] = round(bench(lambda: dequant_tensors(packed, qabs, s2, o2, nq, code2, nf4, ws)), 3)
    res["gu_gemm_ms"] = round(bench(lambda: torch.mm(a, wgu.t())), 3)

    def both():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            dequant_tensors(packed, qabs, s2, o2, nq, code2, nf4, ws2)
        torch.mm(a, wgu.t())
        torch.cuda.current_stream().wait_stream(side)

    res["gu_gemm_with_concurrent_dequant_ms"] = round(bench(both), 3)
    return res


def main():
    from solution import precompile
    precompile.start()
    precompile.wait()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    for n in (830, 1500):
        RES[f"gemm_N{n}"] = gemms(n)
        print(f"gemm N={n}", RES[f"gemm_N{n}"], flush=True)
    RES["layer_kernels_N830"] = layer_kernels(830)
    print("kernels", RES["layer_kernels_N830"], flush=True)
    RES["dequant"] = dequant_overlap(830)
    print("dequant", RES["dequant"], flush=True)
    RES["head_N830"] = head(830, 500)
    print("head", RES["head_N830"], flush=True)
    try:
        RES["triton_dot"] = triton_dot_path()
    except Exception as exc:  # noqa: BLE001
        RES["triton_dot"] = f"{type(exc).__name__}: {exc}"
    print("triton dot", RES["triton_dot"], flush=True)
    RES["attn_838_of_2036"] = attention([509, 120, 100, 109], [(64, 32, 4, 32, 4)])
    RES["attn_830_even"] = attention([180, 250, 120, 280], [])
    RES["attn_1724"] = attention([512, 512, 400, 300], [])
    print("\nRESULT " + json.dumps(RES))


if __name__ == "__main__":
    main()
