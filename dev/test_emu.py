"""Run every Triton kernel of solution/ in the NumPy emulator against NumPy references."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tl_emu  # noqa: E402

tri, tl = tl_emu.install()
sys.path.insert(0, os.path.join(HERE, ".."))
from solution import k_attn, k_head, k_mlp, k_nf4, k_norm, k_rope  # noqa: E402

rng = np.random.default_rng(0)
f16, f32 = np.float16, np.float32
FAIL = []


def check(name, got, ref, atol, rtol=0.0):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    err = np.abs(got - ref)
    lim = atol + rtol * np.abs(ref)
    bad = ~(err <= lim)
    status = "ok " if not bad.any() else "BAD"
    print(f"  {status} {name:40s} max|err|={err.max() if err.size else 0:.3e}  (atol {atol}, rtol {rtol})")
    if bad.any():
        FAIL.append(name)


# ------------------------------------------------------------------ NF4 dequant
def test_nf4():
    print("nf4 dequant")
    n = 64 * 256 * 3                         # three second-level blocks
    nq = n // 64
    packed = rng.integers(0, 256, n // 2, dtype=np.uint8)
    qabs = rng.integers(0, 256, nq, dtype=np.uint8)
    code2 = np.sort(rng.uniform(-1, 1, 256)).astype(f32)
    scale2 = rng.uniform(0.001, 0.02, nq // 256).astype(f32)
    offset2 = np.repeat(rng.uniform(0.01, 0.03, 1).astype(f32), nq // 256)
    nf4 = np.array(k_nf4.NF4_CODE, dtype=f32)
    out = np.zeros(n, dtype=f16)
    grid = (tri.cdiv(nq, k_nf4.QBLOCKS_PER_PROGRAM),)
    k_nf4.nf4_layer_dequant_kernel[grid](packed, qabs, scale2, offset2, code2, nf4, out, nq,
                                         QB=k_nf4.QBLOCKS_PER_PROGRAM)
    # bitsandbytes semantics
    absmax = (code2[qabs] * scale2[np.arange(nq) // 256]).astype(f32) + offset2[np.arange(nq) // 256]
    hi = nf4[packed >> 4]
    lo = nf4[packed & 15]
    ref = np.empty(n, dtype=f16)
    blk = np.arange(n // 2) // 32
    ref[0::2] = (hi * absmax[blk]).astype(f16)
    ref[1::2] = (lo * absmax[blk]).astype(f16)
    check("dequant bit-exact", out.astype(np.float64), ref.astype(np.float64), 0.0)
    # partial last program
    nq2 = 256 * 2 + 0
    out2 = np.zeros(nq2 * 64, dtype=f16)
    k_nf4.nf4_layer_dequant_kernel[(tri.cdiv(nq2, 128),)](packed[:nq2 * 32], qabs[:nq2], scale2[:2],
                                                         offset2[:2], code2, nf4, out2, nq2, QB=128)
    check("dequant prefix", out2, ref[:nq2 * 64], 0.0)
    # a range starting inside the layer (what the split attention / MLP dequant launches do)
    q0, q1 = 256, 3 * 256
    out3 = np.zeros(n, dtype=f16)
    k_nf4.nf4_layer_dequant_kernel[(tri.cdiv(q1 - q0, 128),)](
        packed[q0 * 32:q1 * 32], qabs[q0:q1], scale2[q0 // 256:q1 // 256], offset2[q0 // 256:q1 // 256],
        code2, nf4, out3[q0 * 64:q1 * 64], q1 - q0, QB=128)
    check("dequant range", out3[q0 * 64:q1 * 64], ref[q0 * 64:q1 * 64], 0.0)
    check("dequant range leaves the rest", out3[:q0 * 64], np.zeros(q0 * 64), 0.0)


# ------------------------------------------------------------------ RMSNorm
def rand_mask(seed, off, p):
    return tl_emu.rand(seed, off).a >= p


def test_norm():
    print("rmsnorm fwd/bwd")
    n, d = 5, 1536
    eps, p, ks = 1e-6, 0.05, 1.0 / 0.95
    x = rng.standard_normal((n, d)).astype(f32)
    delta = (rng.standard_normal((n, d)) * 0.5).astype(f16)
    w = rng.uniform(0.5, 1.5, d).astype(f32)
    y = np.zeros((n, d), f16)
    xout = np.zeros((n, d), f32)
    rstd = np.zeros(n, f32)
    xq = np.zeros((n, d), f16)
    xv = np.zeros((n, d), f16)
    k_norm.rms_fwd_kernel[(n,)](x, delta, xout, w, y, rstd, xq, xv, n, eps, p, ks, 11, 22, 1, 1, 1,
                                D=d, BLOCK=2048)
    h = x + delta.astype(f32)
    r = 1.0 / np.sqrt((h.astype(np.float64) ** 2).mean(1) + eps)
    yr = w * (h * r[:, None])
    check("x + delta", xout, h, 0.0)
    check("rstd", rstd, r, 0, 1e-6)
    check("y fp16", y, yr, 1e-3, 2e-3)
    off = (np.arange(n)[:, None] * d + np.arange(d)[None, :]).astype(np.int32)
    mq, mv = rand_mask(11, off, p), rand_mask(22, off, p)
    check("dropout q", xq, np.where(mq, yr * ks, 0), 1e-3, 2e-3)
    check("dropout v", xv, np.where(mv, yr * ks, 0), 1e-3, 2e-3)
    print(f"     keep rate q {mq.mean():.4f} v {mv.mean():.4f} (expected 0.95), q/v masks differ "
          f"{(mq != mv).mean():.3f}")
    # without delta / write / dropout
    y2 = np.zeros((n, d), f16)
    k_norm.rms_fwd_kernel[(n,)](x, delta, xout, w, y2, rstd, xq, xv, n, eps, p, ks, 11, 22, 0, 0, 0,
                                D=d, BLOCK=2048)
    r2 = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(1) + eps)
    check("y (no delta)", y2, w * (x * r2[:, None]), 1e-3, 2e-3)

    # backward: dL/dh for L = sum(dy_total * y) with y = w * h * rstd(h)
    dy16 = (rng.standard_normal((n, d))).astype(f16)
    lq = (rng.standard_normal((n, d)) * 0.3).astype(f16)
    lv = (rng.standard_normal((n, d)) * 0.3).astype(f16)
    dres = rng.standard_normal((n, d)).astype(f32)
    gscale = np.array([0.75], f32)
    out = dres.copy()
    out16 = np.zeros((n, d), f16)
    k_norm.rms_bwd_kernel[(n,)](dy16, gscale, h, w, r.astype(f32), out, out, lq, lv, out16, n, p, ks, 11, 22,
                                1, 1, 1, D=d, BLOCK=2048)
    dyt = dy16.astype(np.float64) * 0.75 + np.where(mq, lq.astype(np.float64) * ks, 0) + np.where(mv, lv.astype(np.float64) * ks, 0)
    hh = h.astype(np.float64)
    rr = 1.0 / np.sqrt((hh ** 2).mean(1) + eps)
    xh = hh * rr[:, None]
    g = dyt * w
    ref = rr[:, None] * (g - xh * (g * xh).mean(1, keepdims=True)) + dres
    check("rms bwd (+lora, +res, in place)", out, ref, 1e-4, 1e-4)
    check("rms bwd fp16 copy == fp16(out)", out16, out.astype(f16), 0)
    # finite-difference check of the analytic formula on one row
    i, j = 2, 77
    def f(hrow):
        rr_ = 1.0 / np.sqrt((hrow ** 2).mean() + eps)
        return np.sum(dyt[i] * w * hrow * rr_)
    e = np.zeros(d)
    e[j] = 1e-5
    fd = (f(hh[i] + e) - f(hh[i] - e)) / 2e-5
    check("rms bwd vs finite difference", ref[i, j] - dres[i, j], fd, 1e-6, 1e-5)


# ------------------------------------------------------------------ QKV epilogue
def rope_tables(P, hd):
    inv = 1.0 / (1e6 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    fr = np.arange(P, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], 1)
    return np.cos(emb).astype(f32), np.sin(emb).astype(f32)


def test_rope():
    print("qkv epilogue fwd/bwd")
    nq, nkv, hd = 12, 2, 128
    n = 7
    W = (nq + 2 * nkv) * hd
    cos, sin = rope_tables(64, hd)
    pos = np.array([0, 1, 2, 0, 5, 31, 63], np.int32)
    qkv = (rng.standard_normal((n, W))).astype(f16)
    lq = (rng.standard_normal((n, nq * hd)) * 0.1).astype(f16)
    lv = (rng.standard_normal((n, nkv * hd)) * 0.1).astype(f16)
    orig = qkv.astype(np.float64).copy()
    k_rope.qkv_epilogue_fwd_kernel[(n,)](qkv, lq, lv, pos, cos, sin, n, 2.0, NQ=nq, NKV=nkv, HD=hd, NQ_P2=16)
    c, s = cos[pos], sin[pos]

    def rope(x):  # x [n, H, hd]
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return np.concatenate([x1 * c[:, None, :hd // 2] - x2 * s[:, None, :hd // 2],
                               x2 * c[:, None, :hd // 2] + x1 * s[:, None, :hd // 2]], -1)

    q = orig[:, :nq * hd].reshape(n, nq, hd) + 2.0 * lq.reshape(n, nq, hd)
    k = orig[:, nq * hd:(nq + nkv) * hd].reshape(n, nkv, hd)
    v = orig[:, (nq + nkv) * hd:] + 2.0 * lv
    check("q rope+lora", qkv[:, :nq * hd], rope(q).reshape(n, -1), 2e-3, 2e-3)
    check("k rope", qkv[:, nq * hd:(nq + nkv) * hd], rope(k).reshape(n, -1), 2e-3, 2e-3)
    check("v lora", qkv[:, (nq + nkv) * hd:], v, 2e-3, 2e-3)
    # backward = transpose of the forward linear map (dot-product test) + GQA sum
    G = nq // nkv
    dq = rng.standard_normal((n, nq * hd)).astype(f32)
    dkp = rng.standard_normal((n, nq * hd)).astype(f32)
    dvp = rng.standard_normal((n, nq * hd)).astype(f32)
    out = np.zeros((n, W), f16)
    k_rope.qkv_epilogue_bwd_kernel[(n,)](dq, dkp, dvp, pos, cos, sin, out, n, NQ=nq, NKV=nkv, HD=hd,
                                         NQ_P2=16, GIN=G, GIN_P2=8)

    def rope_t(dd):  # adjoint
        d1, d2 = dd[..., :hd // 2], dd[..., hd // 2:]
        cc, ss = c[:, None, :hd // 2], s[:, None, :hd // 2]
        return np.concatenate([d1 * cc + d2 * ss, d2 * cc - d1 * ss], -1)

    ref_q = rope_t(dq.reshape(n, nq, hd).astype(np.float64)).reshape(n, -1)
    dk = dkp.reshape(n, nkv, G, hd).sum(2)
    ref_k = rope_t(dk.astype(np.float64)).reshape(n, -1)
    ref_v = dvp.reshape(n, nkv, G, hd).sum(2).reshape(n, -1)
    check("dq rope^T", out[:, :nq * hd], ref_q, 2e-3, 2e-3)
    check("dk group-sum + rope^T", out[:, nq * hd:(nq + nkv) * hd], ref_k, 1e-2, 2e-3)
    check("dv group-sum", out[:, (nq + nkv) * hd:], ref_v, 1e-2, 2e-3)
    out1 = np.zeros((n, W), f16)
    k_rope.qkv_epilogue_bwd_kernel[(n,)](dq.astype(f16), dk.astype(f16).reshape(n, -1),
                                         ref_v.astype(f16), pos, cos, sin, out1, n, NQ=nq, NKV=nkv,
                                         HD=hd, NQ_P2=16, GIN=1, GIN_P2=1)
    check("GIN=1 (SDPA layout) dq", out1[:, :nq * hd], ref_q, 1e-2, 3e-3)
    check("GIN=1 dk", out1[:, nq * hd:(nq + nkv) * hd], ref_k, 2e-2, 3e-3)
    check("GIN=1 dv", out1[:, (nq + nkv) * hd:], ref_v, 2e-2, 3e-3)
    # adjoint property <rope(x), d> == <x, rope_t(d)>
    x = rng.standard_normal((n, 1, hd))
    dd = rng.standard_normal((n, 1, hd))
    check("rope adjoint identity", np.sum(rope(x) * dd), np.sum(x * rope_t(dd)), 1e-9, 1e-9)


# ------------------------------------------------------------------ attention
def ref_attention(q, k, v, lens, G):
    """q [N, H, D], k/v [N, KV, D]; per-sequence causal softmax attention, float64."""
    N, H, D = q.shape
    o = np.zeros_like(q)
    lse = np.zeros((H, N))
    s0 = 0
    for L in lens:
        sl = slice(s0, s0 + L)
        for h in range(H):
            qq, kk, vv = q[sl, h], k[sl, h // G], v[sl, h // G]
            sc = qq @ kk.T / np.sqrt(D)
            sc = np.where(np.tril(np.ones((L, L), bool)), sc, -np.inf)
            m = sc.max(1, keepdims=True)
            p = np.exp(sc - m)
            l = p.sum(1, keepdims=True)
            o[sl, h] = (p / l) @ vv
            lse[h, sl] = (m + np.log(l))[:, 0]
        s0 += L
    return o, lse


def ref_attention_bwd(q, k, v, do, lens, G):
    N, H, D = q.shape
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    s0 = 0
    for L in lens:
        sl = slice(s0, s0 + L)
        for h in range(H):
            qq, kk, vv, dd = q[sl, h], k[sl, h // G], v[sl, h // G], do[sl, h]
            sc = qq @ kk.T / np.sqrt(D)
            sc = np.where(np.tril(np.ones((L, L), bool)), sc, -np.inf)
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            dv[sl, h // G] += p.T @ dd
            dp = dd @ vv.T
            ds = p * (dp - (dp * p).sum(1, keepdims=True))
            dq[sl, h] += ds @ kk / np.sqrt(D)
            dk[sl, h // G] += ds.T @ qq / np.sqrt(D)
        s0 += L
    return dq, dk, dv


def test_attention():
    print("flash attention varlen causal GQA")
    nq, nkv, hd = 4, 2, 128
    G = nq // nkv
    lens = [1, 37, 64, 70, 5]
    n = sum(lens)
    W = (nq + 2 * nkv) * hd
    qkv = (rng.standard_normal((n, W)) * 0.7).astype(f16)
    cu = np.array([0] + list(np.cumsum(lens)), np.int32)
    tab_f = np.array(k_attn.block_table(lens, k_attn.FWD_BLOCK_M), np.int32)
    tab_b = np.array(k_attn.block_table(lens, k_attn.BWD_BLOCK), np.int32)
    out = np.zeros((n, nq * hd), f16)
    lse = np.zeros((nq, n), f32)
    sm = 1.0 / np.sqrt(hd)
    k_attn.attn_fwd_kernel[(len(tab_f) // 2, nq)](qkv, out, lse, cu, tab_f, n, sm * k_attn.LOG2E,
                                                  NQ=nq, NKV=nkv, HD=hd, BLOCK_M=k_attn.FWD_BLOCK_M,
                                                  BLOCK_N=k_attn.FWD_BLOCK_N)
    q = qkv[:, :nq * hd].reshape(n, nq, hd).astype(np.float64)
    k = qkv[:, nq * hd:(nq + nkv) * hd].reshape(n, nkv, hd).astype(np.float64)
    v = qkv[:, (nq + nkv) * hd:].reshape(n, nkv, hd).astype(np.float64)
    o_ref, lse_ref = ref_attention(q, k, v, lens, G)
    check("attn out", out.reshape(n, nq, hd), o_ref, 2e-3, 2e-3)
    check("attn lse (log2)", lse, lse_ref / np.log(2), 1e-4, 1e-5)
    do = (rng.standard_normal((n, nq * hd))).astype(f16)
    delta = np.zeros((nq, n), f32)
    k_attn.attn_bwd_delta_kernel[(n,)](out, do, delta, n, NQ=nq, HD=hd, NQ_P2=4)
    check("delta", delta, (out.astype(np.float64) * do).reshape(n, nq, hd).sum(-1).T, 1e-3, 1e-4)
    dq = np.zeros((n, nq * hd), f32)
    dkp = np.zeros((n, nq * hd), f32)
    dvp = np.zeros((n, nq * hd), f32)
    nb = len(tab_b) // 2
    k_attn.attn_bwd_dkdv_kernel[(nb, nq)](qkv, do, lse, delta, dkp, dvp, cu, tab_b, n, sm * k_attn.LOG2E, sm,
                                          NQ=nq, NKV=nkv, HD=hd, BLOCK=k_attn.BWD_BLOCK)
    k_attn.attn_bwd_dq_kernel[(nb, nq)](qkv, do, lse, delta, dq, cu, tab_b, n, sm * k_attn.LOG2E, sm,
                                        NQ=nq, NKV=nkv, HD=hd, BLOCK=k_attn.BWD_BLOCK)
    dq_r, dk_r, dv_r = ref_attention_bwd(q, k, v, do.reshape(n, nq, hd).astype(np.float64), lens, G)
    check("dq", dq.reshape(n, nq, hd), dq_r, 2e-2, 2e-2)
    check("dk (group-summed partials)", dkp.reshape(n, nkv, G, hd).sum(2), dk_r, 2e-2, 2e-2)
    check("dv (group-summed partials)", dvp.reshape(n, nkv, G, hd).sum(2), dv_r, 2e-2, 2e-2)


# ------------------------------------------------------------------ gated SiLU
def test_mlp():
    print("gated silu fwd/bwd")
    n, inter = 3, 1100                     # not a multiple of the block
    gu = (rng.standard_normal((n, 2 * inter)) * 2).astype(f16)
    act = np.zeros((n, inter), f16)
    k_mlp.gated_silu_fwd_kernel[(n, tri.cdiv(inter, 1024))](gu, act, n, INTER=inter, BLOCK=1024)
    g = gu[:, :inter].astype(np.float64)
    u = gu[:, inter:].astype(np.float64)
    sg = 1 / (1 + np.exp(-g))
    check("act", act, g * sg * u, 2e-3, 2e-3)
    da = rng.standard_normal((n, inter)).astype(f16)
    buf = gu.copy()
    k_mlp.gated_silu_bwd_kernel[(n, tri.cdiv(inter, 1024))](buf, da, n, INTER=inter, BLOCK=1024)
    dd = da.astype(np.float64)
    check("d gate", buf[:, :inter], dd * u * sg * (1 + g * (1 - sg)), 3e-3, 3e-3)
    check("d up", buf[:, inter:], dd * g * sg, 3e-3, 3e-3)


# ------------------------------------------------------------------ head
def test_head():
    print("lm head rows (CE / entropy / argmax / grad)")
    V, C = 3000, 6
    logits = (rng.standard_normal((C, V)) * 3).astype(f16)
    logits[1, 5] = logits[1, 2900] = 30.0           # tie, first index wins
    logits[4, :] = 0.0                              # all-equal row -> argmax 0
    tgt = np.array([7, 5, 2999, -1, 0, 12, -1, -1], np.int32)   # global sorted targets, row0 = 0
    stats = np.zeros((8, 4), f32)
    lg = logits.copy()
    k_head.lm_head_rows_kernel[(C,)](lg, tgt, stats, 0, C, V, 1, 1024.0, BLOCK_V=2048)
    x = logits.astype(np.float64)
    m = x.max(1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]
    p = np.exp(x - lse[:, None])
    ent = -(p * np.log(p)).sum(1)
    am = x.argmax(1)
    t = tgt[:C]
    loss = np.where(t >= 0, lse - x[np.arange(C), np.maximum(t, 0)], 0)
    hit = np.where((t >= 0) & (am == t), 1.0, 0.0)
    check("loss", stats[:C, 0], loss, 1e-4, 1e-5)
    check("entropy", stats[:C, 1], ent, 1e-4, 1e-4)
    check("hit (argmax, first index on ties)", stats[:C, 2], hit, 0)
    check("lse", stats[:C, 3], lse, 1e-4, 1e-6)
    grad = (p - np.eye(V)[np.maximum(t, 0)]) * 1024.0
    rows = t >= 0
    check("grad rows (in place, x1024)", lg[rows], grad[rows], 0.05, 2e-3)
    check("rows without target untouched", lg[~rows], logits[~rows], 0)
    # stats-only chunk with row offset
    stats2 = np.zeros((8, 4), f32)
    lg2 = logits[:2].copy()
    k_head.lm_head_rows_kernel[(2,)](lg2, tgt, stats2, 6, 2, V, 0, 1024.0, BLOCK_V=2048)
    check("offset rows: loss 0 for no target", stats2[6:8, 0], [0, 0], 0)
    check("offset rows: entropy", stats2[6:8, 1], ent[:2], 1e-4, 1e-4)
    check("stats-only leaves logits", lg2, logits[:2], 0)


def _head_chunks_case(V, N, D, CH, n_valid_rows, seed, plant_tie=True, label=""):
    """Drive head_chunk_kernel / head_finalize_kernel exactly like k_head.streamed_head does (the
    two GEMMs per chunk done here in fp32 with an fp16 logits output, like cuBLAS), against a
    float64 reference on the full fp16 logits."""
    r = np.random.default_rng(seed)
    E = (r.standard_normal((V, D)) * 0.5).astype(f16)
    Y = (r.standard_normal((N, D)) * 0.8).astype(f16)
    tgt = r.integers(0, V, N).astype(np.int64)
    invalid = r.permutation(N)[: N - n_valid_rows]
    tgt[invalid] = -1
    if plant_tie and N > 1 and V > 2000:
        # rows 5 and 2000 of E identical, token 1 aligned with them: its max logit ties across two
        # different vocabulary chunks -> the first index must win
        E[2000] = E[5]
        Y[1] = (E[5].astype(f32) * 6.0).astype(f16)
        tgt[1] = 2000
    valid = tgt >= 0
    perm = np.concatenate([np.nonzero(valid)[0], np.nonzero(~valid)[0]]).astype(np.int32)
    n_lab = int(valid.sum())
    ys = Y[perm]
    tgt_sorted = tgt[perm].astype(np.int32)
    state = np.zeros((N, 8), f32)
    state[:, 0] = -np.inf
    acc = np.zeros((max(n_lab, 1), D), f32)
    n_chunks = -(-V // CH)
    for c in range(n_chunks):
        v0 = c * CH
        vc = min(CH, V - v0)
        ec = E[v0:v0 + vc]
        lg = (ys.astype(f32) @ ec.astype(f32).T).astype(f16)
        k_head.head_chunk_kernel[(N,)](lg, tgt_sorted, state, acc, N, n_lab, vc, v0, 1024.0,
                                       D=D, BLOCK_V=256, BLOCK_D=tri.next_power_of_2(D))
        if n_lab:
            acc[:n_lab] += lg[:n_lab].astype(f32) @ ec.astype(f32)
    stats = np.zeros((N, 4), f32)
    dy = np.full((N, D), np.nan, f16)
    et = E[np.maximum(tgt_sorted[:max(n_lab, 1)], 0)]
    k_head.head_finalize_kernel[(N,)](state, tgt_sorted, perm, acc, et, stats, dy, N, n_lab,
                                      1024.0, 1.0 / 1024.0, D=D, BLOCK_D=tri.next_power_of_2(D))
    full = (ys.astype(f32) @ E.astype(f32).T).astype(f16).astype(np.float64)
    m = full.max(1, keepdims=True)
    lse = (m + np.log(np.exp(full - m).sum(1, keepdims=True)))[:, 0]
    p = np.exp(full - lse[:, None])
    ent = -(p * np.log(p)).sum(1)
    am = full.argmax(1)
    t = tgt_sorted.astype(np.int64)
    ok = t >= 0
    loss = np.where(ok, lse - full[np.arange(N), np.maximum(t, 0)], 0.0)
    hit = np.where(ok & (am == t), 1.0, 0.0)
    check(f"{label}loss", stats[:, 0], loss, 2e-4, 1e-5)
    check(f"{label}entropy", stats[:, 1], ent, 2e-4, 1e-4)
    check(f"{label}hit (first index across chunks)", stats[:, 2], hit, 0)
    check(f"{label}lse", stats[:, 3], lse, 2e-4, 1e-6)
    onehot = np.zeros_like(p)
    onehot[np.arange(N)[ok], t[ok]] = 1.0
    g_sorted = (p - onehot) @ E.astype(np.float64) * 1024.0
    g_sorted[~ok] = 0.0
    ref = np.zeros((N, D))
    ref[perm] = g_sorted
    check(f"{label}d(hidden) x1024, token order", dy.astype(np.float64), ref, 2e-2, 3e-3)
    if plant_tie and N > 1 and V > 2000:
        row1 = int(np.nonzero(perm == 1)[0][0])
        check(f"{label}planted tie -> index 5", [stats[row1, 2]], [0.0], 0)


def test_head_chunks():
    print("lm head over vocabulary chunks (online softmax, gradient accumulated in the pass)")
    _head_chunks_case(2600, 9, 48, 512, 6, 11, label="")
    _head_chunks_case(2600, 5, 48, 512, 0, 12, plant_tie=False, label="no targets: ")
    _head_chunks_case(1000, 4, 40, 1000, 4, 13, plant_tie=False, label="one chunk: ")
    # confident predictions: p_t -> 1, the gradient is a small difference (target term in fp32)
    r = np.random.default_rng(14)
    V, N, D, CH = 1500, 6, 32, 256
    E = (r.standard_normal((V, D)) * 0.3).astype(f16)
    tgt = r.integers(0, V, N).astype(np.int32)
    Y = (E[tgt].astype(f32) * 6.0).astype(f16)       # p_t between 0.87 and 0.99998
    perm = np.arange(N, dtype=np.int32)
    state = np.zeros((N, 8), f32)
    state[:, 0] = -np.inf
    acc = np.zeros((N, D), f32)
    for c in range(-(-V // CH)):
        v0, vc = c * CH, min(CH, V - c * CH)
        ec = E[v0:v0 + vc]
        lg = (Y.astype(f32) @ ec.astype(f32).T).astype(f16)
        k_head.head_chunk_kernel[(N,)](lg, tgt, state, acc, N, N, vc, v0, 1024.0, D=D, BLOCK_V=128,
                                       BLOCK_D=32)
        acc += lg.astype(f32) @ ec.astype(f32)
    stats = np.zeros((N, 4), f32)
    dy = np.zeros((N, D), f16)
    k_head.head_finalize_kernel[(N,)](state, tgt, perm, acc, E[tgt], stats, dy, N, N, 1024.0,
                                      1.0 / 1024.0, D=D, BLOCK_D=32)
    full16 = (Y.astype(f32) @ E.astype(f32).T).astype(f16)
    full = full16.astype(np.float64)
    m = full.max(1, keepdims=True)
    p = np.exp(full - m)
    p /= p.sum(1, keepdims=True)
    oh = np.zeros_like(p)
    oh[np.arange(N), tgt] = 1.0
    g = (p - oh) @ E.astype(np.float64) * 1024.0
    # What the base computes for the same rows: fp32 log-softmax gradient, cast to fp16 under the
    # GradScaler's scale (65536 / ~500 supervised tokens) for the fp16 lm_head GEMM backward. For a
    # very confident row (p_t = 0.99998) the fp32 p_t - 1 alone is off by a few percent, in the base
    # as here; the streamed head must be no less precise than that, row by row.
    x32 = full16.astype(f32)
    mx = x32.max(1, keepdims=True)
    lse32 = mx + np.log(np.exp(x32 - mx).sum(1, keepdims=True, dtype=f32))
    gl32 = (np.exp(x32 - lse32) - oh.astype(f32)).astype(f32)
    sc = f32(65536 / 500.0)
    base = ((gl32 * sc).astype(f16).astype(f32) @ E.astype(f32)).astype(np.float64) / float(sc) * 1024.0
    rel = np.abs(dy.astype(np.float64) - g).max(1) / np.abs(g).max(1)
    rel_base = np.abs(base - g).max(1) / np.abs(g).max(1)
    print(f"    (p_t {np.array2string(p[np.arange(N), tgt], precision=5)}; relative error ours "
          f"{np.array2string(rel, precision=4)}, base {np.array2string(rel_base, precision=4)})")
    check("confident rows: d(hidden) no less precise than the base", rel,
          np.zeros(N), 1e-3 + 1.5 * rel_base)


if __name__ == "__main__":
    for fn in (test_nf4, test_norm, test_rope, test_attention, test_mlp, test_head, test_head_chunks):
        fn()
    print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nALL EMULATED KERNEL CHECKS PASSED")
    sys.exit(1 if FAIL else 0)
