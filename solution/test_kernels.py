"""Correctness of our kernels against plain PyTorch / bitsandbytes, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Every Triton kernel of the fast path is checked
against an independent PyTorch implementation of the same math (fp32/fp64 references).
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def _close(got, ref, atol, rtol, what):
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol, msg=lambda m: f"{what}: {m}")


# ------------------------------------------------------------------ NF4 dequantisation
class _FakeLinear:
    def __init__(self, weight):
        self.weight = weight


def _qs_to(state, device):
    """A copy of a bitsandbytes QuantState on `device` (QuantState.to works in place)."""
    import copy

    qs = copy.copy(state)
    qs.code = state.code.to(device)
    qs.absmax = state.absmax.to(device)
    if getattr(state, "nested", False):
        qs.offset = state.offset.to(device)
        qs.state2 = copy.copy(state.state2)
        qs.state2.absmax = state.state2.absmax.to(device)
        qs.state2.code = state.state2.code.to(device)
    return qs


def test_nf4_layer_dequant_is_bit_exact():
    bnb = pytest.importorskip("bitsandbytes")
    from bitsandbytes.nn import Params4bit

    from solution.k_nf4 import LAYER_WEIGHTS, NF4_CODE, dequant_layer, repack_layer

    torch.manual_seed(0)
    shapes = {"q_proj": (256, 128), "k_proj": (128, 128), "v_proj": (128, 128), "o_proj": (256, 128),
              "gate_proj": (384, 128), "up_proj": (384, 128), "down_proj": (128, 384)}
    linears, refs = {}, {}
    for name in LAYER_WEIGHTS:
        w = (torch.randn(shapes[name]) * 0.05).half()
        p = Params4bit(w, requires_grad=False, quant_type="nf4", compress_statistics=True,
                       blocksize=64).to(DEV)
        linears[name] = _FakeLinear(p)
        refs[name] = bnb.functional.dequantize_4bit(p.data, p.quant_state).half()
    packed, code2 = repack_layer(linears, torch.device(DEV))
    out = torch.empty(packed.numel, dtype=torch.float16, device=DEV)
    nf4 = torch.tensor(NF4_CODE, dtype=torch.float32, device=DEV)
    dequant_layer(packed, code2, nf4, out)
    for name in LAYER_WEIGHTS:
        start, rows, cols = packed.segments[name]
        got = out[start:start + rows * cols].view(rows, cols)
        assert torch.equal(got, refs[name]), f"{name} differs from bitsandbytes"
        # bitsandbytes still reads the (re-pointed) packed weight correctly
        p = linears[name].weight
        again = bnb.functional.dequantize_4bit(p.data, p.quant_state).half()
        assert torch.equal(again, refs[name])
    # split dequantisation (attention part / MLP part), the second one on a side stream
    from solution.k_nf4 import RANGE_ALIGN, dequant_range

    split = RANGE_ALIGN
    assert packed.n_qblocks > split
    out3 = torch.zeros_like(out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        dequant_range(packed, split, packed.n_qblocks, code2, nf4, out3)
    torch.cuda.current_stream().wait_stream(side)
    assert not out3[:split * 64].any(), "range dequant wrote outside its range"
    dequant_range(packed, 0, split, code2, nf4, out3)
    assert torch.equal(out3, out), "split dequant differs from the whole-layer dequant"
    with pytest.raises(ValueError):
        dequant_range(packed, 256, packed.n_qblocks, code2, nf4, out3)
    # v6: layers kept in pinned host memory and streamed through the weight pipe's two GPU slots
    # (layer j -> slot j % 2), dequantised from the slot on a side stream; three different layers
    # walked forward and back like the training step does, with one layer resident on the GPU
    from solution.k_nf4 import WeightPipe

    host_layer, code2h = repack_layer(linears, torch.device(DEV), host=True)
    assert not host_layer.flat.is_cuda and host_layer.flat.is_pinned()
    assert torch.equal(code2h, code2)
    for name in LAYER_WEIGHTS:          # bitsandbytes' tensors now point into the pinned buffer
        p = linears[name].weight
        assert not p.data.is_cuda
        assert torch.equal(bnb.functional.dequantize_4bit(p.data.to(DEV), _qs_to(p.quant_state, DEV)).half(),
                           refs[name])
    others = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        lin = {}
        for name in LAYER_WEIGHTS:
            w = (torch.randn(shapes[name]) * 0.05).half()
            lin[name] = _FakeLinear(Params4bit(w, requires_grad=False, quant_type="nf4",
                                               compress_statistics=True, blocksize=64).to(DEV))
        others.append(lin)
    ref_b = torch.empty_like(out)
    pk_b, _ = repack_layer(others[0], torch.device(DEV))
    dequant_layer(pk_b, code2, nf4, ref_b)
    ref_c = torch.empty_like(out)
    pk_c, _ = repack_layer(others[1], torch.device(DEV))
    dequant_layer(pk_c, code2, nf4, ref_c)
    host_b, _ = repack_layer(others[0], torch.device(DEV), host=True)
    host_c, _ = repack_layer(others[1], torch.device(DEV), host=True)
    expect = [out, ref_b, ref_c]
    side = torch.cuda.Stream()
    for resident, layers in (((), [host_layer, host_b, host_c]), ((1,), [host_layer, pk_b, host_c])):
        pipe = WeightPipe(layers, torch.device(DEV), side=side, resident=resident)
        pipe.prefetch(0)
        pipe.prefetch(1)
        walk = [0, 1, 2, 2, 1, 0, 0, 1, 2]
        for step, j in enumerate(walk):
            got = torch.full_like(out, float("nan"))
            side.wait_stream(torch.cuda.current_stream())
            src = pipe.acquire(j, side)
            with torch.cuda.stream(side):
                dequant_range(src, 0, split, code2, nf4, got)
                dequant_range(src, split, src.n_qblocks, code2, nf4, got)
            torch.cuda.current_stream().wait_stream(side)
            ahead = walk[step + 2] if step + 2 < len(walk) else None
            if ahead is not None and ahead != j:
                pipe.prefetch(ahead)
            assert torch.equal(got, expect[j]), f"streamed layer {j} differs (resident={resident})"
        n_streamed = sum(1 for j in walk if j not in resident)
        assert pipe.copies <= n_streamed, (pipe.copies, n_streamed)


# ------------------------------------------------------------------ RMSNorm
def test_rmsnorm_forward_backward_and_dropout():
    from solution.k_norm import rms_backward, rms_forward

    torch.manual_seed(1)
    n, d, eps, p = 300, 1536, 1e-6, 0.05
    ks = 1.0 / float(torch.tensor(1.0 - p, dtype=torch.float32))
    x = torch.randn(n, d, device=DEV)
    delta = (torch.randn(n, d, device=DEV) * 0.5).half()
    w = torch.rand(d, device=DEV) + 0.5
    y = torch.empty(n, d, dtype=torch.float16, device=DEV)
    xq, xv = torch.empty_like(y), torch.empty_like(y)
    xout = torch.empty(n, d, device=DEV)
    rstd = torch.empty(n, device=DEV)
    d16, d32 = torch.zeros(16, dtype=torch.float16, device=DEV), torch.zeros(16, device=DEV)
    # v15: the keep-masks are PyTorch's own dropout masks, passed in as uint8
    _, keep_q = torch.native_dropout(torch.empty(n, d, device=DEV), p, True)
    _, keep_v = torch.native_dropout(torch.empty(n, d, device=DEV), p, True)
    mq, mv = keep_q.view(torch.uint8), keep_v.view(torch.uint8)
    rms_forward(x, w, y, rstd, eps, delta=delta, xout=xout, xq=xq, xv=xv, mq=mq, mv=mv, keep_scale=ks,
                dummy16=d16, dummy32=d32)
    h = x + delta.float()
    ref = w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))
    assert torch.equal(xout, h)
    _close(y, ref.half(), 2e-3, 2e-3, "y")
    # the LoRA inputs: PyTorch's dropout (this mask, scale 1/(1-p)) of the norm output, in fp16
    _close(xq.float(), (ref * keep_q * ks).half().float(), 2e-3, 2e-3, "dropout q")
    _close(xv.float(), (ref * keep_v * ks).half().float(), 2e-3, 2e-3, "dropout v")
    assert torch.equal(xq[~keep_q], torch.zeros_like(xq[~keep_q]))
    assert torch.equal(xv[~keep_v], torch.zeros_like(xv[~keep_v]))
    xq2, xv2 = torch.empty_like(y), torch.empty_like(y)
    # backward vs autograd, including the two LoRA input-grad branches and the residual
    dy16 = torch.randn(n, d, device=DEV).half()
    lq = (torch.randn(n, d, device=DEV) * 0.3).half()
    lv = (torch.randn(n, d, device=DEV) * 0.3).half()
    dres = torch.randn(n, d, device=DEV)
    gscale = torch.tensor([0.5], device=DEV)
    hh = h.double().requires_grad_(True)
    yy = w.double() * (hh * torch.rsqrt(hh.pow(2).mean(-1, keepdim=True) + eps))
    upstream = dy16.double() * 0.5 + keep_q.double() * lq.double() * ks + keep_v.double() * lv.double() * ks
    (yy * upstream).sum().backward()
    out = dres.clone()
    out16 = torch.empty(n, d, dtype=torch.float16, device=DEV)
    rms_backward(dy16, gscale, h, w, rstd, out, dres=out, lq=lq, lv=lv, mq=mq, mv=mv, keep_scale=ks,
                 out16=out16, dummy16=d16, dummy32=d32)
    _close(out, hh.grad + dres.double(), 1e-4, 1e-4, "rms backward")
    assert torch.equal(out16, out.half())
    # v7: the backward given the norm input as x + delta (no recomputed h) is bit-identical
    out_d = dres.clone()
    out16_d = torch.empty_like(out16)
    rms_backward(dy16, gscale, x, w, rstd, out_d, dres=out_d, lq=lq, lv=lv, mq=mq, mv=mv, keep_scale=ks,
                 out16=out16_d, delta=delta, dummy16=d16, dummy32=d32)
    assert torch.equal(out_d, out) and torch.equal(out16_d, out16)
    # v7: the forward without y (the backward's norm-1 recompute) writes the same rstd and masks
    y_keep = torch.full_like(y, 3.0)
    rstd3, xq3, xv3 = torch.empty_like(rstd), torch.empty_like(y), torch.empty_like(y)
    rms_forward(x, w, None, rstd3, eps, delta=delta, xq=xq3, xv=xv3, mq=mq, mv=mv, keep_scale=ks,
                dummy16=d16, dummy32=d32)
    rstd_ref = torch.empty_like(rstd)
    rms_forward(x, w, y_keep, rstd_ref, eps, delta=delta, xq=xq2, xv=xv2, mq=mq, mv=mv, keep_scale=ks,
                dummy16=d16, dummy32=d32)
    assert torch.equal(rstd3, rstd_ref) and torch.equal(xq3, xq2) and torch.equal(xv3, xv2)
    # dummy16 must be large enough for nothing: y=None never writes through the dummy
    assert torch.equal(d16, torch.zeros_like(d16))


# ------------------------------------------------------------------ QKV epilogue (LoRA add + RoPE)
def _rope_ref(x, cos, sin):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def test_qkv_epilogue_forward_backward():
    from solution.k_rope import qkv_epilogue_backward, qkv_epilogue_forward

    torch.manual_seed(2)
    nq, nkv, hd, n = 12, 2, 128, 257
    inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    pos = torch.randint(0, 512, (n,), device=DEV, dtype=torch.int32)
    fr = torch.arange(512, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), -1).to(DEV)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    qkv = torch.randn(n, (nq + 2 * nkv) * hd, device=DEV).half()
    lq = (torch.randn(n, nq * hd, device=DEV) * 0.1).half()
    lv = (torch.randn(n, nkv * hd, device=DEV) * 0.1).half()
    orig = qkv.clone()
    qkv_epilogue_forward(qkv, lq, lv, pos, cos, sin, 2.0, nq, nkv, hd)
    c, s = cos[pos.long()][:, None, :], sin[pos.long()][:, None, :]
    q = orig[:, :nq * hd].float().view(n, nq, hd) + lq.float().view(n, nq, hd) * 2.0
    k = orig[:, nq * hd:(nq + nkv) * hd].float().view(n, nkv, hd)
    v = orig[:, (nq + nkv) * hd:].float() + lv.float() * 2.0
    _close(qkv[:, :nq * hd].view(n, nq, hd), _rope_ref(q, c, s).half(), 2e-3, 2e-3, "q")
    _close(qkv[:, nq * hd:(nq + nkv) * hd].view(n, nkv, hd), _rope_ref(k, c, s).half(), 2e-3, 2e-3, "k")
    _close(qkv[:, (nq + nkv) * hd:], v.half(), 2e-3, 2e-3, "v")
    # backward: autograd through the reference rope and the GQA expansion
    dq = torch.randn(n, nq * hd, device=DEV)
    dkp = torch.randn(n, nq * hd, device=DEV)
    dvp = torch.randn(n, nq * hd, device=DEV)
    out = torch.empty_like(qkv)
    qkv_epilogue_backward(dq, dkp, dvp, pos, cos, sin, out, nq, nkv, hd, nq // nkv)
    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    vr = v.clone().view(n, nkv, hd).requires_grad_(True)
    g = nq // nkv
    loss = (_rope_ref(qr, c, s) * dq.view(n, nq, hd)).sum()
    loss = loss + (_rope_ref(kr, c, s).repeat_interleave(g, 1) * dkp.view(n, nq, hd)).sum()
    loss = loss + (vr.repeat_interleave(g, 1) * dvp.view(n, nq, hd)).sum()
    loss.backward()
    _close(out[:, :nq * hd], qr.grad.reshape(n, -1), 3e-3, 2e-3, "dq")
    _close(out[:, nq * hd:(nq + nkv) * hd], kr.grad.reshape(n, -1), 1e-2, 2e-3, "dk")
    _close(out[:, (nq + nkv) * hd:], vr.grad.reshape(n, -1), 1e-2, 2e-3, "dv")
    # SDPA layout: fp16 grads already reduced per KV head (GIN = 1)
    dk16 = dkp.view(n, nkv, g, hd).sum(2).reshape(n, -1).half()
    dv16 = dvp.view(n, nkv, g, hd).sum(2).reshape(n, -1).half()
    out1 = torch.empty_like(qkv)
    qkv_epilogue_backward(dq.half(), dk16, dv16, pos, cos, sin, out1, nq, nkv, hd, 1)
    _close(out1[:, :nq * hd], qr.grad.reshape(n, -1), 1e-2, 3e-3, "dq (fp16 in)")
    _close(out1[:, nq * hd:(nq + nkv) * hd], kr.grad.reshape(n, -1), 3e-2, 3e-3, "dk (reduced)")
    _close(out1[:, (nq + nkv) * hd:], vr.grad.reshape(n, -1), 3e-2, 3e-3, "dv (reduced)")


# ------------------------------------------------------------------ attention
def _meta(lens):
    from solution.k_attn import BWD_BLOCK, FWD_BLOCK_M, AttnMeta, block_table

    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    tf = block_table(lens, FWD_BLOCK_M)
    tb = block_table(lens, BWD_BLOCK)
    return AttnMeta(cu, torch.tensor(tf, dtype=torch.int32, device=DEV),
                    torch.tensor(tb, dtype=torch.int32, device=DEV), len(tf) // 2, len(tb) // 2, sum(lens))


@pytest.mark.parametrize("lens", [[1, 37, 64, 200], [512, 511, 3, 129]])
def test_varlen_causal_gqa_attention(lens):
    from solution.k_attn import attention_backward, attention_forward

    torch.manual_seed(3)
    nq, nkv, hd = 12, 2, 128
    g = nq // nkv
    n = sum(lens)
    qkv = (torch.randn(n, (nq + 2 * nkv) * hd, device=DEV) * 0.8).half()
    meta = _meta(lens)
    out, lse = attention_forward(qkv, meta, nq, nkv, hd)
    q = qkv[:, :nq * hd].float().view(n, nq, hd).requires_grad_(True)
    k = qkv[:, nq * hd:(nq + nkv) * hd].float().view(n, nkv, hd).requires_grad_(True)
    v = qkv[:, (nq + nkv) * hd:].float().view(n, nkv, hd).requires_grad_(True)
    outs, s0 = [], 0
    for L in lens:
        sl = slice(s0, s0 + L)
        qq = q[sl].transpose(0, 1)
        kk = k[sl].transpose(0, 1).repeat_interleave(g, 0)
        vv = v[sl].transpose(0, 1).repeat_interleave(g, 0)
        o = torch.nn.functional.scaled_dot_product_attention(qq[None], kk[None], vv[None], is_causal=True,
                                                             scale=1.0 / math.sqrt(hd))[0]
        outs.append(o.transpose(0, 1))
        s0 += L
    ref = torch.cat(outs).reshape(n, nq * hd)
    _close(out, ref, 3e-3, 3e-3, "attention out")
    dout = torch.randn(n, nq * hd, device=DEV).half()
    ref.backward(dout.float())
    dq, dkp, dvp = attention_backward(qkv, out, lse, dout, meta, nq, nkv, hd)
    _close(dq.view(n, nq, hd), q.grad, 3e-2, 2e-2, "dq")
    _close(dkp.view(n, nkv, g, hd).sum(2), k.grad, 3e-2, 2e-2, "dk")
    _close(dvp.view(n, nkv, g, hd).sum(2), v.grad, 3e-2, 2e-2, "dv")


@pytest.mark.parametrize("lens", [[1, 37, 64, 200], [512, 511, 3, 129]])
def test_varlen_efficient_attention_path(lens):
    """The default attention path: PyTorch's memory-efficient kernel with cu_seqlens on the packed
    tokens, against per-sequence fp32 SDPA; then its fp16 per-query-head dK/dV partials through our
    QKV epilogue backward (GIN = 6)."""
    from solution.k_attn import varlen_attention, varlen_attention_backward, varlen_selftest
    from solution.k_rope import qkv_epilogue_backward

    ok, err = varlen_selftest(torch.device(DEV))
    assert ok, f"varlen self-test failed ({err})"
    torch.manual_seed(6)
    nq, nkv, hd = 12, 2, 128
    g = nq // nkv
    n = sum(lens)
    qkv = (torch.randn(n, (nq + 2 * nkv) * hd, device=DEV) * 0.8).half()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    out, ctx = varlen_attention(qkv, cu, max(lens), nq, nkv, hd, need_lse=True)
    q = qkv[:, :nq * hd].float().view(n, nq, hd).requires_grad_(True)
    k = qkv[:, nq * hd:(nq + nkv) * hd].float().view(n, nkv, hd).requires_grad_(True)
    v = qkv[:, (nq + nkv) * hd:].float().view(n, nkv, hd).requires_grad_(True)
    outs, s0 = [], 0
    for L in lens:
        sl = slice(s0, s0 + L)
        qq = q[sl].transpose(0, 1)
        kk = k[sl].transpose(0, 1).repeat_interleave(g, 0)
        vv = v[sl].transpose(0, 1).repeat_interleave(g, 0)
        o = torch.nn.functional.scaled_dot_product_attention(qq[None], kk[None], vv[None], is_causal=True,
                                                             scale=1.0 / math.sqrt(hd))[0]
        outs.append(o.transpose(0, 1))
        s0 += L
    ref = torch.cat(outs).reshape(n, nq * hd)
    _close(out, ref, 3e-3, 3e-3, "attention out")
    dout = torch.randn(n, nq * hd, device=DEV).half()
    ref.backward(dout.float())
    dq, dkp, dvp = varlen_attention_backward(ctx, dout, cu, max(lens), nq, nkv, hd)
    assert dq.is_contiguous() and dkp.shape == (n, nq * hd) and dkp.dtype == torch.float16
    _close(dq.view(n, nq, hd), q.grad, 3e-2, 2e-2, "dq")
    _close(dkp.float().view(n, nkv, g, hd).sum(2), k.grad, 3e-2, 2e-2, "dk")
    _close(dvp.float().view(n, nkv, g, hd).sum(2), v.grad, 3e-2, 2e-2, "dv")
    # the epilogue backward sums the fp16 partials exactly like it sums fp32 ones
    pos = torch.cat([torch.arange(L) for L in lens]).to(DEV).int()
    cos = torch.randn(1024, hd, device=DEV)
    sin = torch.randn(1024, hd, device=DEV)
    o16 = torch.empty_like(qkv)
    o32 = torch.empty_like(qkv)
    qkv_epilogue_backward(dq, dkp, dvp, pos, cos, sin, o16, nq, nkv, hd, g)
    qkv_epilogue_backward(dq.float(), dkp.float(), dvp.float(), pos, cos, sin, o32, nq, nkv, hd, g)
    assert torch.equal(o16, o32)


# ------------------------------------------------------------------ gated SiLU
def test_gated_silu_forward_backward():
    from solution.k_mlp import gated_silu_backward, gated_silu_forward

    torch.manual_seed(4)
    n, inter = 333, 8960
    gu = (torch.randn(n, 2 * inter, device=DEV) * 2).half()
    act = torch.empty(n, inter, dtype=torch.float16, device=DEV)
    gated_silu_forward(gu, act)
    g = gu[:, :inter].float().requires_grad_(True)
    u = gu[:, inter:].float().requires_grad_(True)
    ref = torch.nn.functional.silu(g) * u
    _close(act, ref.half(), 2e-3, 2e-3, "act")
    da = torch.randn(n, inter, device=DEV).half()
    ref.backward(da.float())
    buf = gu.clone()
    gated_silu_backward(buf, da)
    _close(buf[:, :inter], g.grad.half(), 3e-3, 3e-3, "d gate")
    _close(buf[:, inter:], u.grad.half(), 3e-3, 3e-3, "d up")


# ------------------------------------------------------------------ LM head rows
def test_head_rows_stats_and_grad():
    from solution.k_head import head_rows

    torch.manual_seed(5)
    c, vocab = 37, 151936
    logits = (torch.randn(c, vocab, device=DEV) * 4).half()
    logits[3, 10] = logits[3, 100000] = 40.0                    # tie: first index wins
    tgt = torch.randint(0, vocab, (c,), device=DEV, dtype=torch.int32)
    tgt[[1, 5, 9]] = -1
    tgt[3] = 10
    x = logits.float()
    stats = torch.empty(c, 4, device=DEV)
    work = logits.clone()
    head_rows(work, tgt, stats, 0, c, 1, 1024.0)
    lse = torch.logsumexp(x, -1)
    logp = torch.log_softmax(x, -1)
    ent = -(logp.exp() * logp).sum(-1)
    valid = tgt >= 0
    ce = torch.nn.functional.cross_entropy(x, tgt.long().clamp(min=0), reduction="none")
    _close(stats[:, 0], torch.where(valid, ce, torch.zeros_like(ce)), 1e-4, 1e-5, "loss")
    _close(stats[:, 1], ent, 2e-4, 1e-4, "entropy")
    hit = (x.argmax(-1) == tgt.long()) & valid
    assert torch.equal(stats[:, 2] > 0.5, hit)
    _close(stats[:, 3], lse, 1e-4, 1e-6, "lse")
    grad = (logp.exp() - torch.nn.functional.one_hot(tgt.long().clamp(min=0), vocab).float()) * 1024.0
    _close(work[valid], grad[valid].half(), 2e-2, 3e-3, "grad")
    assert torch.equal(work[~valid], logits[~valid])


# ------------------------------------------------------------------ LM head over vocabulary chunks (v6)
def test_fp32_accumulating_gemm():
    """acc (fp32) += a16 @ b16 in place, through torch's addmm with an fp32 out_dtype."""
    from solution import k_head

    torch.manual_seed(8)
    a = torch.randn(77, 4096, device=DEV).half()
    b = (torch.randn(4096, 1536, device=DEV) * 0.1).half()
    acc = torch.randn(77, 1536, device=DEV)
    ref = acc.double() + a.double() @ b.double()
    k_head._accumulate(acc, a, b)
    assert acc.dtype == torch.float32
    _close(acc, ref, 2e-3, 1e-4, "fp32-accumulated GEMM")
    assert k_head._ACC_OK[0], "torch.addmm(out_dtype=float32) unavailable: fell back to an fp16 product"


@pytest.mark.parametrize("stream", [False, True])
def test_streamed_head_matches_torch_and_v5(stream):
    """The v6 head (vocabulary chunks, online softmax, gradient accumulated in the pass), with the
    embedding on the GPU or streamed from pinned host memory, against fp64 torch math on the same
    fp16 logits and against the v5 head."""
    from solution.k_head import EmbeddingStream, ResidentEmbedding, fused_head, streamed_head

    torch.manual_seed(7)
    V, D, n = 151936, 1536, 300
    emb = (torch.randn(V, D, device=DEV) * 0.02).half()
    yf = (torch.randn(n, D, device=DEV) * 2.0).half()
    tgt = torch.randint(0, V, (n,), device=DEV)
    tgt[torch.randperm(n, device=DEV)[:90]] = -1
    tgt[7] = int((yf[7].float() @ emb.float().t()).argmax())        # a hit
    valid = tgt >= 0
    perm = torch.cat([valid.nonzero().flatten(), (~valid).nonzero().flatten()])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=DEV)
    tgt_sorted = tgt[perm].int()
    n_lab = int(valid.sum())
    e_target = emb.index_select(0, tgt_sorted[:n_lab])
    dummy = torch.zeros(16, device=DEV)
    if stream:
        src = EmbeddingStream(emb.cpu().pin_memory(), 4096, 3, torch.device(DEV))
        src.start()
        rows = src.rows(None, tgt_sorted[:n_lab].cpu().long())
        assert torch.equal(rows, e_target)
    else:
        src = ResidentEmbedding(emb, 4096)
    loss, stats, dy = streamed_head(yf, src, tgt_sorted, perm.int(), n_lab, e_target, 1024.0, dummy)
    if stream:
        assert src.copies == src.n_chunks, (src.copies, src.n_chunks)
    logits = (yf @ emb.t()).float()                  # fp16 GEMM output, as the base computes it
    ys = logits[perm].double()
    lse = torch.logsumexp(ys, -1)
    logp = torch.log_softmax(ys, -1)
    ent = -(logp.exp() * logp).sum(-1)
    t = tgt_sorted.long()
    ok = t >= 0
    xt = ys.gather(1, t.clamp(min=0)[:, None])[:, 0]
    ce = torch.where(ok, lse - xt, torch.zeros_like(lse))
    _close(stats[:, 0], ce, 2e-4, 1e-5, "loss")
    _close(stats[:, 1], ent, 3e-4, 1e-4, "entropy")
    assert torch.equal(stats[:, 2] > 0.5, (ys.argmax(-1) == t) & ok)
    assert bool(stats[:, 2].sum() >= 1)
    _close(stats[:, 3], lse, 2e-4, 1e-6, "lse")
    assert abs(float(loss) - float(ce.sum())) <= 1e-4 * float(ce.sum())
    g = logp.exp()
    g[ok, t[ok]] -= 1.0
    g[~ok] = 0.0
    ref = torch.zeros(n, D, dtype=torch.float32, device=DEV)
    ref[perm] = (g.float() @ emb.float()) * 1024.0
    _close(dy, ref, 3e-2, 5e-3, "d(hidden)")
    l5, s5, d5 = fused_head(yf, emb, tgt_sorted, perm.int(), inv.int(), n_lab, 256, 1024.0)
    _close(stats, s5, 3e-4, 1e-4, "stats vs the v5 head")
    _close(dy, d5, 5e-2, 1e-2, "d(hidden) vs the v5 head")
    # a second pass through the same ring (the next step) gives the same result
    if stream:
        src.start()
        loss2, stats2, dy2 = streamed_head(yf, src, tgt_sorted, perm.int(), n_lab, e_target, 1024.0, dummy)
        _close(stats2, stats, 1e-6, 1e-6, "second step: stats")
        _close(dy2, dy, 1e-3, 1e-3, "second step: d(hidden)")
