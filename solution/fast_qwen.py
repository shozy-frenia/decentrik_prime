"""Our own forward and backward of the 4-bit PEFT Qwen2 model for Soup's SFT step.

What the step computes is unchanged: the same NF4 weights (dequantised bit-exactly), the same
LoRA (q_proj, v_proj, r=16, alpha=32, dropout 0.05) on the same fp32 parameters, the same fp32
residual stream and fp16 GEMMs as the base run, the same summed cross-entropy over the assistant
tokens. How it is computed changes:

* padding-free: the real tokens of the batch are packed into one [N, 1536] sequence (N is about
  half of batch x padded length), with per-row positions and a block-diagonal causal mask in our
  attention kernel; padding is never computed;
* one autograd node for the whole model: the forward keeps only each layer's fp32 residual input
  (6 KB per token per layer, optionally in pinned host memory) and the head's d(hidden); the
  backward re-runs each layer's forward from its checkpoint and back-propagates through it by
  hand, so at most one layer's activations are alive at a time;
* fused kernels for everything that is not a GEMM (see k_*.py); GEMMs are cuBLAS on fp16
  weights that our NF4 kernel dequantises once per layer per pass into one workspace, with Q|K|V
  and gate|up fused into single GEMMs;
* the LM head never materialises the full logits (k_head.py) and computes its gradient in the
  forward, scaled by the incoming grad in the backward;
* the tied embedding is stored in fp16 (bit-exact, see integration.py).
"""

import math

import torch

from . import config
from .k_attn import (BWD_BLOCK, FWD_BLOCK_M, AttnMeta, attention_backward, attention_forward,
                     block_table)
from .k_head import head_rows
from .k_mlp import gated_silu_backward, gated_silu_forward
from .k_nf4 import (NF4_CODE, WeightStreamer, dequant_layer, dequant_tensors, layer_numel,
                    repack_layer, section_layout)
from .k_norm import rms_backward, rms_forward
from .k_rope import qkv_epilogue_backward, qkv_epilogue_forward

ALIGN = 16  # int32 elements: keeps every packed metadata slice 64-byte aligned for Triton


class Unsupported(Exception):
    """The model or run is not the configuration the fast path implements."""


def _require(cond, msg):
    if not cond:
        raise Unsupported(msg)


def _lora_parts(module, adapter):
    """(A, B, dropout p, scaling) of a PEFT LoRA layer, checked to be plain LoRA."""
    for attr in ("lora_A", "lora_B", "lora_dropout", "scaling", "base_layer"):
        _require(hasattr(module, attr), f"{type(module).__name__} is not a PEFT LoRA layer")
    _require(list(module.active_adapters) == [adapter], "exactly one active adapter expected")
    _require(not getattr(module, "merged", False), "merged adapter")
    _require(not getattr(module, "disable_adapters", False), "adapters disabled")
    use_dora = getattr(module, "use_dora", {})
    _require(not (use_dora.get(adapter, False) if isinstance(use_dora, dict) else use_dora), "DoRA")
    lora_bias = getattr(module, "lora_bias", {})
    _require(not (lora_bias.get(adapter, False) if isinstance(lora_bias, dict) else lora_bias),
             "LoRA bias")
    variants = getattr(module, "lora_variant", {})
    _require(adapter not in variants, "LoRA variant")
    a, b = module.lora_A[adapter], module.lora_B[adapter]
    _require(isinstance(a, torch.nn.Linear) and isinstance(b, torch.nn.Linear), "LoRA A/B not Linear")
    _require(a.bias is None and b.bias is None, "LoRA A/B with bias")
    drop = module.lora_dropout[adapter]
    if isinstance(drop, torch.nn.Dropout):
        p = float(drop.p)
    else:
        _require(isinstance(drop, torch.nn.Identity), f"unknown LoRA dropout {type(drop).__name__}")
        p = 0.0
    _require(a.weight.dtype == torch.float32 and b.weight.dtype == torch.float32,
             "LoRA weights are expected in fp32")
    _require(a.weight.requires_grad and b.weight.requires_grad, "LoRA weights must be trainable")
    return a.weight, b.weight, p, float(module.scaling[adapter])


def _is_bnb_4bit(module):
    return type(module).__name__ in ("Linear4bit", "LinearNF4") and hasattr(module.weight, "quant_state")


class FastQwen2:
    """Holds the repacked model and runs the fused forward / backward."""

    @torch.no_grad()
    def __init__(self, peft_model, seed=0):
        self.peft_model = peft_model
        base = peft_model.get_base_model()
        _require(type(base).__name__ == "Qwen2ForCausalLM", f"model {type(base).__name__}")
        cfg = base.config
        self.cfg = cfg
        qwen = base.model
        self.L = int(cfg.num_hidden_layers)
        self.D = int(cfg.hidden_size)
        self.NQ = int(cfg.num_attention_heads)
        self.NKV = int(cfg.num_key_value_heads)
        self.HD = int(getattr(cfg, "head_dim", None) or self.D // self.NQ)
        self.I = int(cfg.intermediate_size)
        self.V = int(cfg.vocab_size)
        self.eps = float(cfg.rms_norm_eps)
        _require(self.NQ * self.HD == self.D and self.NQ % self.NKV == 0, "head layout")
        _require(self.HD == 128 and self.D == 1536 and self.NKV == 2, "Qwen2.5-1.5B shapes expected")
        _require(getattr(cfg, "hidden_act", "silu") == "silu", "hidden_act must be silu")
        _require(float(getattr(cfg, "attention_dropout", 0.0) or 0.0) == 0.0, "attention dropout")
        layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * self.L
        _require(all(t == "full_attention" for t in layer_types), "sliding-window layers")
        _require(len(qwen.layers) == self.L, "layer count")
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        rope_type = rope.get("rope_type", rope.get("type", "default")) if isinstance(rope, dict) else "default"
        _require(rope_type == "default", f"rope type {rope_type}")

        self.adapter = peft_model.active_adapter
        if isinstance(self.adapter, (list, tuple)):
            _require(len(self.adapter) == 1, "one adapter")
            self.adapter = self.adapter[0]
        emb = qwen.embed_tokens
        _require(base.lm_head.weight is emb.weight or
                 base.lm_head.weight.data_ptr() == emb.weight.data_ptr(), "lm_head must be tied")
        _require(getattr(base.lm_head, "bias", None) is None, "lm_head bias")
        _require(not emb.weight.requires_grad, "embedding must be frozen")
        dev = emb.weight.device
        _require(dev.type == "cuda", "model must be on CUDA")
        self.device = dev
        _require(emb.weight.shape == (self.V, self.D), "embedding shape")

        # ---- phase 1: validate everything; nothing is modified until all checks passed
        self.w_in, self.w_post, self.bias_qkv = [], [], []
        self.lora_params = []          # flat list: Aq, Bq, Av, Bv per layer
        self._linears = []
        p_seen, scale_seen = set(), set()
        code_ref = None
        for i, layer in enumerate(qwen.layers):
            at, mlp = layer.self_attn, layer.mlp
            q, k, v = at.q_proj, at.k_proj, at.v_proj
            aq, bq, pq, sq = _lora_parts(q, self.adapter)
            av, bv, pv, sv = _lora_parts(v, self.adapter)
            _require(aq.shape == (bq.shape[1], self.D) and bq.shape[0] == self.NQ * self.HD, "q LoRA shape")
            _require(av.shape == (bv.shape[1], self.D) and bv.shape[0] == self.NKV * self.HD, "v LoRA shape")
            _require(aq.is_cuda and av.is_cuda, "LoRA weights must be on the GPU")
            p_seen.update((pq, pv))
            scale_seen.update((sq, sv))
            linears = {"q_proj": q.base_layer, "k_proj": k, "v_proj": v.base_layer, "o_proj": at.o_proj,
                       "gate_proj": mlp.gate_proj, "up_proj": mlp.up_proj, "down_proj": mlp.down_proj}
            for name, mod in linears.items():
                _require(_is_bnb_4bit(mod), f"layer {i} {name} is not a bitsandbytes 4-bit linear")
                _require(not hasattr(mod, "lora_A"), f"layer {i} {name} has an unexpected adapter")
                _require(mod.weight.data.is_cuda, f"layer {i} {name} weight is not on the GPU")
                code = mod.weight.quant_state.state2.code if getattr(mod.weight.quant_state, "nested", False) else None
                _require(code is not None, f"layer {i} {name}: no double quantisation")
                if code_ref is None:
                    code_ref = code
                _require(torch.equal(code_ref, code), "second-level code tables differ")
            try:
                n_layer = layer_numel(linears)
            except ValueError as exc:
                raise Unsupported(f"layer {i}: {exc}") from exc
            _require(n_layer % (64 * 256) == 0, "layer size")
            for name in ("o_proj", "gate_proj", "up_proj", "down_proj"):
                _require(linears[name].bias is None, f"{name} bias")
            biases = [linears[n].bias for n in ("q_proj", "k_proj", "v_proj")]
            _require(all(b is not None for b in biases), "q/k/v biases expected")
            bias16 = torch.cat([b.data.to(torch.float16) for b in biases])
            _require(torch.equal(bias16.float(), torch.cat([b.data.float() for b in biases])),
                     "q/k/v bias not fp16-representable")
            _require(abs(layer.input_layernorm.variance_epsilon - self.eps) < 1e-12, "norm eps")
            self._linears.append(linears)
            self.bias_qkv.append(bias16)
            self.w_in.append(layer.input_layernorm.weight.data.float().contiguous())
            self.w_post.append(layer.post_attention_layernorm.weight.data.float().contiguous())
            self.lora_params += [aq, bq, av, bv]
        _require(len(p_seen) == 1 and len(scale_seen) == 1, "one dropout p and LoRA scaling expected")
        w = emb.weight
        if w.dtype != torch.float16:
            _require(torch.equal(w.data.to(torch.float16).float(), w.data.float()),
                     "embedding not fp16-representable")

        # ---- phase 2: fp16 embedding (bit-exact: the T4 load path stores it as fp16 before the
        # fp32 upcast) and repacked 4-bit layers (optionally in pinned host memory)
        if w.dtype != torch.float16:
            w.data = w.data.to(torch.float16)
            torch.cuda.empty_cache()
        self.emb = w.data
        host_all, layer_bytes = None, 0
        if config.OFFLOAD_WEIGHTS:
            _, layer_bytes = section_layout(layer_numel(self._linears[0]) // 64)
            host_all = torch.empty(self.L * layer_bytes, dtype=torch.uint8, pin_memory=True)
        self.packed = []
        code2 = None
        for i, linears in enumerate(self._linears):
            out = host_all[i * layer_bytes:(i + 1) * layer_bytes] if host_all is not None else None
            packed, c2 = repack_layer(linears, dev, host=host_all is not None, out=out)
            code2 = c2 if code2 is None else code2
            self.packed.append(packed)
        self.p_drop = p_seen.pop()
        self.lora_scale = scale_seen.pop()
        self.keep_scale = 1.0 / (1.0 - self.p_drop)
        self.code2 = code2
        self.streamer = WeightStreamer(self.packed, dev) if config.OFFLOAD_WEIGHTS else None
        self.nf4 = torch.tensor(NF4_CODE, dtype=torch.float32, device=dev)
        self.w_final = qwen.norm.weight.data.float().contiguous()
        self.rank = int(self.lora_params[0].shape[0])

        # one fp16 workspace for the dequantised weights of a layer, with fixed views
        seg = self.packed[0].segments
        for p in self.packed:
            _require(p.segments == seg, "layers differ in shape")
        self.ws = torch.empty(self.packed[0].numel, dtype=torch.float16, device=dev)
        nqkv = (self.NQ + 2 * self.NKV) * self.HD

        def view(name, rows, cols):
            start = seg[name][0]
            return self.ws[start:start + rows * cols].view(rows, cols)

        _require(seg["k_proj"][0] == seg["q_proj"][0] + self.D * self.D and
                 seg["v_proj"][0] == seg["k_proj"][0] + self.NKV * self.HD * self.D, "qkv order")
        _require(seg["up_proj"][0] == seg["gate_proj"][0] + self.I * self.D, "gate/up order")
        self.W_qkv = view("q_proj", nqkv, self.D)
        self.W_o = view("o_proj", self.D, self.D)
        self.W_gu = view("gate_proj", 2 * self.I, self.D)
        self.W_d = view("down_proj", self.D, self.I)

        # rotary tables from the model's own rotary module (bit-identical cos/sin)
        self.rotary = qwen.rotary_emb
        self.cos = self.sin = None
        self._rope_table(1024)

        # fp16 shadows of the LoRA weights, refreshed once per step
        self.lora16 = [torch.empty(p.shape, dtype=torch.float16, device=dev) for p in self.lora_params]
        self.dummy16 = torch.zeros(16, dtype=torch.float16, device=dev)
        self.dummy32 = torch.zeros(16, dtype=torch.float32, device=dev)
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed) & 0x7FFFFFFF)
        self.copy_stream = None
        self.host_ckpt = None
        self.steps = 0

    @torch.no_grad()
    def restore_stock(self):
        """Undo what the fast path changed, for the stock forward: 4-bit weights back on the GPU (if
        they were moved to host memory) and the fp32 embedding the stock path expects."""
        if self.streamer is not None:
            for i, linears in enumerate(self._linears):
                self.packed[i], _ = repack_layer(linears, self.device, host=False)
            self.streamer = None
        w = self.peft_model.get_base_model().model.embed_tokens.weight
        if w.dtype == torch.float16:
            w.data = w.data.float()

    # ------------------------------------------------------------------ helpers
    def _rope_table(self, n):
        with torch.no_grad():
            pos = torch.arange(n, device=self.device)[None, :]
            x = torch.zeros(1, dtype=torch.float32, device=self.device)
            cos, sin = self.rotary(x, pos)
        self.cos = cos[0].float().contiguous()
        self.sin = sin[0].float().contiguous()
        _require(self.cos.shape == (n, self.HD), "rotary table shape")

    @torch.no_grad()
    def _refresh_lora16(self):
        try:
            torch._foreach_copy_(self.lora16, self.lora_params)
        except Exception:  # noqa: BLE001 - older torch: plain copies
            for dst, src in zip(self.lora16, self.lora_params):
                dst.copy_(src)

    def _dequant(self, i, nxt=None):
        """fp16 weights of layer i into the workspace; with streaming, prefetch layer `nxt`."""
        if self.streamer is None:
            dequant_layer(self.packed[i], self.code2, self.nf4, self.ws)
            return
        pk, qa, s2, o2 = self.streamer.acquire(i)
        dequant_tensors(pk, qa, s2, o2, self.packed[i].n_qblocks, self.code2, self.nf4, self.ws)
        self.streamer.release()
        self.streamer.prefetch(nxt)

    def _seeds(self):
        return torch.randint(1, 2 ** 31 - 1, (2 * self.L,), generator=self.gen).tolist()

    # ------------------------------------------------------------------ batch packing
    def pack(self, input_ids, attention_mask, labels):
        """Host-side packing of a right-padded batch. Returns None if the batch is not one we can
        pack exactly (then the caller uses the stock path)."""
        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
            return None
        B, T = input_ids.shape
        am = attention_mask.to("cpu", torch.int64)
        lens = am.sum(1)
        ar = torch.arange(T)
        if not torch.equal(am, (ar[None, :] < lens[:, None]).to(torch.int64)):
            return None                       # not a contiguous right-padded mask
        lens_l = [int(x) for x in lens.tolist()]
        if min(lens_l) < 1:
            return None
        n = sum(lens_l)
        if max(lens_l) > self.cos.shape[0]:
            self._rope_table(1 << (max(lens_l) - 1).bit_length())
        sel = am.bool()
        ids = input_ids.to("cpu")[sel]
        lab = labels.to("cpu", torch.int64)
        nxt = torch.full((B, T), -100, dtype=torch.int64)
        nxt[:, :-1] = lab[:, 1:]
        tgt = nxt[sel]
        tgt = torch.where(tgt < 0, torch.full_like(tgt, -1), tgt)
        if int(tgt.max()) >= self.V or int(ids.max()) >= self.V or int(ids.min()) < 0:
            return None
        pos = ar[None, :].expand(B, T)[sel]
        row = torch.arange(B)[:, None].expand(B, T)[sel]
        valid = tgt >= 0
        perm = torch.cat([valid.nonzero().flatten(), (~valid).nonzero().flatten()])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        cu = [0]
        for x in lens_l:
            cu.append(cu[-1] + x)
        tab_f = block_table(lens_l, FWD_BLOCK_M)
        tab_b = block_table(lens_l, BWD_BLOCK)
        parts = [ids, pos, tgt[perm], perm, inv, torch.tensor(cu), torch.tensor(tab_f), torch.tensor(tab_b)]
        offs, chunks, at = [], [], 0
        for p in parts:
            p = p.to(torch.int32).flatten()
            pad = (-p.numel()) % ALIGN
            offs.append((at, p.numel()))
            chunks.append(p)
            if pad:
                chunks.append(torch.zeros(pad, dtype=torch.int32))
            at += p.numel() + pad
        host = torch.cat(chunks)
        dev = host.to(self.device)
        views = [dev[o:o + c] for o, c in offs]
        batch = _Batch()
        batch.n, batch.B, batch.lens = n, B, lens_l
        batch.ids, batch.pos, batch.tgt_sorted, batch.perm, batch.inv = views[:5]
        batch.meta = AttnMeta(views[5], views[6], views[7], len(tab_f) // 2, len(tab_b) // 2, n)
        batch.n_lab = int(valid.sum())
        batch.Tm = max(lens_l)
        if config.ATTN_IMPL == "sdpa":
            pad_index = row * batch.Tm + pos            # packed token -> row of a [B*Tm] padded view
            batch.pad_index = pad_index.to(self.device)
        batch.row_sorted = row[perm]
        batch.valid_sorted = valid[perm]
        batch.n_real = n
        return batch

    # ------------------------------------------------------------------ checkpoint storage
    def _store(self, n):
        if config.OFFLOAD_CHECKPOINTS:
            if self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(device=self.device)
            return _HostStore(self, n)
        return _GpuStore()

    # ------------------------------------------------------------------ one layer, forward part
    def _attn_part(self, i, x_norm, xq, xv, batch, grad=False):
        """QKV (+LoRA, RoPE) and attention of layer i; x_norm/xq/xv from the norm kernel."""
        aq16, bq16, av16, bv16 = self.lora16[4 * i:4 * i + 4]
        aq = torch.mm(xq, aq16.t())
        av = torch.mm(xv, av16.t())
        lq = torch.mm(aq, bq16.t())
        lv = torch.mm(av, bv16.t())
        qkv = torch.addmm(self.bias_qkv[i], x_norm, self.W_qkv.t())
        qkv_epilogue_forward(qkv, lq, lv, batch.pos, self.cos, self.sin, self.lora_scale,
                             self.NQ, self.NKV, self.HD)
        del lq, lv
        out, actx = self._attention(qkv, batch, grad)
        return qkv, aq, av, out, actx

    # ------------------------------------------------------------------ attention
    def _attention(self, qkv, batch, grad):
        """Causal attention of the packed batch. Returns (out [N, NQ*HD] fp16, context for backward).

        ATTN_IMPL == "triton": our varlen GQA flash-attention kernels (k_attn.py).
        ATTN_IMPL == "sdpa": PyTorch's memory-efficient SDPA (what base Soup runs) on a zero-padded
        [B, H, Tmax, 128] view of the packed tokens; right padding needs no mask beyond causality.
        In the backward recompute (grad=True) the SDPA call records a local autograd graph that
        `_attention_backward` differentiates."""
        if config.ATTN_IMPL == "triton":
            return attention_forward(qkv, batch.meta, self.NQ, self.NKV, self.HD)
        n, B, T = batch.n, batch.B, batch.Tm
        H, KV, HD = self.NQ, self.NKV, self.HD
        idx = batch.pad_index

        def padded(cols, heads):
            buf = qkv.new_zeros((B * T, heads, HD))
            buf.index_copy_(0, idx, qkv[:, cols].reshape(n, heads, HD))
            if grad:
                buf.requires_grad_(True)
            return buf

        qb = padded(slice(0, H * HD), H)
        kb = padded(slice(H * HD, (H + KV) * HD), KV)
        vb = padded(slice((H + KV) * HD, (H + 2 * KV) * HD), KV)
        g = H // KV
        with torch.set_grad_enabled(grad):
            q = qb.view(B, T, H, HD).transpose(1, 2)
            k = kb.view(B, T, KV, HD).transpose(1, 2)[:, :, None].expand(B, KV, g, T, HD).reshape(B, H, T, HD)
            v = vb.view(B, T, KV, HD).transpose(1, 2)[:, :, None].expand(B, KV, g, T, HD).reshape(B, H, T, HD)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                                 scale=1.0 / math.sqrt(HD))
        out = o.detach().transpose(1, 2).reshape(B * T, H * HD).index_select(0, idx)
        return out, ((o, qb, kb, vb) if grad else None)

    def _attention_backward(self, qkv, out, actx, dout, batch):
        """(dq, dk, dv, GIN) for the QKV epilogue backward."""
        if config.ATTN_IMPL == "triton":
            dq, dkp, dvp = attention_backward(qkv, out, actx, dout, batch.meta, self.NQ, self.NKV, self.HD)
            return dq, dkp, dvp, self.NQ // self.NKV
        o, qb, kb, vb = actx
        n, B, T = batch.n, batch.B, batch.Tm
        H, KV, HD = self.NQ, self.NKV, self.HD
        dpad = dout.new_zeros((B * T, H * HD))
        dpad.index_copy_(0, batch.pad_index, dout)
        dq, dk, dv = torch.autograd.grad(o, (qb, kb, vb), dpad.view(B, T, H, HD).transpose(1, 2))
        idx = batch.pad_index
        return (dq.reshape(B * T, H * HD).index_select(0, idx), dk.reshape(B * T, KV * HD).index_select(0, idx),
                dv.reshape(B * T, KV * HD).index_select(0, idx), 1)

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        n, D = batch.n, self.D
        dev = self.device
        f16, f32 = torch.float16, torch.float32
        self._refresh_lora16()
        seeds = self._seeds()
        store = self._store(n)
        y = torch.empty((n, D), dtype=f16, device=dev)
        xq = torch.empty((n, D), dtype=f16, device=dev)
        xv = torch.empty((n, D), dtype=f16, device=dev)
        y2 = torch.empty((n, D), dtype=f16, device=dev)
        rstd = torch.empty(n, dtype=f32, device=dev)
        x = self.emb.index_select(0, batch.ids).float()
        d16 = None
        for i in range(self.L):
            if i == 0:
                rms_forward(x, self.w_in[0], y, rstd, self.eps, xq=xq, xv=xv, p_drop=self.p_drop,
                            keep_scale=self.keep_scale, seed_q=seeds[0], seed_v=seeds[1],
                            dummy16=self.dummy16, dummy32=self.dummy32)
            else:
                x_new = torch.empty((n, D), dtype=f32, device=dev)
                rms_forward(x, self.w_in[i], y, rstd, self.eps, delta=d16, xout=x_new, xq=xq, xv=xv,
                            p_drop=self.p_drop, keep_scale=self.keep_scale, seed_q=seeds[2 * i],
                            seed_v=seeds[2 * i + 1], dummy16=self.dummy16, dummy32=self.dummy32)
                x = x_new
            store.put(i, x)
            self._dequant(i, i + 1 if i + 1 < self.L else None)
            qkv, _aq, _av, out, _actx = self._attn_part(i, y, xq, xv, batch)
            del qkv, _aq, _av, _actx
            op = torch.mm(out, self.W_o.t())
            del out
            h1 = torch.empty((n, D), dtype=f32, device=dev)
            rms_forward(x, self.w_post[i], y2, rstd, self.eps, delta=op, xout=h1,
                        dummy16=self.dummy16, dummy32=self.dummy32)
            del op
            gu = torch.mm(y2, self.W_gu.t())
            act = torch.empty((n, self.I), dtype=f16, device=dev)
            gated_silu_forward(gu, act)
            del gu
            d16 = torch.mm(act, self.W_d.t())
            del act
            x = h1
        del y, xq, xv, y2
        hf = torch.empty((n, D), dtype=f32, device=dev)
        yf = torch.empty((n, D), dtype=f16, device=dev)
        rstd_f = torch.empty(n, dtype=f32, device=dev)
        rms_forward(x, self.w_final, yf, rstd_f, self.eps, delta=d16, xout=hf,
                    dummy16=self.dummy16, dummy32=self.dummy32)
        del x, d16
        loss_sum, stats, dy_head = self._head(yf, batch)
        del yf
        self.steps += 1
        saved = dict(store=store, hf=hf, rstd_f=rstd_f, dy_head=dy_head, seeds=seeds, batch=batch)
        return loss_sum, stats, saved

    def _head(self, yf, batch):
        n, nl, D, V = batch.n, batch.n_lab, self.D, self.V
        dev = self.device
        ys = yf.index_select(0, batch.perm)          # tokens with a target first
        stats = torch.empty((n, 4), dtype=torch.float32, device=dev)
        dys = torch.zeros((n, D), dtype=torch.float16, device=dev)
        chunk = max(16, int(config.HEAD_CHUNK))
        buf = torch.empty((min(chunk, n), V), dtype=torch.float16, device=dev)
        spans = [(s, min(s + chunk, nl), 1) for s in range(0, nl, chunk)]
        spans += [(s, min(s + chunk, n), 0) for s in range(nl, n, chunk)]
        for s, e, grad in spans:
            c = e - s
            logits = buf[:c]
            torch.mm(ys[s:e], self.emb.t(), out=logits)
            head_rows(logits, batch.tgt_sorted, stats, s, c, grad, config.HEAD_GRAD_SCALE)
            if grad:
                torch.mm(logits, self.emb, out=dys[s:e])
        del buf, ys
        dy = dys.index_select(0, batch.inv)
        loss_sum = stats[:, 0].sum()
        return loss_sum, stats, dy

    # ------------------------------------------------------------------ backward
    def backward(self, saved, g_loss):
        batch, store, seeds = saved["batch"], saved["store"], saved["seeds"]
        n, D = batch.n, self.D
        dev = self.device
        f16, f32 = torch.float16, torch.float32
        gscale = (g_loss.detach().reshape(1).to(f32) / float(config.HEAD_GRAD_SCALE)).contiguous()
        dh = torch.empty((n, D), dtype=f32, device=dev)
        rms_backward(saved.pop("dy_head"), gscale, saved.pop("hf"), self.w_final, saved.pop("rstd_f"),
                     dh, dummy16=self.dummy16, dummy32=self.dummy32)
        y = torch.empty((n, D), dtype=f16, device=dev)
        xq = torch.empty((n, D), dtype=f16, device=dev)
        xv = torch.empty((n, D), dtype=f16, device=dev)
        y2 = torch.empty((n, D), dtype=f16, device=dev)
        rstd1 = torch.empty(n, dtype=f32, device=dev)
        rstd2 = torch.empty(n, dtype=f32, device=dev)
        grads = [None] * len(self.lora_params)
        s = self.lora_scale
        store.prefetch(self.L - 1)
        for i in reversed(range(self.L)):
            x = store.get(i)
            if i > 0:
                store.prefetch(i - 1)
            aq16, bq16, av16, bv16 = self.lora16[4 * i:4 * i + 4]
            self._dequant(i, i - 1 if i > 0 else None)
            # recompute the layer's forward from its checkpoint
            rms_forward(x, self.w_in[i], y, rstd1, self.eps, xq=xq, xv=xv, p_drop=self.p_drop,
                        keep_scale=self.keep_scale, seed_q=seeds[2 * i], seed_v=seeds[2 * i + 1],
                        dummy16=self.dummy16, dummy32=self.dummy32)
            qkv, aq, av, out, actx = self._attn_part(i, y, xq, xv, batch, grad=True)
            op = torch.mm(out, self.W_o.t())
            h1 = torch.empty((n, D), dtype=f32, device=dev)
            rms_forward(x, self.w_post[i], y2, rstd2, self.eps, delta=op, xout=h1,
                        dummy16=self.dummy16, dummy32=self.dummy32)
            del op
            gu = torch.mm(y2, self.W_gu.t())
            # MLP backward
            dact = torch.mm(dh.to(f16), self.W_d)
            gated_silu_backward(gu, dact)
            del dact
            dy2 = torch.mm(gu, self.W_gu)
            del gu
            rms_backward(dy2, self.one, h1, self.w_post[i], rstd2, dh, dres=dh,
                         dummy16=self.dummy16, dummy32=self.dummy32)
            del dy2, h1
            # attention backward
            dout = torch.mm(dh.to(f16), self.W_o)
            dq, dk, dv, gin = self._attention_backward(qkv, out, actx, dout, batch)
            del out, actx, dout
            qkv_epilogue_backward(dq, dk, dv, batch.pos, self.cos, self.sin, qkv,
                                  self.NQ, self.NKV, self.HD, gin)
            del dq, dk, dv
            dy1 = torch.mm(qkv, self.W_qkv)
            gq = qkv[:, :self.NQ * self.HD]
            gv = qkv[:, (self.NQ + self.NKV) * self.HD:]
            # LoRA gradients (fp16 GEMMs like autocast's, accumulated into fp32 grads)
            d_bq = torch.mm(gq.t(), aq).float().mul_(s)
            daq = torch.mm(gq, bq16).mul_(s)
            d_aq = torch.mm(daq.t(), xq).float()
            lqx = torch.mm(daq, aq16)
            d_bv = torch.mm(gv.t(), av).float().mul_(s)
            dav = torch.mm(gv, bv16).mul_(s)
            d_av = torch.mm(dav.t(), xv).float()
            lvx = torch.mm(dav, av16)
            del qkv, gq, gv, aq, av, daq, dav
            rms_backward(dy1, self.one, x, self.w_in[i], rstd1, dh, dres=dh, lq=lqx, lv=lvx,
                         p_drop=self.p_drop, keep_scale=self.keep_scale, seed_q=seeds[2 * i],
                         seed_v=seeds[2 * i + 1], dummy16=self.dummy16, dummy32=self.dummy32)
            del dy1, lqx, lvx, x
            store.drop(i)
            grads[4 * i:4 * i + 4] = [d_aq, d_bq, d_av, d_bv]
        store.close()
        return grads


class _Batch:
    pass


class _GpuStore:
    def __init__(self):
        self.t = {}

    def put(self, i, x):
        self.t[i] = x

    def prefetch(self, i):
        pass

    def get(self, i):
        return self.t[i]

    def drop(self, i):
        self.t.pop(i, None)

    def close(self):
        self.t.clear()


class _HostStore:
    """Residual checkpoints in pinned host memory: D2H on a side stream during forward, H2D one
    layer ahead during backward. Stream order: the side stream waits for the producer on the main
    stream; the main stream waits for the copy before it reads a prefetched tensor; record_stream
    keeps the allocator from reusing a source buffer while a copy still reads it."""

    def __init__(self, model, n):
        self.m, self.n = model, n
        rows = max(n, 2048)
        if model.host_ckpt is None or model.host_ckpt.shape[1] < rows:
            model.host_ckpt = torch.empty((model.L, rows, model.D), dtype=torch.float32, pin_memory=True)
        self.host = model.host_ckpt
        self.side = model.copy_stream
        self.out_ev, self.pending = {}, {}

    def put(self, i, x):
        main = torch.cuda.current_stream(self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            self.host[i, :self.n].copy_(x, non_blocking=True)
            x.record_stream(self.side)
            done = torch.cuda.Event()
            done.record(self.side)
        self.out_ev[i] = done

    def prefetch(self, i):
        if i in self.pending:
            return
        main = torch.cuda.current_stream(self.m.device)
        g = torch.empty((self.n, self.m.D), dtype=torch.float32, device=self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)                      # g's allocation is ordered on main
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            self.side.wait_event(self.out_ev[i])
            g.copy_(self.host[i, :self.n], non_blocking=True)
            g.record_stream(self.side)
            done = torch.cuda.Event()
            done.record(self.side)
        self.pending[i] = (g, done)

    def get(self, i):
        if i not in self.pending:
            self.prefetch(i)
        g, done = self.pending.pop(i)
        torch.cuda.current_stream(self.m.device).wait_event(done)
        return g

    def drop(self, i):
        pass

    def close(self):
        # the next step's D2H into the same host rows is ordered after this point on main
        self.pending.clear()


class FusedCausalLM(torch.autograd.Function):
    """Whole-model autograd node: inputs are the LoRA parameters, output the summed CE."""

    @staticmethod
    def forward(ctx, runner, batch, *lora_params):
        with torch.autocast("cuda", enabled=False):
            loss_sum, stats, saved = runner.forward(batch)
        ctx.runner, ctx.saved = runner, saved
        ctx.mark_non_differentiable(stats)
        return loss_sum, stats

    @staticmethod
    def backward(ctx, g_loss, _g_stats):
        with torch.autocast("cuda", enabled=False):
            grads = ctx.runner.backward(ctx.saved, g_loss)
        ctx.saved = None
        return (None, None, *grads)


def run_step(runner, batch):
    """(loss_sum tensor with grad, stats [N, 4] in sorted-token order)."""
    return FusedCausalLM.apply(runner, batch, *runner.lora_params)


def batch_metrics(batch, stats_cpu):
    """Soup / TRL metrics of one packed batch from the per-token stats (host)."""
    st = stats_cpu.double()
    loss, ent, hit = st[:, 0], st[:, 1], st[:, 2]
    valid = batch.valid_sorted
    n_lab = int(valid.sum())
    entropy = float(ent.sum() / batch.n_real)
    accuracy = float(hit[valid].sum() / n_lab) if n_lab > 0 else 0.0
    rows = batch.row_sorted
    row_sum = torch.zeros(batch.B, dtype=torch.float64).index_add_(0, rows[valid], loss[valid])
    row_cnt = torch.zeros(batch.B, dtype=torch.int64).index_add_(0, rows[valid],
                                                                 torch.ones(n_lab, dtype=torch.int64))
    row_loss = [float(s / max(int(c), 1)) for s, c in zip(row_sum.tolist(), row_cnt.tolist())]
    return dict(entropy=entropy, accuracy=accuracy, n_lab=n_lab, row_loss=row_loss,
                row_tokens=[int(c) for c in row_cnt.tolist()])


def warmup_specs(runner_cfg=None):
    """Every (kernel, args, kwargs) the fast path launches, for ahead-of-time compilation."""
    import torch as _torch

    from . import k_attn, k_head, k_mlp, k_nf4, k_norm, k_rope
    hidden, nq, nkv, hd, inter = 1536, 12, 2, 128, 8960
    if runner_cfg is not None:
        hidden, nq, nkv, hd, inter = runner_cfg
    triton_attn = config.ATTN_IMPL == "triton"
    specs = (k_nf4.warmup_specs() + k_norm.warmup_specs(hidden)
             + k_rope.warmup_specs(nq, nkv, hd, nq // nkv if triton_attn else 1,
                                   _torch.float32 if triton_attn else _torch.float16)
             + k_mlp.warmup_specs(inter) + k_head.warmup_specs())
    if triton_attn:
        specs += k_attn.warmup_specs(nq, nkv, hd)
    return specs


_ = math  # keep import for readers computing scales by hand
