"""Our own forward and backward of the 4-bit PEFT Qwen2 model for Soup's SFT step.

What the step computes is unchanged: the same NF4 weights (dequantised bit-exactly), the same
LoRA (q_proj, v_proj, r=16, alpha=32, dropout 0.05) on the same fp32 parameters, the same fp32
residual stream and fp16 GEMMs as the base run, the same summed cross-entropy over the assistant
tokens. How it is computed changes:

* padding-free: the real tokens of the batch are packed into one [N, 1536] sequence (N is about
  half of batch x padded length), with per-row positions; attention runs per packed row through
  cu_seqlens (PyTorch's memory-efficient kernel, see k_attn.py); padding is never computed;
* one autograd node for the whole model: the forward keeps only each layer's fp32 residual input
  (6 KB per token per layer, optionally in pinned host memory) and the head's d(hidden); the
  backward re-runs each layer's forward from its checkpoint and back-propagates through it by
  hand, so at most one layer's activations are alive at a time;
* fused kernels for everything that is not a GEMM (see k_*.py); GEMMs are cuBLAS on fp16
  weights that our NF4 kernel dequantises once per layer per pass into one workspace, with Q|K|V
  and gate|up fused into single GEMMs; the next layer's attention / MLP half is dequantised on a
  side stream while the current layer's GEMMs run, and the layer at each forward/backward turn
  is reused instead of dequantised again;
* the LM head never materialises the full logits (k_head.py) and computes its gradient in the
  forward, scaled by the incoming grad in the backward;
* the tied embedding is stored in fp16 (bit-exact, see integration.py).
"""

import math
import threading
import time

import torch

from . import config
from .k_attn import (_VARLEN_ERR, BWD_BLOCK, FWD_BLOCK_M, AttnMeta, attention_backward,
                     attention_forward, block_table, sdpa_attention, sdpa_attention_backward,
                     varlen_attention, varlen_attention_backward, varlen_context, varlen_selftest)
from .k_head import fused_head
from .k_mlp import gated_silu_backward, gated_silu_forward
from .k_nf4 import (NF4_CODE, RANGE_ALIGN, WeightStreamer, dequant_layer, dequant_range,
                    dequant_tensors, layer_numel, repack_layer, section_layout)
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

        # Dequantisation schedule. The workspace has an attention part (q|k|v|o, the first
        # split_q quantisation blocks) and an MLP part (gate|up|down). We track which layer each
        # part holds: the layer the forward ends with is the one the backward starts with, and the
        # layer the backward ends with is the next forward's first, so those are never
        # dequantised twice. With DEQUANT_OVERLAP, the next layer's part is dequantised on a side
        # stream as soon as the main stream has issued the last GEMM reading that part (events
        # order the two streams), so the memory-bound dequant runs under the compute-bound GEMMs.
        self.split_q = seg["gate_proj"][0] // 64
        _require(self.split_q % RANGE_ALIGN == 0 and seg["o_proj"][0] < seg["gate_proj"][0], "layer split")
        self.overlap = bool(config.DEQUANT_OVERLAP) and self.streamer is None
        self.ws_have = {"attn": None, "mlp": None}
        self.ev_free = {"attn": torch.cuda.Event(), "mlp": torch.cuda.Event()}
        self.ev_ready = {"attn": torch.cuda.Event(), "mlp": torch.cuda.Event()}
        self.dq_stream = torch.cuda.Stream(device=dev) if self.overlap else None
        self.n_dequant = 0

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
        self.steps = 0

        # attention implementation: the varlen path is checked against the padded SDPA path on
        # a small random batch first (forward and backward); any difference or error -> padded.
        self.attn_impl = config.ATTN_IMPL
        self.attn_note = self.attn_impl
        if self.attn_impl == "varlen":
            ok, err = varlen_selftest(dev, self.NQ, self.NKV, self.HD)
            if ok:
                self.attn_note = f"varlen (self-test max rel. err {err:.1e})"
            else:
                self.attn_impl = "sdpa"
                why = _VARLEN_ERR[-1] if _VARLEN_ERR else f"max rel. err {err}"
                self.attn_note = f"padded sdpa (varlen self-test failed: {why})"

        # Activations kept from the forward instead of recomputed in the backward (SAVE_GU,
        # SAVE_ATTN) and the residual checkpoints (OFFLOAD_CHECKPOINTS) live in pinned host
        # arenas, one row per layer: the forward copies them out on the copy stream while it
        # computes the next layer, the backward copies them back one layer ahead. The arenas are
        # allocated in a background thread now, while Soup builds the trainer.
        self.copy_stream = torch.cuda.Stream(device=dev)
        self.host, self.host_cap, self.host_note = {}, 0, "none"
        self._host_thread = None
        self.start_host_alloc(int(config.HOST_CAPACITY))

    @torch.no_grad()
    def restore_stock(self):
        """Undo what the fast path changed, for the stock forward: 4-bit weights back on the GPU (if
        they were moved to host memory) and the fp32 embedding the stock path expects."""
        self._host_ready()
        self.host, self.host_cap = {}, 0
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
        """Streaming mode only: fp16 weights of layer i into the workspace, prefetch layer `nxt`."""
        if self.streamer is None:
            return
        pk, qa, s2, o2 = self.streamer.acquire(i)
        dequant_tensors(pk, qa, s2, o2, self.packed[i].n_qblocks, self.code2, self.nf4, self.ws)
        self.n_dequant += 1
        self.streamer.release()
        self.streamer.prefetch(nxt)
        self.ws_have = {"attn": None, "mlp": None}

    def _ws_load(self, part, i):
        """Start dequantising `part` ("attn" or "mlp") of layer i into the workspace, unless the
        workspace already holds (or is already receiving) it."""
        if i is None or self.ws_have[part] == i:
            return
        q0, q1 = (0, self.split_q) if part == "attn" else (self.split_q, self.packed[i].n_qblocks)
        if self.overlap:
            side = self.dq_stream
            side.wait_event(self.ev_free[part])         # the main stream's last reader of the part
            with torch.cuda.stream(side):
                dequant_range(self.packed[i], q0, q1, self.code2, self.nf4, self.ws)
            self.ev_ready[part].record(side)
        else:
            dequant_range(self.packed[i], q0, q1, self.code2, self.nf4, self.ws)
        self.ws_have[part] = i
        self.n_dequant += 1

    def _ws_use(self, part, i):
        """The main stream is about to read `part` of layer i from the workspace."""
        if self.streamer is not None:
            return
        self._ws_load(part, i)
        if self.overlap:
            torch.cuda.current_stream(self.device).wait_event(self.ev_ready[part])

    def _ws_release(self, part, nxt):
        """The main stream has issued its last read of `part`; prefetch layer `nxt`'s part."""
        if self.streamer is not None:
            return
        if self.overlap:
            self.ev_free[part].record(torch.cuda.current_stream(self.device))
        self._ws_load(part, nxt)

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
        batch.cu = views[5]
        if self.attn_impl == "sdpa":
            bt = torch.stack([row, pos]).to(self.device)   # packed token -> (row, position)
            batch.b_idx, batch.t_idx = bt[0], bt[1]
        batch.row_sorted = row[perm]
        batch.valid_sorted = valid[perm]
        batch.n_real = n
        return batch

    # ------------------------------------------------------------------ activation storage
    HOST_KINDS = ("x", "gu", "qkv", "out", "op")

    def start_host_alloc(self, cap, kinds=None):
        """Allocate the pinned host arenas (rows for layers 0..L-2; the last layer's tensors are
        used right after the forward and stay on the GPU) in a background thread."""
        self._host_ready()
        if kinds is None:
            kinds = set()
            if config.OFFLOAD_CHECKPOINTS:
                kinds.add("x")
            if config.SAVE_GU:
                kinds.add("gu")
            if config.SAVE_ATTN:
                kinds |= {"qkv", "out", "op"}
        if self.attn_impl != "varlen":
            kinds -= {"qkv", "out", "op"}
        width = {"x": (self.D, torch.float32), "gu": (2 * self.I, torch.float16),
                 "qkv": ((self.NQ + 2 * self.NKV) * self.HD, torch.float16),
                 "out": (self.NQ * self.HD, torch.float16), "op": (self.D, torch.float16)}
        kinds = sorted(k for k in kinds if k not in self.host)
        if not kinds or cap <= 0:
            return

        def work():
            started = time.perf_counter()
            try:
                made = {k: torch.empty((self.L - 1, cap, width[k][0]), dtype=width[k][1], pin_memory=True)
                        for k in kinds}
                self.host = dict(self.host, **made)
                self.host_cap = cap if not self.host_cap else min(self.host_cap, cap)
                gib = sum(t.numel() * t.element_size() for t in self.host.values()) / 2**30
                self.host_note = (f"{gib:.2f} GiB pinned for {sorted(self.host)} "
                                  f"({cap} tokens, {time.perf_counter() - started:.2f} s)")
            except Exception as exc:  # noqa: BLE001 - the backward recomputes instead
                self.host_note = f"host arenas unavailable ({type(exc).__name__}: {exc})"

        self._host_thread = threading.Thread(target=work, daemon=True, name="soup-pinned")
        self._host_thread.start()

    def _host_ready(self):
        if self._host_thread is not None:
            self._host_thread.join()
            self._host_thread = None

    def _save_plan(self, n):
        """Which activations go through the host arenas this step (a batch larger than the
        arenas, or a missing arena, falls back to recomputation, which gives the same result)."""
        self._host_ready()
        fits = n <= self.host_cap
        plan = _Plan()
        plan.x = fits and config.OFFLOAD_CHECKPOINTS and "x" in self.host
        plan.gu = fits and config.SAVE_GU and "gu" in self.host
        plan.attn = (fits and config.SAVE_ATTN and self.attn_impl == "varlen"
                     and all(k in self.host for k in ("qkv", "out", "op")))
        plan.arenas = self.host
        plan.last = self.L - 1
        return plan

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
        ATTN_IMPL == "sdpa": PyTorch's memory-efficient SDPA (what base Soup runs) on zero-padded
        [B, H, Tmax, 128] buffers written straight from the packed tokens (one index_put each; K/V
        expanded to the 12 query heads like repeat_kv); right padding needs only the causal mask.
        In the backward recompute (grad=True) the SDPA call records a local autograd graph that
        `_attention_backward` differentiates."""
        if self.attn_impl == "varlen":
            return varlen_attention(qkv, batch.cu, batch.Tm, self.NQ, self.NKV, self.HD, need_lse=grad)
        if self.attn_impl == "triton":
            return attention_forward(qkv, batch.meta, self.NQ, self.NKV, self.HD)
        return sdpa_attention(qkv, batch.b_idx, batch.t_idx, batch.B, batch.Tm, self.NQ, self.NKV,
                              self.HD, grad)

    def _attention_backward(self, qkv, out, actx, dout, batch):
        """(dq, dk, dv, GIN) for the QKV epilogue backward."""
        if self.attn_impl == "varlen":
            dq, dkp, dvp = varlen_attention_backward(actx, dout, batch.cu, batch.Tm, self.NQ,
                                                     self.NKV, self.HD)
            return dq, dkp, dvp, self.NQ // self.NKV
        if self.attn_impl == "triton":
            dq, dkp, dvp = attention_backward(qkv, out, actx, dout, batch.meta, self.NQ, self.NKV, self.HD)
            return dq, dkp, dvp, self.NQ // self.NKV
        dq, dk, dv = sdpa_attention_backward(actx, dout, batch.b_idx, batch.t_idx, batch.B, batch.Tm,
                                             self.NQ, self.NKV, self.HD)
        return dq, dk, dv, 1

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        n, D = batch.n, self.D
        dev = self.device
        f16, f32 = torch.float16, torch.float32
        self._refresh_lora16()
        if self.overlap:
            self.dq_stream.wait_stream(torch.cuda.current_stream(dev))
        seeds = self._seeds()
        plan = self._save_plan(n)
        self.last_plan = plan
        store = _ActStore(self, n)
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
            store.retire(i - store.IN_FLIGHT)
            store.put(i, "x", x, plan.x)
            nxt = i + 1 if i + 1 < self.L else None
            self._dequant(i, nxt)
            self._ws_use("attn", i)
            qkv, _aq, _av, out, actx = self._attn_part(i, y, xq, xv, batch, grad=plan.attn)
            op = torch.mm(out, self.W_o.t())
            self._ws_release("attn", nxt)
            if plan.attn:
                # what the attention backward and the norm-2 backward need, so that the backward
                # does not re-run norm-1 -> QKV GEMM -> epilogue -> attention -> O GEMM
                store.put(i, "qkv", qkv, True)
                store.put(i, "out", out, True)
                store.put(i, "op", op, True)
                store.put(i, "actx", actx[4:], False)       # logsumexp, philox seed/offset
            del qkv, _aq, _av, out, actx
            h1 = torch.empty((n, D), dtype=f32, device=dev)
            rms_forward(x, self.w_post[i], y2, rstd, self.eps, delta=op, xout=h1,
                        dummy16=self.dummy16, dummy32=self.dummy32)
            del op
            self._ws_use("mlp", i)
            gu = torch.mm(y2, self.W_gu.t())
            act = torch.empty((n, self.I), dtype=f16, device=dev)
            gated_silu_forward(gu, act)
            if plan.gu:
                store.put(i, "gu", gu, True)        # the backward reads it instead of a gate|up GEMM
            del gu
            d16 = torch.mm(act, self.W_d.t())
            self._ws_release("mlp", nxt)
            del act
            x = h1
        del y, xq, xv, y2
        store.retire(self.L)
        hf = torch.empty((n, D), dtype=f32, device=dev)
        yf = torch.empty((n, D), dtype=f16, device=dev)
        rstd_f = torch.empty(n, dtype=f32, device=dev)
        rms_forward(x, self.w_final, yf, rstd_f, self.eps, delta=d16, xout=hf,
                    dummy16=self.dummy16, dummy32=self.dummy32)
        del x, d16
        loss_sum, stats, dy_head = self._head(yf, batch)
        del yf
        self.steps += 1
        saved = dict(store=store, plan=plan, hf=hf, rstd_f=rstd_f, dy_head=dy_head, seeds=seeds,
                     batch=batch)
        return loss_sum, stats, saved

    def _head(self, yf, batch):
        return fused_head(yf, self.emb, batch.tgt_sorted, batch.perm, batch.inv, batch.n_lab,
                          config.HEAD_CHUNK, config.HEAD_GRAD_SCALE)

    # ------------------------------------------------------------------ backward
    def backward(self, saved, g_loss):
        batch, store, seeds, plan = saved["batch"], saved["store"], saved["seeds"], saved["plan"]
        n, D = batch.n, self.D
        dev = self.device
        f16, f32 = torch.float16, torch.float32
        gscale = (g_loss.detach().reshape(1).to(f32) / float(config.HEAD_GRAD_SCALE)).contiguous()
        dh = torch.empty((n, D), dtype=f32, device=dev)
        dh16 = torch.empty((n, D), dtype=f16, device=dev)     # fp16 copy of dh for the GEMMs
        rms_backward(saved.pop("dy_head"), gscale, saved.pop("hf"), self.w_final, saved.pop("rstd_f"),
                     dh, out16=dh16, dummy16=self.dummy16, dummy32=self.dummy32)
        y = torch.empty((n, D), dtype=f16, device=dev)
        xq = torch.empty((n, D), dtype=f16, device=dev)
        xv = torch.empty((n, D), dtype=f16, device=dev)
        y2 = torch.empty((n, D), dtype=f16, device=dev)
        rstd1 = torch.empty(n, dtype=f32, device=dev)
        rstd2 = torch.empty(n, dtype=f32, device=dev)
        grads = [None] * len(self.lora_params)
        s = self.lora_scale
        for i in reversed(range(self.L)):
            x = store.get(i, "x")
            if i > 0:
                store.prefetch(i - 1, "x")
                store.prefetch(i - 1, "op")
            aq16, bq16, av16, bv16 = self.lora16[4 * i:4 * i + 4]
            prv = i - 1 if i > 0 else None
            self._dequant(i, prv)
            # norm 1 again: the dropout-masked LoRA inputs (same Philox seeds) and rstd
            rms_forward(x, self.w_in[i], y, rstd1, self.eps, xq=xq, xv=xv, p_drop=self.p_drop,
                        keep_scale=self.keep_scale, seed_q=seeds[2 * i], seed_v=seeds[2 * i + 1],
                        dummy16=self.dummy16, dummy32=self.dummy32)
            if plan.attn:
                aq = torch.mm(xq, aq16.t())
                av = torch.mm(xv, av16.t())
                op = store.get(i, "op")
            else:
                self._ws_use("attn", i)
                qkv, aq, av, out, actx = self._attn_part(i, y, xq, xv, batch, grad=True)
                op = torch.mm(out, self.W_o.t())
            h1 = torch.empty((n, D), dtype=f32, device=dev)
            rms_forward(x, self.w_post[i], y2, rstd2, self.eps, delta=op, xout=h1,
                        dummy16=self.dummy16, dummy32=self.dummy32)
            del op
            self._ws_use("mlp", i)
            gu = store.get(i, "gu") if plan.gu else torch.mm(y2, self.W_gu.t())
            # MLP backward
            dact = torch.mm(dh16, self.W_d)
            gated_silu_backward(gu, dact)
            del dact
            dy2 = torch.mm(gu, self.W_gu)
            self._ws_release("mlp", prv)        # layer i-1's MLP weights dequantise under the attention backward
            del gu
            if plan.gu and i > 0:
                store.prefetch(i - 1, "gu")     # into the block gu just released
            rms_backward(dy2, self.one, h1, self.w_post[i], rstd2, dh, dres=dh, out16=dh16,
                         dummy16=self.dummy16, dummy32=self.dummy32)
            del dy2, h1
            # attention backward
            if plan.attn:
                self._ws_use("attn", i)
            dout = torch.mm(dh16, self.W_o)
            if plan.attn:
                qkv = store.get(i, "qkv")
                out = store.get(i, "out")
                actx = varlen_context(qkv, out, store.get(i, "actx"), self.NQ, self.NKV, self.HD)
            dq, dk, dv, gin = self._attention_backward(qkv, out, actx, dout, batch)
            del out, actx, dout
            qkv_epilogue_backward(dq, dk, dv, batch.pos, self.cos, self.sin, qkv,
                                  self.NQ, self.NKV, self.HD, gin)
            del dq, dk, dv
            if plan.attn and i > 0:
                store.prefetch(i - 1, "qkv")
                store.prefetch(i - 1, "out")
            # layer 0: nothing below needs d(input), only its LoRA gradients
            dy1 = torch.mm(qkv, self.W_qkv) if i > 0 else None
            self._ws_release("attn", prv)
            gq = qkv[:, :self.NQ * self.HD]
            gv = qkv[:, (self.NQ + self.NKV) * self.HD:]
            # LoRA gradients (fp16 GEMMs like autocast's, accumulated into fp32 grads)
            d_bq = torch.mm(gq.t(), aq).float().mul_(s)
            daq = torch.mm(gq, bq16).mul_(s)
            d_aq = torch.mm(daq.t(), xq).float()
            d_bv = torch.mm(gv.t(), av).float().mul_(s)
            dav = torch.mm(gv, bv16).mul_(s)
            d_av = torch.mm(dav.t(), xv).float()
            if i > 0:
                lqx = torch.mm(daq, aq16)
                lvx = torch.mm(dav, av16)
                rms_backward(dy1, self.one, x, self.w_in[i], rstd1, dh, dres=dh, lq=lqx, lv=lvx,
                             p_drop=self.p_drop, keep_scale=self.keep_scale, seed_q=seeds[2 * i],
                             seed_v=seeds[2 * i + 1], out16=dh16, dummy16=self.dummy16,
                             dummy32=self.dummy32)
                del lqx, lvx
            del qkv, gq, gv, aq, av, daq, dav, dy1, x
            grads[4 * i:4 * i + 4] = [d_aq, d_bq, d_av, d_bv]
        store.close()
        return grads


class _Batch:
    pass


class _Plan:
    """Per-step choice of what the forward keeps for the backward (see FastQwen2._save_plan)."""
    x = gu = attn = False
    arenas = None
    last = 0


class _ActStore:
    """Per-layer tensors the backward needs from the forward.

    put(i, kind, t, host): keep t on the GPU, or (host=True, an arena exists for `kind`, and i is
    not the last layer) copy it into row i of the pinned host arena on the copy stream, which
    first waits for the main stream (so t is complete).
    retire(i): called at the start of forward layer i; the main stream waits for the copies of
    layer i - IN_FLIGHT and only then are their source tensors released. The order matters: the
    caching allocator hands a freed block to the next allocation on the main stream right away,
    so the GPU-side wait must be enqueued before the release. (record_stream would instead keep a
    block pending until its copy is done *when the CPU allocates*, and the CPU runs many layers
    ahead of the GPU: every layer would get a fresh block.) A slow copy therefore stalls the main
    stream two layers later instead of growing memory.
    prefetch(i, kind): allocate the GPU tensor on the main stream (usually the block that was
    just released) and start the host-to-device copy on the copy stream once the main stream has
    reached that point, i.e. after every queued kernel that read the old contents of the block.
    get(i, kind): the GPU tensor; the main stream waits for its copy. The copy stream is one FIFO
    for both directions, so a host row is never read before it is written, and the next step's
    forward never overwrites a row before this step's backward has read it.
    """

    IN_FLIGHT = 2

    def __init__(self, model, n):
        self.m, self.n = model, n
        self.side = model.copy_stream
        self.gpu, self.pending, self.where, self.inflight = {}, {}, {}, {}

    def put(self, i, kind, t, host):
        arena = self.m.host.get(kind) if host and i < self.m.L - 1 else None
        if arena is None:
            self.gpu[(i, kind)] = t
            return
        main = torch.cuda.current_stream(self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            arena[i, :self.n].copy_(t, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.side)
        self.inflight.setdefault(i, []).append((t, done))
        self.where[(i, kind)] = (arena, t.dtype, tuple(t.shape))

    def retire(self, upto):
        """Main stream waits for the copies of layers <= upto, then their sources are released."""
        main = torch.cuda.current_stream(self.m.device)
        for j in sorted(k for k in self.inflight if k <= upto):
            for _t, done in self.inflight[j]:
                main.wait_event(done)
            del self.inflight[j]

    def prefetch(self, i, kind):
        key = (i, kind)
        if key in self.gpu or key in self.pending or key not in self.where:
            return
        arena, dtype, shape = self.where.pop(key)
        main = torch.cuda.current_stream(self.m.device)
        g = torch.empty(shape, dtype=dtype, device=self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            g.copy_(arena[i, :self.n], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.side)
        self.pending[key] = (g, done)

    def get(self, i, kind):
        key = (i, kind)
        if key in self.gpu:
            return self.gpu.pop(key)
        if key not in self.pending:
            self.prefetch(i, kind)
        g, done = self.pending.pop(key)
        torch.cuda.current_stream(self.m.device).wait_event(done)
        return g

    def close(self):
        self.retire(10 ** 9)
        self.gpu.clear()
        self.pending.clear()
        self.where.clear()


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
    impl = config.ATTN_IMPL
    f16, f32 = _torch.float16, _torch.float32
    if impl == "triton":
        epi = k_rope.warmup_specs(nq, nkv, hd, nq // nkv, f32)
    elif impl == "varlen":
        # the varlen path (GIN = 6 partials) and its padded-SDPA fallback (GIN = 1)
        epi = k_rope.warmup_specs(nq, nkv, hd, nq // nkv, f16) + k_rope.warmup_specs(nq, nkv, hd, 1, f16)[1:]
    else:
        epi = k_rope.warmup_specs(nq, nkv, hd, 1, f16)
    specs = (k_nf4.warmup_specs() + k_norm.warmup_specs(hidden) + epi
             + k_mlp.warmup_specs(inter) + k_head.warmup_specs())
    if impl == "triton":
        specs += k_attn.warmup_specs(nq, nkv, hd)
    return specs


_ = math  # keep import for readers computing scales by hand
