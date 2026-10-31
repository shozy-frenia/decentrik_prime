"""Our own forward and backward of the 4-bit PEFT Qwen2 model for Soup's SFT step.

What the step computes is unchanged: the same NF4 weights (dequantised bit-exactly), the same
LoRA (q_proj, v_proj, r=16, alpha=32, dropout 0.05) on the same fp32 parameters, the same fp32
residual stream and fp16 GEMMs as the base run, the same summed cross-entropy over the assistant
tokens. How it is computed changes:

* padding-free: the real tokens of the batch are packed into one [N, 1536] sequence (N is about
  half of batch x padded length), with per-row positions; attention runs per packed row through
  cu_seqlens (PyTorch's memory-efficient kernel, see k_attn.py); padding is never computed;
* one autograd node for the whole model: the forward keeps each layer's fp32 residual input and
  the expensive activations (gate|up, post-RoPE QKV, attention output, O output) in pinned host
  memory, copied out by the copy engine while the next layer computes and copied back one layer
  ahead in the backward, which back-propagates through each layer by hand;
* fused kernels for everything that is not a GEMM (see k_*.py); GEMMs are cuBLAS on fp16
  weights that our NF4 kernel dequantises once per layer per pass into one workspace, with Q|K|V
  and gate|up fused into single GEMMs; the next layer's attention / MLP half is dequantised on a
  side stream while the current layer's GEMMs run, and the layer at each forward/backward turn
  is reused instead of dequantised again;
* v6 placement: the repacked 4-bit layers live in pinned host memory and are streamed into two
  GPU slots two layers ahead of their dequantisation (k_nf4.WeightPipe); the fp16 tied embedding
  lives in pinned host memory too: the input rows are gathered on the CPU and the LM head streams
  the matrix through a small ring of GPU buffers (k_head.streamed_head). Between uses the GPU
  holds no model weights at all;
* the LM head never materialises the full logits (k_head.py) and computes its gradient in the
  forward, scaled by the incoming grad in the backward;
* the tied embedding is stored in fp16 (bit-exact, see integration.py).
"""

import contextlib
import math
import threading
import time

import torch

from . import config, hostmem
from .k_attn import (_VARLEN_ERR, BWD_BLOCK, FWD_BLOCK_M, AttnMeta, attention_backward,
                     attention_forward, block_table, sdpa_attention, sdpa_attention_backward,
                     varlen_attention, varlen_attention_backward, varlen_context, varlen_selftest)
from .k_head import EmbeddingStream, ResidentEmbedding, fused_head, streamed_head
from .k_mlp import gated_silu_backward, gated_silu_forward
from .k_nf4 import (NF4_CODE, RANGE_ALIGN, WeightPipe, dequant_range, layer_numel, repack_layer,
                    section_layout)
from .k_norm import rms_backward, rms_forward
from .k_rope import qkv_epilogue_backward, qkv_epilogue_forward

ALIGN = 16  # int32 elements: keeps every packed metadata slice 64-byte aligned for Triton

# Shapes of Qwen2.5-1.5B (the only model the fast path accepts), used to size the pinned buffers
# that apply() allocates before the model exists. FastQwen2 checks the real model and allocates
# afresh if anything differs.
QWEN15 = dict(L=28, D=1536, I=8960, NQ=12, NKV=2, HD=128, V=151936)


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
    _require(a.weight.device.type in ("cuda", "cpu") and b.weight.device == a.weight.device,
             "LoRA weights on an unexpected device")
    return a.weight, b.weight, p, float(module.scaling[adapter])


def _is_bnb_4bit(module):
    return type(module).__name__ in ("Linear4bit", "LinearNF4") and hasattr(module.weight, "quant_state")


@torch.no_grad()
def restore_to_gpu(model, device):
    """Put everything a stock forward reads back on the GPU, wherever our load hooks or FastQwen2
    left it: 4-bit weights and their quantisation state, every other parameter and buffer, and the
    embedding as fp32 (what prepare_model_for_kbit_training gives the stock path)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for mod in base.modules():
        if _is_bnb_4bit(mod):
            w = mod.weight
            if w.device != device:
                w.data = w.data.to(device)
            qs = getattr(w, "quant_state", None)
            if qs is not None:
                qs.to(device)
    for p in model.parameters():
        if type(p).__name__ != "Params4bit" and p.device != device:
            p.data = p.data.to(device)
    for b in model.buffers():
        if b.device != device:
            b.data = b.data.to(device)
    emb = base.get_input_embeddings().weight
    if emb.dtype == torch.float16:
        emb.data = emb.data.float()
    torch.cuda.empty_cache()


class FastQwen2:
    """Holds the repacked model and runs the fused forward / backward."""

    @torch.no_grad()
    def __init__(self, peft_model, seed=0, background=False):
        """Validate the model (nothing is modified until every check passed), move the LoRA
        parameters to the GPU, then place and repack the model (_build). With background=True the
        build runs in a thread (it copies ~1.1 GB into pinned host memory while Soup tokenises the
        data and builds the trainer); ready() joins it and re-raises its error, if any."""
        self.peft_model = peft_model
        self._builder, self._build_error = None, None
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
        # the norms and biases stay on the GPU in every placement; the 4-bit weights and the
        # embedding may already be in host memory (integration.py's load hooks)
        dev = qwen.norm.weight.device
        _require(dev.type == "cuda", "model must be on CUDA")
        self.device = dev
        _require(emb.weight.shape == (self.V, self.D), "embedding shape")
        _require(emb.weight.device.type in ("cuda", "cpu"), "embedding device")

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
            p_seen.update((pq, pv))
            scale_seen.update((sq, sv))
            linears = {"q_proj": q.base_layer, "k_proj": k, "v_proj": v.base_layer, "o_proj": at.o_proj,
                       "gate_proj": mlp.gate_proj, "up_proj": mlp.up_proj, "down_proj": mlp.down_proj}
            for name, mod in linears.items():
                _require(_is_bnb_4bit(mod), f"layer {i} {name} is not a bitsandbytes 4-bit linear")
                _require(not hasattr(mod, "lora_A"), f"layer {i} {name} has an unexpected adapter")
                _require(mod.weight.data.device.type in ("cuda", "cpu"),
                         f"layer {i} {name} weight on an unexpected device")
                code = mod.weight.quant_state.state2.code if getattr(mod.weight.quant_state, "nested", False) else None
                _require(code is not None, f"layer {i} {name}: no double quantisation")
                if code_ref is None:
                    code_ref = code
                _require(torch.equal(code_ref.cpu(), code.cpu()), "second-level code tables differ")
            try:
                n_layer = layer_numel(linears)
            except ValueError as exc:
                raise Unsupported(f"layer {i}: {exc}") from exc
            _require(n_layer % (64 * 256) == 0, "layer size")
            for name in ("o_proj", "gate_proj", "up_proj", "down_proj"):
                _require(linears[name].bias is None, f"{name} bias")
            biases = [linears[n].bias for n in ("q_proj", "k_proj", "v_proj")]
            _require(all(b is not None for b in biases), "q/k/v biases expected")
            bias16 = torch.cat([b.data.to(dev, torch.float16) for b in biases])
            _require(torch.equal(bias16.float(), torch.cat([b.data.to(dev).float() for b in biases])),
                     "q/k/v bias not fp16-representable")
            _require(abs(layer.input_layernorm.variance_epsilon - self.eps) < 1e-12, "norm eps")
            self._linears.append(linears)
            self.bias_qkv.append(bias16)
            self.w_in.append(layer.input_layernorm.weight.data.to(dev).float().contiguous())
            self.w_post.append(layer.post_attention_layernorm.weight.data.to(dev).float().contiguous())
            self.lora_params += [aq, bq, av, bv]
        _require(len(p_seen) == 1 and len(scale_seen) == 1, "one dropout p and LoRA scaling expected")
        self.p_drop = p_seen.pop()
        self.lora_scale = scale_seen.pop()
        w = emb.weight
        if w.dtype != torch.float16:
            _require(torch.equal(w.data.to(torch.float16).float(), w.data.float()),
                     "embedding not fp16-representable")

        # LoRA parameters on the GPU (they follow the base layer's device in PEFT, i.e. the host
        # when the load hooks kept the 4-bit weights there); no optimizer exists yet, and TRL casts
        # them to bf16 when it builds the trainer, so this must happen now, not in the background
        for p in self.lora_params:
            if p.device != dev:
                p.data = p.data.to(dev)
        self._qwen, self._seed = qwen, seed
        self._overrides = {}           # placement switches forced by use_resident_layout()
        self._rx_check = None          # self-test: max error of the rebuilt residual stream
        self._gu_check = None          # self-test: recomputed gate|up rows vs the forward's
        self.pipe = None
        self.emb_on_host = False
        self.test_note = "not run"
        self.host, self.host_cap, self.host_note = {}, 0, "none"
        self._host_thread = None
        if background:
            self._builder = threading.Thread(target=self._build_safe, daemon=True, name="soup-build")
            self._builder.start()
        else:
            self._build()

    def _build_safe(self):
        """The background build: placement, then (config.SELF_TEST) the start-up self-test, with
        the v5 placement as the fallback if the streamed one fails it. Errors are re-raised by
        ready() (the caller then falls back to the stock path)."""
        try:
            try:
                self._build()
                if config.SELF_TEST:
                    from . import precompile
                    precompile.wait(report=False)    # never JIT-compile next to the compile threads
                    self.test_note = self.self_test()
            except Exception as exc:  # noqa: BLE001
                streamed_cfg = (config.OFFLOAD_WEIGHTS or config.OFFLOAD_EMBEDDING
                                or config.OFFLOAD_CHECKPOINTS or config.HEAD_IMPL != "rows")
                if not (config.FALLBACK_RESIDENT and streamed_cfg and self._overrides == {}):
                    raise
                reason = f"{type(exc).__name__}: {exc}"
                self.test_note = f"streamed placement failed ({reason})"
                self._rebuild_resident(reason)
                if config.SELF_TEST:
                    self.test_note += "; " + self.self_test()
        except BaseException as exc:  # noqa: BLE001 - re-raised by ready()
            self._build_error = exc

    def ready(self):
        """Wait for the background build; raise its error (the caller falls back)."""
        thread = self._builder
        if thread is not None and thread is threading.current_thread():
            return                       # the build's own self-test
        self._builder = None
        if thread is not None:
            thread.join()
        if self._build_error is not None:
            exc, self._build_error = self._build_error, None
            raise exc

    def _opt(self, name):
        return self._overrides.get(name, getattr(config, name))

    def streaming(self):
        """True when some part of the model lives in host memory (the v6 placement)."""
        return self.pipe is not None or self.emb_on_host

    def use_resident_layout(self, reason):
        """Rebuild with the v5 placement (4-bit layers and fp16 embedding on the GPU, v5 LM head,
        checkpoints on the GPU)."""
        self.ready()
        self._rebuild_resident(reason)

    def _rebuild_resident(self, reason):
        """Works from whatever state a failed build or step left: every layer is repacked from
        where its weights are now, the embedding is moved to the GPU."""
        self._overrides = dict(OFFLOAD_WEIGHTS=False, OFFLOAD_EMBEDDING=False, HEAD_IMPL="rows",
                               OFFLOAD_CHECKPOINTS=False)
        self.pipe, self.emb_source, self.emb_on_host = None, None, False
        self._pending = None
        for p in self.lora_params:
            p.grad = None
        torch.cuda.synchronize(self.device)
        self._build()
        self.notes.append(f"v5 placement after: {reason}")

    @torch.no_grad()
    def _build(self):
        qwen, seed, dev = self._qwen, self._seed, self.device
        w = qwen.embed_tokens.weight
        # ---- phase 2: placement. Streams first (the weight pipe orders its copies after them).
        self.notes = []
        self.overlap = bool(config.DEQUANT_OVERLAP)
        self.dq_stream = torch.cuda.Stream(device=dev) if self.overlap else None
        self.copy_stream = torch.cuda.Stream(device=dev)
        # the fp16 embedding: pinned host memory (streamed) or the GPU
        offload_emb = bool(self._opt("OFFLOAD_EMBEDDING"))
        if offload_emb and w.device.type == "cpu" and w.is_pinned() and w.dtype == torch.float16:
            pass                                   # already there (a rebuild)
        elif offload_emb:
            try:
                host = hostmem.take("embedding", (self.V, self.D), torch.float16)
                host.copy_(w.data)
                w.data = host
            except Exception as exc:  # noqa: BLE001 - keep it on the GPU
                offload_emb = False
                self.notes.append(f"embedding on GPU ({type(exc).__name__}: {exc})")
        if not offload_emb and (w.device != dev or w.dtype != torch.float16):
            w.data = w.data.to(dev, torch.float16)
        self.emb = w.data
        self.emb_on_host = offload_emb
        # the repacked 4-bit layers: pinned host memory (streamed) or the GPU
        _, layer_bytes = section_layout(layer_numel(self._linears[0]) // 64)
        host_all, resident = None, set()
        if self._opt("OFFLOAD_WEIGHTS"):
            resident = set(range(min(self.L, max(0, int(config.RESIDENT_LAYERS)))))
            try:
                host_all = hostmem.take("weights", (self.L * layer_bytes,), torch.uint8)
            except Exception as exc:  # noqa: BLE001 - keep them on the GPU
                host_all = None
                self.notes.append(f"weights on GPU ({type(exc).__name__}: {exc})")
        self.packed = []
        self._host_refs = []
        code2 = None
        for i, linears in enumerate(self._linears):
            on_host = host_all is not None and i not in resident
            out = host_all[i * layer_bytes:(i + 1) * layer_bytes] if on_host else None
            packed, c2 = repack_layer(linears, dev, host=on_host, out=out)
            code2 = c2 if code2 is None else code2
            self.packed.append(packed)
            if on_host:
                for mod in linears.values():
                    wt = mod.weight
                    qs = wt.quant_state
                    s2 = qs.state2
                    self._host_refs.append((wt, wt.data, qs, qs.code, qs.absmax, qs.offset,
                                            s2.absmax, s2.code))
        self.pipe = (WeightPipe(self.packed, dev, side=self.dq_stream, resident=resident)
                     if host_all is not None else None)
        torch.cuda.empty_cache()     # whatever the load left in the cache goes back to the driver

        self.keep_scale = 1.0 / (1.0 - self.p_drop)
        self.code2 = code2
        self.nf4 = torch.tensor(NF4_CODE, dtype=torch.float32, device=dev)
        self.w_final = qwen.norm.weight.data.to(dev).float().contiguous()
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
        self.ws_have = {"attn": None, "mlp": None}
        self.ev_free = {"attn": torch.cuda.Event(), "mlp": torch.cuda.Event()}
        self.ev_ready = {"attn": torch.cuda.Event(), "mlp": torch.cuda.Event()}
        self.n_dequant = 0
        if self.pipe is not None:
            self.pipe.prefetch(0)
            self.pipe.prefetch(1)

        # the LM head's view of the embedding
        if self.emb_on_host:
            self.emb_source = EmbeddingStream(self.emb, config.HEAD_VOCAB_CHUNK, config.HEAD_RING, dev)
        else:
            self.emb_source = ResidentEmbedding(self.emb, config.HEAD_VOCAB_CHUNK)
        self.head_impl = "rows" if (self._opt("HEAD_IMPL") == "rows" and not self.emb_on_host) else "chunks"

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
        self._pending = None

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
        # computes the next layer, the backward copies them back one layer ahead. The arenas come
        # from the early pinned pool (hostmem) or are allocated in a background thread now.
        # (a rebuild keeps the arenas it already has; start_host_alloc joins a running allocation)
        self.start_host_alloc(int(config.HOST_CAPACITY))

    def placement(self):
        """Where the model lives, for the log."""
        self.ready()
        parts = []
        if self.pipe is not None:
            res = len(self.pipe.resident)
            parts.append("4-bit layers in pinned host memory, streamed"
                         + (f" ({res} resident on GPU)" if res else ""))
        else:
            parts.append("4-bit layers on GPU")
        parts.append("embedding in pinned host memory, streamed head" if self.emb_on_host
                     else f"embedding on GPU, {self.head_impl} head")
        return "; ".join(parts + self.notes)

    @torch.no_grad()
    def reclaim_gpu_copies(self):
        """The streamed placement keeps no model weight on the GPU. If something copied the
        module's weights there anyway (accelerate's prepare_model calls model.to(device) on a model
        without hf_device_map), point the modules back at the host buffers they were views of and
        free the GPU copies. Returns the MiB freed."""
        self.ready()
        freed = 0

        def back(obj, attr, ref):
            nonlocal freed
            cur = getattr(obj, attr, None)
            if cur is None or ref is None or cur.data_ptr() == ref.data_ptr():
                return
            if cur.is_cuda:
                freed += cur.numel() * cur.element_size()
            setattr(obj, attr, ref)

        for wt, data, qs, code, absmax, offset, abs2, code2 in getattr(self, "_host_refs", []):
            if wt.data.data_ptr() != data.data_ptr():
                if wt.data.is_cuda:
                    freed += wt.data.numel() * wt.data.element_size()
                wt.data = data
            back(qs, "code", code)
            back(qs, "absmax", absmax)
            back(qs, "offset", offset)
            back(qs.state2, "absmax", abs2)
            back(qs.state2, "code", code2)
        if self.emb_on_host:
            w = self._qwen.embed_tokens.weight
            if w.data.data_ptr() != self.emb.data_ptr():
                if w.data.is_cuda:
                    freed += w.data.numel() * w.data.element_size()
                w.data = self.emb
        if freed:
            torch.cuda.empty_cache()
        return freed / 2**20

    @torch.no_grad()
    def restore_stock(self):
        """Undo what the fast path changed, for the stock forward: everything back on the GPU and
        the fp32 embedding the stock path expects."""
        try:
            self.ready()
        except BaseException:  # noqa: BLE001 - whatever the build did, put everything back
            pass
        if getattr(self, "_host_thread", None) is not None:
            self._host_ready()
        self.host, self.host_cap = {}, 0
        self.pipe, self.emb_on_host = None, False
        restore_to_gpu(self.peft_model, self.device)

    # ------------------------------------------------------------------ start-up self-test
    def self_test(self):
        """Run the configured placement once before training: the LM head on a few random rows
        against an independent fp32 reference (plain torch ops, the embedding copied chunk by
        chunk), then one forward + backward of a small synthetic batch through every kernel, stream
        and copy path of the step (finite loss, statistics and LoRA gradients). Raises on any
        failure; the caller then falls back (integration._runner_ready). Leaves no trace: the
        dropout generator, step counter and LoRA gradients are restored."""
        self.ready()
        gen_state, steps, n_dq = self.gen.get_state(), self.steps, self.n_dequant
        try:
            started = time.perf_counter()
            head = self._head_check()
            step = self._step_check()
            torch.cuda.synchronize(self.device)
            return f"head {head}; step {step}; {time.perf_counter() - started:.2f} s"
        finally:
            for p in self.lora_params:
                p.grad = None
            self.gen.set_state(gen_state)
            self.steps, self.n_dequant = steps, n_dq
            self._pending = None

    @torch.no_grad()
    def _head_check(self, n=8):
        dev, D, V = self.device, self.D, self.V
        g = torch.Generator(device="cpu")
        g.manual_seed(20261031)
        yf = (torch.randn(n, D, generator=g) * 2.0).to(torch.float16)
        tgt = torch.randint(0, V, (n,), generator=g)
        tgt[1] = -1                                     # rows without a target sort last
        valid = tgt >= 0
        perm = torch.cat([valid.nonzero().flatten(), (~valid).nonzero().flatten()])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        n_lab = int(valid.sum())
        tgt_sorted = tgt[perm]
        yf_d = yf.to(dev)
        tgt_d, perm_d, inv_d = (t.to(dev, torch.int32) for t in (tgt_sorted, perm, inv))
        gs = float(config.HEAD_GRAD_SCALE)
        if self.head_impl == "rows":
            _, stats, dy = fused_head(yf_d, self.emb, tgt_d, perm_d, inv_d, n_lab, config.HEAD_CHUNK, gs)
        else:
            e_target = self.emb_source.rows(tgt_d[:n_lab], tgt_sorted[:n_lab])
            self.emb_source.start()
            _, stats, dy = streamed_head(yf_d, self.emb_source, tgt_d, perm_d, n_lab, e_target, gs,
                                         self.dummy32)
        # reference: fp32 logits and fp32 online softmax over vocabulary chunks, plain torch ops,
        # the embedding copied chunk by chunk (not through the ring)
        ys = yf_d.index_select(0, perm_d).float()
        m = torch.full((n,), float("-inf"), device=dev)
        s = torch.zeros(n, device=dev)
        tt = torch.zeros(n, device=dev)            # sum exp(x - m) (x - m)
        acc = torch.zeros((n, D), device=dev)      # sum exp(x - m) E_x
        step = 2048
        for v0 in range(0, V, step):
            ec = self.emb[v0:v0 + step].to(dev).float()
            lg = ys @ ec.t()
            m_new = torch.maximum(m, lg.max(1).values)
            a = torch.exp(m - m_new)               # 0 on the first chunk
            z = lg - m_new[:, None]
            pe = torch.exp(z)
            tt = torch.where(s > 0, a * (tt + s * (m - m_new)), torch.zeros_like(tt)) + (pe * z).sum(1)
            s = s * a + pe.sum(1)
            acc = acc * a[:, None] + pe @ ec
            m = m_new
            del ec, lg, z, pe
        lse = m + torch.log(s)
        ent = torch.log(s) - tt / s
        e_t = self.emb.index_select(0, tgt_sorted[:n_lab].to(self.emb.device)).to(dev).float()
        loss = lse[:n_lab] - (ys[:n_lab] * e_t).sum(1)
        d_ref = acc[:n_lab] / s[:n_lab, None] - e_t
        d_got = dy.index_select(0, perm_d).float()[:n_lab] / gs
        err = {
            "loss": float((stats[:n_lab, 0] - loss).abs().max()),
            "lse": float((stats[:, 3] - lse).abs().max()),
            "entropy": float((stats[:, 1] - ent).abs().max()),
            "d_hidden_rel": float(((d_got - d_ref).norm(dim=1) / d_ref.norm(dim=1).clamp_min(1e-30)).max()),
            "untargeted_rows_zero": bool((dy.index_select(0, perm_d)[n_lab:] == 0).all()),
        }
        ok = (err["loss"] < 3e-2 and err["lse"] < 3e-2 and err["entropy"] < 5e-2
              and err["d_hidden_rel"] < 3e-2 and err["untargeted_rows_zero"])
        text = ", ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in err.items())
        if not ok:
            raise RuntimeError(f"LM head self-test mismatch ({self.head_impl} head): {text}")
        return f"{self.head_impl} ok ({text})"

    def _step_check(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(31102026)
        lens = [97, 61]
        B, T = len(lens), max(lens)
        ids = torch.randint(0, self.V, (B, T), generator=g)
        mask = (torch.arange(T)[None, :] < torch.tensor(lens)[:, None]).to(torch.int64)
        labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
        labels[:, :24] = -100                            # a prompt part without targets
        batch = self.pack(ids, mask, labels)
        if batch is None:
            raise RuntimeError("self-test batch could not be packed")
        self._rx_check = {"max": 0.0}
        self._gu_check = {"max": 0.0, "rows": 0, "checked": 0}
        try:
            loss_sum, stats = run_step(self, batch)
            loss_sum.backward()
            torch.cuda.synchronize(self.device)
        finally:
            rx = self._rx_check
            self._rx_check = None
            guc = self._gu_check
            self._gu_check = None
        rx_note = ""
        if guc["rows"]:
            # the recomputed gate|up rows must be the forward's (the same kernels on the same
            # inputs; a different cuBLAS algorithm would differ in the last fp16 bits at most)
            rx_note += (f", gate|up rows recomputed in the backward {guc['rows']} (max rel. diff "
                        f"{guc['max']:.1e} over {guc['checked']} layers)")
            if not (guc["max"] < 1e-2 and guc["checked"] > 0):
                self._overrides["GU_RECOMPUTE"] = 0.0
                rx_note += " -> recompute off, every row kept"
        elif self._gu_tail(batch.n) and getattr(self.last_plan, "gu", False):
            # the path was planned but never ran: do not use it untested
            self._overrides["GU_RECOMPUTE"] = 0.0
            rx_note += ", gate|up recompute not exercised -> off"
        if getattr(self.last_plan, "rx", False):
            rx_note += f", residual rebuild max rel. err {rx['max']:.1e}"
            if not rx["max"] < 1e-4:
                # keep the fp32 checkpoints instead (and their arena) for the real steps
                self._overrides["RECONSTRUCT_X"] = False
                self.start_host_alloc(self.host_cap or int(config.HOST_CAPACITY), {"x"})
                rx_note += " -> rebuild off, fp32 checkpoints kept"
        bad = [i for i, p in enumerate(self.lora_params)
               if p.grad is None or not bool(torch.isfinite(p.grad).all())]
        if not bool(torch.isfinite(loss_sum)) or not bool(torch.isfinite(stats).all()) or bad:
            raise RuntimeError(f"self-test step not finite (loss {float(loss_sum)}, "
                               f"{len(bad)} bad LoRA gradients)")
        return f"ok ({batch.n} tokens, loss/token {float(loss_sum) / max(batch.n_lab, 1):.2f}{rx_note})"

    # ------------------------------------------------------------------ helpers
    def _rope_table(self, n):
        with torch.no_grad():
            inv = getattr(self.rotary, "inv_freq", None)
            rdev = inv.device if isinstance(inv, torch.Tensor) else self.device
            pos = torch.arange(n, device=rdev)[None, :]
            x = torch.zeros(1, dtype=torch.float32, device=rdev)
            cos, sin = self.rotary(x, pos)
        self.cos = cos[0].float().to(self.device).contiguous()
        self.sin = sin[0].float().to(self.device).contiguous()
        _require(self.cos.shape == (n, self.HD), "rotary table shape")

    @torch.no_grad()
    def _refresh_lora16(self):
        try:
            torch._foreach_copy_(self.lora16, self.lora_params)
        except Exception:  # noqa: BLE001 - older torch: plain copies
            for dst, src in zip(self.lora16, self.lora_params):
                dst.copy_(src)

    def _ws_load(self, part, i):
        """Start dequantising `part` ("attn" or "mlp") of layer i into the workspace, unless the
        workspace already holds (or is already receiving) it. With streamed weights the packed
        layer is read from its GPU slot; the dequantising stream waits for the slot's copy."""
        if i is None or self.ws_have[part] == i:
            return
        q0, q1 = (0, self.split_q) if part == "attn" else (self.split_q, self.packed[i].n_qblocks)
        if self.overlap:
            side = self.dq_stream
            side.wait_event(self.ev_free[part])         # the main stream's last reader of the part
            src = self.pipe.acquire(i, side) if self.pipe is not None else self.packed[i]
            with torch.cuda.stream(side):
                dequant_range(src, q0, q1, self.code2, self.nf4, self.ws)
            self.ev_ready[part].record(side)
        else:
            main = torch.cuda.current_stream(self.device)
            src = self.pipe.acquire(i, main) if self.pipe is not None else self.packed[i]
            dequant_range(src, q0, q1, self.code2, self.nf4, self.ws)
        self.ws_have[part] = i
        self.n_dequant += 1

    def _ws_use(self, part, i):
        """The main stream is about to read `part` of layer i from the workspace."""
        self._ws_load(part, i)
        if self.overlap:
            torch.cuda.current_stream(self.device).wait_event(self.ev_ready[part])

    def _ws_release(self, part, nxt):
        """The main stream has issued its last read of `part`; prefetch layer `nxt`'s part."""
        if self.overlap:
            self.ev_free[part].record(torch.cuda.current_stream(self.device))
        self._ws_load(part, nxt)

    def _weights_ahead(self, j):
        """Both halves of the current layer have been queued for dequantisation: its slot may
        receive layer j (two layers further along the walk)."""
        if self.pipe is not None:
            self.pipe.prefetch(j)

    def _seeds(self):
        return torch.randint(1, 2 ** 31 - 1, (2 * self.L,), generator=self.gen).tolist()

    # ------------------------------------------------------------------ deferred metrics
    def defer(self, fn):
        """Run fn (recording the step's metrics from its stats) at the end of this step's backward,
        i.e. once the whole backward is queued, instead of synchronising after the forward."""
        self.flush()
        self._pending = fn

    def flush(self):
        fn, self._pending = self._pending, None
        if fn is not None:
            fn()

    # ------------------------------------------------------------------ batch packing
    def pack(self, input_ids, attention_mask, labels):
        """Host-side packing of a right-padded batch. Returns None if the batch is not one we can
        pack exactly (then the caller uses the stock path)."""
        self.ready()
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
        ids = input_ids.to("cpu", torch.int64)[sel]
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
        tgt_sorted = tgt[perm]
        parts = [ids, pos, tgt_sorted, perm, inv, torch.tensor(cu)]
        if self.attn_impl == "triton":
            parts += [torch.tensor(block_table(lens_l, FWD_BLOCK_M)), torch.tensor(block_table(lens_l, BWD_BLOCK))]
        offs, chunks, at = [], [], 0
        for p in parts:
            p = p.to(torch.int32).flatten()
            pad = (-p.numel()) % ALIGN
            offs.append((at, p.numel()))
            chunks.append(p)
            if pad:
                chunks.append(torch.zeros(pad, dtype=torch.int32))
            at += p.numel() + pad
        host = torch.cat(chunks).pin_memory()
        dev = host.to(self.device, non_blocking=True)
        views = [dev[o:o + c] for o, c in offs]
        batch = _Batch()
        batch.n, batch.B, batch.lens = n, B, lens_l
        batch.ids, batch.pos, batch.tgt_sorted, batch.perm, batch.inv = views[:5]
        batch.cu = views[5]
        if self.attn_impl == "triton":
            tab_f, tab_b = views[6], views[7]
            batch.meta = AttnMeta(batch.cu, tab_f, tab_b, tab_f.numel() // 2, tab_b.numel() // 2, n)
        batch.n_lab = int(valid.sum())
        batch.Tm = max(lens_l)
        if self.attn_impl == "sdpa":
            bt = torch.stack([row, pos]).to(self.device)   # packed token -> (row, position)
            batch.b_idx, batch.t_idx = bt[0], bt[1]
        batch.row_sorted = row[perm]
        batch.valid_sorted = valid[perm]
        batch.n_real = n
        # embedding rows: the inputs, and the targets' rows for the head's (p_t - 1) * E_t term
        batch.x0 = self.emb_source.rows(batch.ids, ids)
        nl = batch.n_lab
        batch.e_target = self.emb_source.rows(batch.tgt_sorted[:nl], tgt_sorted[:nl]) if nl > 0 else None
        return batch

    # ------------------------------------------------------------------ activation storage
    HOST_KINDS = ("x", "d", "gu", "qkv", "out", "op")

    def arena_widths(self):
        return {"x": (self.D, torch.float32), "d": (self.D, torch.float16),
                "gu": (2 * self.I, torch.float16),
                "qkv": ((self.NQ + 2 * self.NKV) * self.HD, torch.float16),
                "out": (self.NQ * self.HD, torch.float16), "op": (self.D, torch.float16)}

    def start_host_alloc(self, cap, kinds=None):
        """Get the pinned host arenas (rows for layers 0..L-2; the last layer's tensors are used
        right after the forward and stay on the GPU) from the early pool, or allocate them, in a
        background thread."""
        self._host_ready()
        if kinds is None:
            kinds = set(arena_kinds())
        if self.attn_impl != "varlen":
            kinds -= {"qkv", "out", "op"}
            if "d" in kinds:                     # reconstruction needs the saved O outputs
                kinds = (kinds - {"d"}) | {"x"}
        width = self.arena_widths()
        kinds = sorted(k for k in kinds if k not in self.host)
        if not kinds or cap <= 0:
            return

        def work():
            started = time.perf_counter()
            try:
                made = {k: hostmem.take(f"act_{k}", (self.L - 1, cap, width[k][0]), width[k][1])
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
        plan.x = fits and self._opt("OFFLOAD_CHECKPOINTS") and "x" in self.host
        plan.gu = fits and config.SAVE_GU and "gu" in self.host
        plan.attn = (fits and config.SAVE_ATTN and self.attn_impl == "varlen"
                     and all(k in self.host for k in ("qkv", "out", "op")))
        plan.rx = bool(plan.attn and self._opt("RECONSTRUCT_X") and "d" in self.host)
        plan.gu_tail = self._gu_tail(n) if plan.gu else 0
        plan.n = n
        plan.arenas = self.host
        plan.last = self.L - 1
        return plan

    def _gu_tail(self, n):
        """v13 (config.GU_RECOMPUTE, GU_RECOMPUTE_ROWS): how many trailing token rows of each
        layer's gate|up output are recomputed in the backward instead of going through host memory
        (a multiple of 16, at most n / 3; 0 for tiny batches or when switched off)."""
        try:
            f = float(self._opt("GU_RECOMPUTE") or 0.0)
            base = int(self._opt("GU_RECOMPUTE_ROWS") or 0) if f > 0.0 else 0
        except (TypeError, ValueError):
            return 0
        if (f <= 0.0 and base <= 0) or n < 64:
            return 0
        k = min(int(base + n * min(f, 0.5)), n // 3) // 16 * 16
        return k if 0 < k < n else 0

    def _gu_rows(self, gu, x, op, i, tail):
        """Rows [n - tail, n) of layer i's gate|up output, as the forward computed them: norm 2 of
        x + o_proj for those rows (the same kernel; RMSNorm is per row and norm 2 has no dropout),
        then the gate|up GEMM on exactly the forward's last row block."""
        n, D = x.shape
        m = n - tail
        y2r = torch.empty((tail, D), dtype=torch.float16, device=self.device)
        rs = torch.empty(tail, dtype=torch.float32, device=self.device)
        rms_forward(x[m:], self.w_post[i], y2r, rs, self.eps, delta=op[m:],
                    dummy16=self.dummy16, dummy32=self.dummy32)
        torch.mm(y2r, self.W_gu.t(), out=gu[m:])
        del y2r, rs

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
        self._timeline_start(batch)
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
        # norm 2's rstd per layer (4 bytes per token): with gate|up kept, the backward needs no
        # norm-2 recompute at all (rms_backward adds the O output to the checkpoint in registers)
        # (rows padded to 16 bytes: Triton specialises pointers on 16-byte alignment, and an
        # unaligned row would compile another variant of the norm kernels mid-run)
        rstd2_all = torch.empty((self.L, -(-n // 4) * 4), dtype=f32, device=dev)[:, :n]
        x = batch.x0.float()
        batch.x0 = None
        d16 = None
        for i in range(self.L):
            if i == self.L - 1:
                self.emb_source.start()        # the head's first embedding chunks, under the last layer
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
            if not plan.rx:
                store.put(i, "x", x, plan.x)
            elif self._rx_check is not None:
                store.put(i, "xchk", x, False)      # self-test: the real x, to check the rebuild
            nxt = i + 1 if i + 1 < self.L else None
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
                # the rank-16 LoRA-A projections (64 bytes per token): no recompute GEMMs later
                store.put(i, "aq", _aq, False)
                store.put(i, "av", _av, False)
            del qkv, _aq, _av, out, actx
            h1 = torch.empty((n, D), dtype=f32, device=dev)
            rms_forward(x, self.w_post[i], y2, rstd2_all[i], self.eps, delta=op, xout=h1,
                        dummy16=self.dummy16, dummy32=self.dummy32)
            del op
            if plan.rx:
                x = None                         # not kept: the backward rebuilds it
            self._ws_use("mlp", i)
            self._weights_ahead(i + 2)
            if config.EARLY_RETIRE:
                # layer i-1's saved tensors (gate|up above all) leave the GPU before gate|up of
                # layer i is allocated: its copy was queued right after layer i-1's gate|up GEMM,
                # i.e. a down GEMM and an attention block ago. One layer of saved activations
                # less at the forward's peak (~44 MB for 838 tokens).
                store.retire(i - 1)
            hosted = plan.gu and i < self.L - 1
            parts = int(config.GU_SPLIT) if hosted else 1
            tail = plan.gu_tail if hosted else 0
            if (parts > 1 and n >= 32 * parts) or tail:
                # gate|up in row blocks: each block's copy to the host arena starts as soon as its
                # GEMM is done, so layer i's copy is complete when layer i+1 retires it (above).
                # v13: the last `tail` rows are computed as one more block but not copied out; the
                # backward recomputes exactly that block (same rows, same GEMM shape).
                gu = torch.empty((n, 2 * self.I), dtype=f16, device=dev)
                m = n - tail
                if parts > 1 and m >= 32 * parts:
                    edges = [0] + [min(m, -(-(m * k // parts) // 16) * 16) for k in range(1, parts)] + [m]
                else:
                    edges = [0, m]
                for r0, r1 in zip(edges[:-1], edges[1:]):
                    if r1 > r0:
                        torch.mm(y2[r0:r1], self.W_gu.t(), out=gu[r0:r1])
                        store.put_rows(i, "gu", gu, r0, r1)
                if tail:
                    torch.mm(y2[m:], self.W_gu.t(), out=gu[m:])
                    store.keep_rows(i, "gu", m)
                    if getattr(self, "_gu_check", None) is not None and i % 3 == 0:
                        store.put(i, "guchk", gu[m:].clone(), False)   # self-test reference
                act = torch.empty((n, self.I), dtype=f16, device=dev)
                gated_silu_forward(gu, act)
            else:
                gu = torch.mm(y2, self.W_gu.t())
                act = torch.empty((n, self.I), dtype=f16, device=dev)
                gated_silu_forward(gu, act)
                if plan.gu:
                    store.put(i, "gu", gu, True)    # the backward reads it instead of a gate|up GEMM
            del gu
            d16 = torch.mm(act, self.W_d.t())
            if plan.rx:
                # the MLP output (fp16) instead of the fp32 layer input: the backward rebuilds the
                # residual stream from the top, x_i = (x_{i+1} - d_i) - o_i
                store.put(i, "d", d16, True)
            self._ws_release("mlp", nxt)
            del act
            x = h1
            self._tl_mark(f"f{i}")
        del y, xq, xv, y2
        store.retire(self.L)
        hf = torch.empty((n, D), dtype=f32, device=dev)
        yf = torch.empty((n, D), dtype=f16, device=dev)
        rstd_f = torch.empty(n, dtype=f32, device=dev)
        rms_forward(x, self.w_final, yf, rstd_f, self.eps, delta=d16, xout=hf,
                    dummy16=self.dummy16, dummy32=self.dummy32)
        del x, d16
        self._tl_mark("head0")
        loss_sum, stats, dy_head = self._head(yf, batch)
        self._tl_mark("head1")
        del yf
        batch.e_target = None
        self.steps += 1
        saved = dict(store=store, plan=plan, hf=hf, rstd_f=rstd_f, dy_head=dy_head, seeds=seeds,
                     batch=batch, rstd2=rstd2_all)
        return loss_sum, stats, saved

    # ------------------------------------------------------------------ step timeline (log only)
    def _tl_mark(self, name):
        tl = getattr(self, "_tl", None)
        if tl is not None and not tl.get("done"):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            tl["ev"].append((name, ev, time.perf_counter()))

    def _timeline_start(self, batch):
        """config.STEP_TIMELINE: CUDA events on the main stream at every layer of a few steps;
        the next step's start prints where the GPU time of the step went (nothing else changes)."""
        tl = getattr(self, "_tl", None)
        if tl is not None:
            self._tl = None
            try:
                self._timeline_report(tl)
            except Exception as exc:  # noqa: BLE001 - diagnostics never disturb the run
                print(f"[solution] timeline: not available ({type(exc).__name__}: {exc})",
                      flush=True)
        if config.STEP_TIMELINE and self.steps in config.STEP_TIMELINE:
            self._tl = {"ev": [], "n": batch.n, "n_lab": batch.n_lab, "step": self.steps + 1}
            self._tl_mark("start")

    def _timeline_report(self, tl):
        import sys
        nxt = torch.cuda.Event(enable_timing=True)
        nxt.record()
        nxt.synchronize()
        ev = dict((name, e) for name, e, _ in tl["ev"])
        cpu = dict((name, t) for name, _, t in tl["ev"])
        if "b0" not in ev:
            return

        def ms(a, b):
            return ev[a].elapsed_time(ev[b])

        L = self.L
        fl = [ms("start" if i == 0 else f"f{i - 1}", f"f{i}") for i in range(L)]
        bl = [ms("b" if i == L - 1 else f"b{i + 1}", f"b{i}") for i in reversed(range(L))]

        def dist(v):
            v = sorted(v)
            return f"{v[0]:.2f}/{v[len(v) // 2]:.2f}/{v[-1]:.2f}"

        print(f"[solution] timeline step {tl['step']} ({tl['n']} tokens, {tl['n_lab']} supervised): "
              f"forward {ms('start', f'f{L - 1}'):.1f} ms (layer min/med/max {dist(fl)}), "
              f"final norm + head {ms(f'f{L - 1}', 'head1'):.1f} ms, autograd turn "
              f"{ms('head1', 'b'):.1f} ms, backward {ms('b', 'b0'):.1f} ms (layer {dist(bl)}), "
              f"end of backward -> next step {ev['b0'].elapsed_time(nxt):.1f} ms; "
              f"CPU: forward queued in {1000 * (cpu['head1'] - cpu['start']):.0f} ms, backward in "
              f"{1000 * (cpu['b0'] - cpu['b']):.0f} ms, GPU finished the forward "
              f"{ms('start', 'head1') - 1000 * (cpu['head1'] - cpu['start']):.0f} ms after the CPU "
              f"had queued it", file=sys.stderr, flush=True)

    def _head(self, yf, batch):
        # the head's GEMMs (incl. the fp32-accumulating one) keep PyTorch's default library choice
        with _blas("default" if config.BLAS_LT else None):
            if self.head_impl == "rows":
                return fused_head(yf, self.emb, batch.tgt_sorted, batch.perm, batch.inv, batch.n_lab,
                                  config.HEAD_CHUNK, config.HEAD_GRAD_SCALE)
            return streamed_head(yf, self.emb_source, batch.tgt_sorted, batch.perm, batch.n_lab,
                                 batch.e_target, config.HEAD_GRAD_SCALE, self.dummy32)

    # ------------------------------------------------------------------ backward
    def backward(self, saved, g_loss):
        batch, store, seeds, plan = saved["batch"], saved["store"], saved["seeds"], saved["plan"]
        n, D = batch.n, self.D
        dev = self.device
        f16, f32 = torch.float16, torch.float32
        self._tl_mark("b")
        gscale = (g_loss.detach().reshape(1).to(f32) / float(config.HEAD_GRAD_SCALE)).contiguous()
        dh = torch.empty((n, D), dtype=f32, device=dev)
        dh16 = torch.empty((n, D), dtype=f16, device=dev)     # fp16 copy of dh for the GEMMs
        x_next = saved.pop("hf")
        rms_backward(saved.pop("dy_head"), gscale, x_next, self.w_final, saved.pop("rstd_f"),
                     dh, out16=dh16, dummy16=self.dummy16, dummy32=self.dummy32)
        # y / y2 (the GEMM inputs of norm 1 / norm 2) only for what is recomputed
        y = None if plan.attn else torch.empty((n, D), dtype=f16, device=dev)
        xq = torch.empty((n, D), dtype=f16, device=dev)
        xv = torch.empty((n, D), dtype=f16, device=dev)
        y2 = None if plan.gu else torch.empty((n, D), dtype=f16, device=dev)
        rstd1 = torch.empty(n, dtype=f32, device=dev)
        rstd2 = torch.empty(n, dtype=f32, device=dev)
        rstd2_all = saved.pop("rstd2")
        grads = [None] * len(self.lora_params)
        s = self.lora_scale
        early_w = bool(config.BWD_WEIGHTS_EARLY)
        for i in reversed(range(self.L)):
            if early_w:
                # layer i's two halves were queued for dequantisation in iteration i + 1 (or in the
                # forward, for the last layer), so slot i % 2 may receive layer i - 2 now: its copy
                # runs under this layer's MLP backward instead of next to the gate|up prefetch
                self._weights_ahead(i - 2)
            op_top = None
            if plan.rx:
                # the residual stream backwards from the final hidden state: the forward computed
                # h_i = x_i + o_i and x_{i+1} = h_i + d_i in fp32; undoing both additions gives
                # back x_i to within an fp32 rounding (~1e-7 relative)
                d_i = store.get(i, "d")
                op_top = store.get(i, "op")
                x = torch.sub(x_next, d_i).sub_(op_top)
                x_next = x
                del d_i
                if self._rx_check is not None:
                    xs = store.get(i, "xchk")
                    err = float((x - xs).abs().max() / xs.abs().max().clamp_min(1e-30))
                    self._rx_check["max"] = max(self._rx_check["max"], err)
                    del xs
                if i > 0:
                    store.prefetch(i - 1, "d")
                    store.prefetch(i - 1, "op")
            else:
                x = store.get(i, "x")
                x_next = None
                if i > 0:
                    store.prefetch(i - 1, "x")
                    store.prefetch(i - 1, "op")
            aq16, bq16, av16, bv16 = self.lora16[4 * i:4 * i + 4]
            prv = i - 1 if i > 0 else None
            # norm 1 again: the dropout-masked LoRA inputs (same Philox seeds) and rstd; y itself
            # only when the attention is recomputed
            rms_forward(x, self.w_in[i], y, rstd1, self.eps, xq=xq, xv=xv,
                        p_drop=self.p_drop, keep_scale=self.keep_scale, seed_q=seeds[2 * i],
                        seed_v=seeds[2 * i + 1], dummy16=self.dummy16, dummy32=self.dummy32)
            if plan.attn:
                aq = store.get(i, "aq")
                av = store.get(i, "av")
                op = op_top if op_top is not None else store.get(i, "op")
                op_top = None
            else:
                self._ws_use("attn", i)
                qkv, aq, av, out, actx = self._attn_part(i, y, xq, xv, batch, grad=True)
                op = torch.mm(out, self.W_o.t())
            if plan.gu:
                # norm 2's input is x + op: rms_backward rebuilds it in registers, with the
                # forward's rstd; nothing of norm 2 is recomputed
                h1, rstd2_i, delta2 = x, rstd2_all[i], op
                del op
            else:
                h1 = torch.empty((n, D), dtype=f32, device=dev)
                rms_forward(x, self.w_post[i], y2, rstd2, self.eps, delta=op, xout=h1,
                            dummy16=self.dummy16, dummy32=self.dummy32)
                rstd2_i, delta2 = rstd2, None
                del op
            self._ws_use("mlp", i)
            # MLP backward. v13: d(act) needs only dh and W_down, so it is issued before the main
            # stream waits for the gate|up copy (v12 waited first): the GEMM runs under the copy.
            # Same live tensors as before (the gate|up buffer was allocated by its prefetch).
            dact = torch.mm(dh16, self.W_d)
            if plan.gu:
                tail = store.tail_rows(i, "gu")
                if tail:
                    # v13: the rows the forward did not copy out, recomputed into the prefetched
                    # tensor while the copy of the other rows is still in flight
                    gbuf = store.peek(i, "gu")
                    self._gu_rows(gbuf, x, delta2, i, tail)
                    if getattr(self, "_gu_check", None) is not None:
                        self._gu_check["rows"] += tail
                        if (i, "guchk") in store.gpu:
                            ref = store.get(i, "guchk").float()
                            got = gbuf[n - tail:].float()
                            err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                            self._gu_check["max"] = max(self._gu_check["max"], err)
                            self._gu_check["checked"] += 1
                            del ref, got
                    del gbuf
                gu = store.get(i, "gu")
            else:
                gu = torch.mm(y2, self.W_gu.t())
            gated_silu_backward(gu, dact)
            del dact
            dy2 = torch.mm(gu, self.W_gu)
            self._ws_release("mlp", prv)        # layer i-1's MLP weights dequantise under the attention backward
            del gu
            if plan.gu and i > 0:
                store.prefetch(i - 1, "gu")     # into the block gu just released
            rms_backward(dy2, self.one, h1, self.w_post[i], rstd2_i, dh, dres=dh, out16=dh16,
                         delta=delta2, dummy16=self.dummy16, dummy32=self.dummy32)
            del dy2, h1, delta2
            # attention backward
            if plan.attn:
                self._ws_use("attn", i)
            if not early_w:
                self._weights_ahead(i - 2)
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
            self._tl_mark(f"b{i}")
        store.close()
        self.flush()          # the whole backward is queued: record this step's metrics now
        return grads


class _Batch:
    pass


class _Plan:
    """Per-step choice of what the forward keeps for the backward (see FastQwen2._save_plan)."""
    x = gu = attn = rx = False
    gu_tail = 0
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
    keep_rows(i, kind, m) (v13): only rows [0, m) of the tensor went to the host; prefetch copies
    those rows back into a full-size tensor, whose remaining rows the backward recomputes on the
    main stream (tail_rows / peek) while the copy is in flight (disjoint rows, so no ordering is
    needed between the two writers; get() orders the copy before every later read).
    """

    IN_FLIGHT = 2

    def __init__(self, model, n):
        self.m, self.n = model, n
        self.side = model.copy_stream
        self.gpu, self.pending, self.where, self.inflight = {}, {}, {}, {}
        self.kept = {}

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

    def put_rows(self, i, kind, t, r0, r1):
        """Copy rows [r0, r1) of t (written by the main stream up to now) into row i of the
        arena; the other rows follow in later calls, before the tensor is retired or read."""
        arena = self.m.host.get(kind) if i < self.m.L - 1 else None
        if arena is None:
            self.gpu[(i, kind)] = t
            return
        main = torch.cuda.current_stream(self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            arena[i, r0:r1].copy_(t[r0:r1], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.side)
        self.inflight.setdefault(i, []).append((t, done))
        self.where[(i, kind)] = (arena, t.dtype, tuple(t.shape))

    def keep_rows(self, i, kind, m):
        """Rows [m, n) of (i, kind) were not copied out (put_rows covered [0, m) only)."""
        if (i, kind) in self.where and 0 < m < self.n:
            self.kept[(i, kind)] = int(m)

    def tail_rows(self, i, kind):
        """How many trailing rows of (i, kind) the caller has to recompute (0 = none)."""
        m = self.kept.get((i, kind))
        return 0 if m is None else self.n - m

    def peek(self, i, kind):
        """The GPU tensor of (i, kind) without waiting for its copy (to write the rows the copy
        does not cover)."""
        key = (i, kind)
        if key in self.gpu:
            return self.gpu[key]
        if key not in self.pending:
            self.prefetch(i, kind)
        return self.pending[key][0]

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
        m = self.kept.get(key, self.n)
        main = torch.cuda.current_stream(self.m.device)
        g = torch.empty(shape, dtype=dtype, device=self.m.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ready)
            if m == self.n:
                g.copy_(arena[i, :self.n], non_blocking=True)
            else:
                g[:m].copy_(arena[i, :m], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.side)
        self.pending[key] = (g, done)

    def get(self, i, kind):
        key = (i, kind)
        if key in self.gpu:
            self.kept.pop(key, None)
            return self.gpu.pop(key)
        if key not in self.pending:
            self.prefetch(i, kind)
        self.kept.pop(key, None)
        g, done = self.pending.pop(key)
        torch.cuda.current_stream(self.m.device).wait_event(done)
        return g

    def close(self):
        self.retire(10 ** 9)
        self.gpu.clear()
        self.pending.clear()
        self.where.clear()
        self.kept.clear()


@contextlib.contextmanager
def _blas(backend):
    """Prefer cuBLAS or cuBLASLt for the GEMMs inside the block (config.BLAS_LT); None = as is."""
    prev = None
    if backend is not None:
        try:
            prev = torch.backends.cuda.preferred_blas_library()
            if str(prev).lower().endswith(backend):
                prev = None
            else:
                torch.backends.cuda.preferred_blas_library(backend)
        except Exception:  # noqa: BLE001 - not available: keep the default
            prev = None
    try:
        yield
    finally:
        if prev is not None:
            torch.backends.cuda.preferred_blas_library(prev)


class FusedCausalLM(torch.autograd.Function):
    """Whole-model autograd node: inputs are the LoRA parameters, output the summed CE."""

    @staticmethod
    def forward(ctx, runner, batch, *lora_params):
        with torch.autocast("cuda", enabled=False), _blas("cublaslt" if config.BLAS_LT else None):
            loss_sum, stats, saved = runner.forward(batch)
        ctx.runner, ctx.saved = runner, saved
        ctx.mark_non_differentiable(stats)
        return loss_sum, stats

    @staticmethod
    def backward(ctx, g_loss, _g_stats):
        with torch.autocast("cuda", enabled=False), _blas("cublaslt" if config.BLAS_LT else None):
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


def arena_kinds():
    """Activation kinds kept in host arenas with the current switches (varlen attention assumed)."""
    kinds = []
    if config.OFFLOAD_CHECKPOINTS:
        reconstruct = config.RECONSTRUCT_X and config.SAVE_ATTN and config.ATTN_IMPL == "varlen"
        kinds.append("d" if reconstruct else "x")
    if config.SAVE_GU:
        kinds.append("gu")
    if config.SAVE_ATTN:
        kinds += ["qkv", "out", "op"]
    return kinds


def pinned_specs(shapes=QWEN15):
    """The pinned host buffers FastQwen2 will ask hostmem for with the current switches, sized for
    Qwen2.5-1.5B: [(name, shape, dtype)], in the order they are needed."""
    L, D, I, NQ, NKV, HD, V = (shapes[k] for k in ("L", "D", "I", "NQ", "NKV", "HD", "V"))
    specs = []
    if config.OFFLOAD_WEIGHTS:
        n_layer = (NQ + 2 * NKV) * HD * D + D * NQ * HD + 3 * I * D
        _, layer_bytes = section_layout(n_layer // 64)
        specs.append(("weights", (L * layer_bytes,), torch.uint8))
    if config.OFFLOAD_EMBEDDING:
        specs.append(("embedding", (V, D), torch.float16))
    width = {"x": (D, torch.float32), "d": (D, torch.float16), "gu": (2 * I, torch.float16),
             "qkv": ((NQ + 2 * NKV) * HD, torch.float16), "out": (NQ * HD, torch.float16),
             "op": (D, torch.float16)}
    cap = int(config.HOST_CAPACITY)
    if cap > 0:
        for k in sorted(arena_kinds()):
            specs.append((f"act_{k}", (L - 1, cap, width[k][0]), width[k][1]))
    return specs


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
    head = "rows" if (config.HEAD_IMPL == "rows" and not config.OFFLOAD_EMBEDDING) else "chunks"
    specs = (k_nf4.warmup_specs() + k_norm.warmup_specs(hidden) + epi
             + k_mlp.warmup_specs(inter) + k_head.warmup_specs(hidden, config.HEAD_VOCAB_CHUNK, head))
    # (v8: the v5 head of the fallback placement is no longer precompiled; it compiles on demand
    # in the rare run that falls back)
    if impl == "triton":
        specs += k_attn.warmup_specs(nq, nkv, hd)
    return specs


_ = math  # keep import for readers computing scales by hand
