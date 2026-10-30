# Soup Kernel Challenge — a padding-free, recomputing, fused-kernel QLoRA step

Our solution makes the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4 4-bit, LoRA r=16 on
`q_proj`/`v_proj`, batch 4, ≤512 tokens, 64 steps, Kaggle T4) **faster and much lighter** with
**our own Triton kernels** (8 in the default configuration, 12 in total) and our own forward/backward of the model, plugged into Soup
through `solution.apply()`. It trains the same adapter: same model, data, batches, LoRA
(dropout included), optimizer, schedule, loss, and the same Soup outputs (loss, entropy, token
accuracy, `num_tokens`, `rewind.jsonl`).

| Links | |
|---|---|
| Public Kaggle notebook (final run) | `<KAGGLE NOTEBOOK LINK>` |
| Final commit | `<COMMIT LINK>` |

## Results (Kaggle T4, `harness/run_task.py --repeats 2`, public eval rows)

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Whole `soup train` command, s | `<base>` | `<ours>` | speedup `<s>`× |
| of which the 64 training steps, s | `<base>` | `<ours>` | |
| Peak GPU memory (nvidia-smi), MiB | `<base>` | `<ours>` | base/ours `<m>`× |
| Eval loss (`eval_adapter.py`) | `<base>` | `<ours>` | `<Δ>` % |
| Our kernels' share of step time (check 4) | – | `<share>` % | |

All 10 checks of `harness/check_rules.py`: `<PASS/FAIL>` (output in the notebook).

## Where base Soup spends its time and memory (and what we did about it)

We read the exact code that runs (Soup 0.77.0 `trainer/sft.py` + `trainer/rewind_hf.py`,
TRL 0.29.1 `SFTTrainer.compute_loss`, transformers 5.16.1 Qwen2 + `Trainer`, PEFT 0.20
`lora/bnb.py`, bitsandbytes 0.50.2 `Linear4bit`/`gemm_4bit`, accelerate 1.14). On a T4 the base
step does this:

| Base behaviour | Cost | Our change |
|---|---|---|
| Batches right-padded to the longest row: a mean of ~45 % of the tokens in the GEMMs are padding | ~2× wasted FLOPs | **padding-free packing**: the 4 rows are concatenated, padding is never computed |
| No gradient checkpointing: every layer keeps its activations (gate/up alone are 36 KB per token per layer) | ~7–9 GB | **per-layer recomputation in our own backward**: only the fp32 residual entering each layer is kept (6 KB/token/layer) |
| Logits `[4, T, 151936]` materialised many times: fp16 GEMM out, accelerate's fp32 copy, the loss' fp32 upcast, log-softmax for backward, TRL's entropy chunks, the argmax for accuracy, rewind's per-row CE | several GB + ~10 % of time in memory-bound passes | **fused LM head**: logits only for 256 tokens at a time, one Triton pass gives loss, entropy, argmax-accuracy, per-row loss and the logit gradient; d(hidden) is computed in the forward |
| Tied embedding/`lm_head` upcast to fp32 (934 MB) by `prepare_model_for_kbit_training`, re-cast to fp16 by autocast for every `lm_head` call | 934 MB + 467 MB transient + time | **kept fp16** — bit-exact: on a T4 Soup loads the checkpoint as fp16, so the fp32 copy only ever holds fp16 values |
| fp32 residual stream with ~10 small elementwise kernels per norm, RoPE via `cat`/`rotate_half`, LoRA dropout in fp32, PEFT's `result.clone()`, SwiGLU in fp32 | ~44 % of GPU time | **fused kernels**: residual+RMSNorm+both dropout masks; LoRA-add+RoPE; SwiGLU on a fused buffer |
| bitsandbytes dequantises each of the 7 weights of a layer separately (and again in backward) | ~5 % | **one kernel per layer** dequantises all 7 weights, bit-exactly, straight into fused **Q\|K\|V** and **gate\|up** buffers, so 7 GEMMs become 4 |
| SDPA (memory-efficient) on padded, `repeat_kv`-expanded heads with an explicit padding mask | ~5 % | kept: PyTorch's SDPA on a zero-padded view of the packed batch (right padding needs only the causal mask). Our own varlen causal GQA flash attention in Triton is implemented and tested correct but was ~40× slower than expected on sm_75 (see *What did not work*), so it is off by default |

What stays exactly as in the base, by construction: the dequantised fp16 weights are
bit-identical to bitsandbytes'; the residual stream is fp32; every GEMM is an fp16 cuBLAS GEMM
with fp32 accumulation like the base's; LoRA A/B run as fp16 GEMMs of fp16-cast fp32 weights
(what autocast does); `cos`/`sin` come from the model's own rotary module; the loss is the
summed token CE divided by HF's `num_items_in_batch`; LoRA gradients are returned as fp32
`.grad` to the same parameters, and the HF optimizer, GradScaler, clipping and schedule are
untouched.

## The kernels (all in `solution/`, all ours)

| Kernel (`@triton.jit`) | File | Replaces | How it works |
|---|---|---|---|
| `nf4_layer_dequant_kernel` | `k_nf4.py` | 7× bitsandbytes `dequantize_4bit` per layer per pass | At setup the 7 NF4 weights of a layer are repacked into one buffer `[nibbles \| absmax codes \| 2nd-level scales \| offsets]` (bitsandbytes' own tensors are re-pointed at views of it, nothing is stored twice). One launch per layer decodes `fp16(NF4[nibble] · (code2[q]·absmax2 + offset))` for 8192 weights per program (`tl.join` interleaves the two nibbles of a byte), writing the fused Q\|K\|V, O, gate\|up, down matrices. Launched with FMA fusion off: bit-exact vs bitsandbytes. |
| `rms_fwd_kernel` | `k_norm.py` | Qwen2RMSNorm (fp32, ~6 kernels) + residual add + PEFT's two `lora_dropout` + autocast casts | One program per token: `x = x_prev + delta`, `rstd`, `y = w·(x·rstd)`, writes the fp32 residual (the layer checkpoint), fp16 `y` for the GEMMs and both dropout-masked LoRA inputs. Masks come from Philox (`tl.rand`) keyed by token×column, so backward regenerates them instead of storing them. |
| `rms_bwd_kernel` | `k_norm.py` | autograd of the above + dropout backward + residual-grad accumulation | `dx = rstd·(g − x̂·mean(g·x̂))`, `g = w·(dy + mask_q·dLoRA_q + mask_v·dLoRA_v)`, accumulated in place into the fp32 residual gradient. |
| `qkv_epilogue_fwd_kernel` | `k_rope.py` | PEFT's LoRA add for q/v, `apply_rotary_pos_emb` (`rotate_half`, `cat`), the fp16 casts for SDPA | In place on the fused QKV GEMM output: `q += 2·lora_q`, rotate q and k with the model's cos/sin, `v += 2·lora_v`. |
| `qkv_epilogue_bwd_kernel` | `k_rope.py` | RoPE backward (+ GQA reduction for our attention) | Transposed rotation (summing per-query-head dK/dV partials when our attention kernel produced them), writes the fp16 grad of the fused QKV output (input to the fused weight-grad GEMM and the LoRA grads). |
| `attn_fwd_kernel` *(off by default)* | `k_attn.py` | SDPA memory-efficient attention | FlashAttention-2 schedule over a per-batch table of (sequence, block) pairs: blocks never straddle two packed rows, so the block-diagonal causal mask is just `m ≥ n`; GQA reads KV head `h//6` directly; exp2-domain online softmax; stores fp16 output and log-sum-exp. Block 64×32 for sm_75 (64 KB smem, no cp.async). |
| `attn_bwd_delta_kernel`, `attn_bwd_dkdv_kernel`, `attn_bwd_dq_kernel` *(off by default)* | `k_attn.py` | SDPA backward | Recompute P from Q, K and the saved LSE; dK/dV per (key block, query head) as fp32 partials (12 programs per key block instead of 2 keeps the 40 SMs busy), dQ per (query block, head). |
| `gated_silu_fwd_kernel`, `gated_silu_bwd_kernel` | `k_mlp.py` | `act_fn(gate) * up` in fp32 + its autograd | Works on the fused `[gate \| up]` GEMM output; the backward overwrites it in place with `[d gate \| d up]`, so the fused weight-grad GEMM reads one buffer and no extra `[N, 17920]` tensor exists. |
| `lm_head_rows_kernel` | `k_head.py` | fp32 logits copies, `cross_entropy`, `log_softmax`, TRL `entropy_from_logits`, `argmax`, rewind's per-row CE | One program per token row of a 256-token logits chunk: pass 1 max + first-index argmax (torch's tie rule), pass 2 `Σe^{x−m}` and `Σe^{x−m}(x−m)` → loss, entropy, LSE; pass 3 (supervised rows) writes `1024·(softmax − onehot)` in place, which one cuBLAS GEMM turns into d(hidden). |

### How the step is organised (`solution/fast_qwen.py`)

* **One autograd node for the whole model** (`FusedCausalLM`): inputs are the 112 LoRA tensors,
  output the summed CE. The forward runs the 28 layers on the packed batch and keeps only each
  layer's fp32 input (optionally in pinned host memory); the backward re-runs each layer's
  forward from its checkpoint and back-propagates through it by hand (fp16 cuBLAS GEMMs with the
  dequantised weights, our kernels for everything else), so at most one layer's activations are
  alive. LoRA gradients follow autocast's fp16 GEMM semantics and land in the fp32 `.grad`.
* **Compute order per layer**: dequant (1 launch) → norm+dropout → LoRA-A GEMMs → fused QKV GEMM
  (bias via `addmm`, as bitsandbytes does) → LoRA-B GEMMs → QKV epilogue → attention (SDPA; in the
  backward recompute it records a small local autograd graph that is differentiated with
  `torch.autograd.grad`) → O GEMM →
  residual+norm → fused gate|up GEMM → gated SiLU → down GEMM.
* **Trainer side** (`solution/integration.py`): the trainer's `compute_loss` is replaced on the
  instance. It packs the batch, runs the step and records exactly what HF `Trainer`, TRL's
  `SFTTrainer` and Soup's `RewindTrainer` record (loss normalisation by `num_items_in_batch`,
  entropy over attention-mask tokens, `num_tokens`, token accuracy over supervised tokens,
  per-row rewind losses popped from the sampler FIFO). Anything else (eval, other loss types,
  extra inputs) goes to the original method.
* **Compilation** (`solution/precompile.py`): `apply()` immediately starts compiling all kernels in
  background threads with `JITFunction.warmup` (exact dtypes/constexprs; all runtime integers are
  `do_not_specialize`), while Soup imports, tokenises and loads the model. Triton's MLIR/LLVM/ptxas
  phases release the GIL, so most of the compile time is hidden. No cache is shipped; every run
  compiles from scratch.

## Correctness

* `solution/test_kernels.py` (run by the notebook, `pytest solution`): every kernel against an
  independent PyTorch / bitsandbytes reference, forward and backward (NF4 bit-exact incl. the
  streamed copy, RMSNorm vs fp64 autograd incl. both LoRA branches and mask regeneration, RoPE vs
  HF's formula with autograd, attention vs SDPA per sequence with autograd, gated SiLU, the head
  vs `cross_entropy`/`log_softmax`/`argmax`).
* `dev/check_fast_path.py` (`dev/kaggle_dev_check.ipynb`): the fast step vs the stock
  transformers+PEFT+bitsandbytes step on the real model and real rows (LoRA-B randomised so all
  112 gradients are non-zero): loss, entropy, accuracy, rewind row losses, gradient cosine and
  relative error; GradScaler-scale finiteness; offloaded variants give bit-identical gradients.
* Before we had a GPU, every kernel body was executed in a small NumPy emulator of
  `triton.language` (`dev/tl_emu.py`, `dev/test_emu.py`) against NumPy references.
* Differences from the base that remain: summation order inside GEMMs/reductions (fused QKV and
  gate|up GEMMs, a different M per GEMM), and the LoRA dropout masks come from a different random
  stream (Philox keyed by our own seeded generator instead of PyTorch's CUDA RNG). With
  `lora_B = 0` at init the first-step loss is identical; afterwards the run is a different
  dropout draw of the same training, well within checks 5–7.

## Rules compliance

| Rule | How |
|---|---|
| Own Triton kernels run and take ≥ 5 % | 12 kernels in `solution/`, all launched every step (dequant, norms, attention, MLP, head) |
| No ready-made kernel libraries | only torch, triton, transformers, peft, trl, bitsandbytes, soup-cli; no imports of the banned list |
| Same task | model, data, batch order (the trainer's sampler/collator are untouched), steps, LoRA r/alpha/dropout/targets, optimizer, schedule, loss; no Soup switch changed (`gradient_checkpointing`, `packing`, `use_liger`, ...): recomputation and padding removal are our own code |
| Same Soup outputs | the metrics and rewind records are computed as TRL/Soup do (see above) |
| Clean measurement | no CUDA graphs, no extra processes, no caches, same allocator; behaviour does not depend on being measured or tested |
| Soup's tests | patches are guarded (fake modules in the tests never qualify) |

## How to run

On Kaggle (what the jury does): import `templates/kaggle_notebook.ipynb`, accelerator **GPU T4
x2**, internet on, set `REPO_URL`/`COMMIT`, **Run All** (~10 min). It installs the pinned packages,
runs `pytest solution`, `harness/run_task.py` (base and ours) and `harness/check_rules.py`.

Quick developer check: `dev/kaggle_dev_check.ipynb` (unit tests, the end-to-end comparison and
one measured run of ours, ~8 min).

Locally on a Linux CUDA machine:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
python dev/check_fast_path.py
python harness/run_task.py --repeats 2 && python harness/check_rules.py
```

Switches are constants in `solution/config.py` (the fast path, head chunk size, host offloading of
checkpoints / 4-bit weights, background compilation).

## Limitations, what did not work, next steps

* The fast path is specific to this configuration (Qwen2-architecture, NF4 double-quant, LoRA on
  q/v, fp16, SFT, single GPU); anything else silently uses stock Soup.
* LoRA dropout masks are not bit-identical to PyTorch's stream, so the loss curve is not
  bit-identical to the base after step 1 (it is a different, equally valid dropout draw).
* Recomputation costs one extra forward of the Q|K|V, O and gate|up GEMMs (~35 % more GEMM FLOPs);
  we trade it for the memory win. GEMMs remain cuBLAS on dequantised fp16 weights; a fused NF4
  GEMM was not attempted.
* The load phase (transformers quantising on the GPU) sets a floor of ~1.5 GB for the peak:
  streaming the weights from host memory only pays off below that floor.
* **Our Triton flash attention on the T4.** `k_attn.py` (varlen, causal, GQA without K/V copies,
  FlashAttention-2 schedule) passes all correctness tests, but in the first measured run it took
  ~0.9 s of a 1.14 s step (forward ~10 ms per call where ~0.3 ms was expected) and compiled for
  3.6–6.8 s per kernel: Triton 3.6's sm_75 path (MMA m16n8k8, no cp.async, 64 KB shared memory) did
  not handle our 64×128 / 32×128 tiles well. The same run showed everything else of the step at
  ~0.23 s. We switched attention to PyTorch's SDPA (the kernel base Soup uses) on a padded view;
  the kernels stay in the repository behind `ATTN_IMPL = "triton"`.

## Sources and credits

Ideas (the code is ours): FlashAttention-2 (Tri Dao, 2023, arXiv:2307.08691) and Triton's
fused-attention tutorial for the attention schedule; the fused linear–cross-entropy idea of
Liger-Kernel and Apple's Cut Cross-Entropy (arXiv:2411.09009) for computing the head gradient in
the forward without full logits; NF4 / double quantisation from QLoRA (Dettmers et al., 2023) and
bitsandbytes' `kDequantizeBlockwise` for the exact format; Unsloth's practice of patching models
through one `apply()` and of offloading/recomputing activations; gradient checkpointing (Chen et
al., 2016). Data: AmanMussa/kazakh-instruction-v2 (MIT), model Qwen2.5-1.5B-Instruct (Apache-2.0).
