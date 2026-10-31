# Soup Kernel Challenge — the whole QLoRA step streamed through a T4 from host memory

Our solution makes the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4 4-bit, LoRA r=16 on
`q_proj`/`v_proj`, batch 4, ≤512 tokens, 64 steps, Kaggle T4) **faster and far lighter** with
**our own Triton kernels** (9 launched in the default configuration, 14 in total) and our own
forward/backward of the model, plugged into Soup through `solution.apply()`. It trains the same
adapter: same model, data, batches, LoRA (dropout included), optimizer, schedule, loss, and the
same Soup outputs (loss, entropy, token accuracy, `num_tokens`, `rewind.jsonl`).

**v6 in one sentence:** between uses, the GPU holds no model weights at all — the 4-bit layers,
the fp16 embedding and every activation the backward needs live in pinned host memory and are
streamed through the GPU by the copy engines while the kernels compute, and the model never
touches the GPU in full even while it loads.

| Links | |
|---|---|
| Public Kaggle notebook (final run) | `<KAGGLE NOTEBOOK LINK>` |
| Final commit | `<COMMIT LINK>` |

## Results (Kaggle T4, `harness/run_task.py --repeats 2`, public eval rows)

| | Base Soup 0.77.0 | Ours (v6) | Ratio |
|---|---|---|---|
| Whole `soup train` command, s | `<base>` | `<ours>` | speedup `<s>`× |
| of which the 64 training steps, s | `<base>` | `<ours>` | |
| Peak GPU memory (nvidia-smi), MiB | `<base>` | `<ours>` | base/ours `<m>`× |
| Eval loss (`eval_adapter.py`) | `<base>` | `<ours>` | `<Δ>` % |
| Our kernels' share of step time (check 4) | – | `<share>` % | |

All 10 checks of `harness/check_rules.py`: `<PASS/FAIL>` (output in the notebook).

### How we got there (our solution alone, `run_task.py --only solution`, Kaggle T4)

Each row is a measured run, except where marked *pending*.

| Version | What changed | `soup train`, s | 64 steps, s | median step, s | peak, MiB | eval loss |
|---|---|---|---|---|---|---|
| v1 | fused kernels, packing, recompute, fused head, our Triton attention | 108.1 | 81.2 | 1.137 | 1915 | 2.0495 |
| v3 | attention → PyTorch's SDPA on a padded view | 55.95 | 31.63 | 0.445 | 1795 | 2.0496 |
| v4 | varlen attention (`cu_seqlens`, no padding), dequant on a side stream, launchers built ahead | 51.98 | 27.97 | 0.420 | 1815 | 2.0495 |
| v5 | gate\|up and attention activations kept in pinned host memory instead of recomputed | 51.83 | 26.22 | 0.393 | 1939 | 2.0495 |
| v6 | model streamed from host memory (weights, embedding, checkpoints), loaded off the GPU, streamed vocabulary-chunk LM head, no sync between forward and backward | *pending* | | | | |

v5 measured profile (per step, `--profile`, GPU time with overlaps): cuBLAS GEMMs ≈ 276 ms,
pinned↔GPU copies 79 + 74 ms (on the copy engines, overlapped), our dequant 33 ms (side stream),
attention 25 ms, our norms 18 ms, SwiGLU 13 ms, head 5 ms. In-process A/B of v5 on the same
batches: keeping activations instead of recomputing −99 ms/step; residual checkpoints on the host
−189 MiB for +11 ms/step.

## Where the memory was, and where it goes now

The memory score is the nvidia-smi peak over the *whole* run, so the load phase counts as much as
training. v5's 1939 MiB peak (1772 MiB reserved by PyTorch + ~167 MiB CUDA context) was:

| v5 GPU memory at the peak | MiB | v6 |
|---|---|---|
| 4-bit weights, repacked (28 layers × 23) | 649 | **pinned host memory**, streamed into 2 GPU slots of 23 MiB (`WeightPipe`) |
| tied fp16 embedding / lm_head | 445 | **pinned host memory**: input rows gathered on the CPU, the LM head streams the matrix through a ring of 3 × 12 MiB (`EmbeddingStream`) |
| fp32 residual checkpoints (28 × 6 KB per token) | ~240 at 1500 tokens | pinned host memory (v5's switch, now on) |
| LM head: 256-token logits chunk + gradient buffers | ~100 | N × 4096 logits chunk (~7 MiB at 900 tokens) |
| fp16 dequant workspace (one layer), LoRA + Adam, layer in flight | ~300 | unchanged |
| **load phase:** transformers' `caching_allocator_warmup` reserves the model size (~1.1 GB) before loading, then the whole model sits on the GPU | floor ~1.4 GB | warm-up skipped, embedding loaded on the CPU, each 4-bit weight moved to host the moment bitsandbytes has quantised it on the GPU: the GPU holds at most one fp16 weight (27 MB) while loading |

## The time budget of streaming (why it costs little)

T4 PCIe 3.0 ×16 measured on Kaggle: 11.5 GB/s host→GPU, 12.2 GB/s GPU→host, full duplex. Per
layer and per pass, the copy engines move:

* **forward:** activations out (51 KB per token: gate\|up 35, QKV 4, attention out 3, O 3,
  checkpoint 6) and the next-but-one layer's 4-bit weights in (24.3 MB) — different directions,
  both under the layer's compute;
* **head:** the embedding in (467 MB) in 12.6 MB chunks, one chunk ahead of the two GEMMs per chunk;
  the first three chunks are copied under the last layer;
* **backward:** activations and weights both host→GPU, `51 KB × N + 24.3 MB`: about 70 MB for a
  900-token batch (≈6 ms at 11.5 GB/s) against ≈5.5 ms of compute — the one place where the link,
  not the GPU, can set the pace for small batches. `RESIDENT_LAYERS = k` keeps k layers' 4-bit
  weights on the GPU (23 MiB each) and takes 2 × 24.3 MB per layer off the link; the dev check
  measures the trade-off.

Weights for the backward never cross the link at the two turns: the layer that ends the forward
starts the backward from the workspace, and the slots still hold the layer before it.

## The kernels (all in `solution/`, all ours)

| Kernel (`@triton.jit`) | File | Replaces | How it works |
|---|---|---|---|
| `head_chunk_kernel` (v6) | `k_head.py` | fp32 logits copies, `cross_entropy`, `log_softmax`, TRL `entropy_from_logits`, `argmax`, rewind's per-row CE, the logits gradient | One program per token row of one **vocabulary chunk** of logits `[N, 4096]` (one cuBLAS GEMM `Y·E_cᵀ`). Pass 1: chunk max + first-index argmax. Pass 2: `Σe^{x−m_c}` and `Σe^{x−m_c}(x−m_c)` merged into the row's running (max, sum, entropy sum, argmax, target logit) — an online softmax over the vocabulary, i.e. flash attention with the hidden states as queries and the embedding as keys *and* values; for rows with a target the same pass writes `P = 1024·e^{x−m_run}` in place (target column zeroed) and rescales the row's fp32 gradient accumulator by `e^{m_old−m_run}`. A cuBLAS GEMM with fp32 output then adds `P·E_c` to that accumulator in place. |
| `head_finalize_kernel` (v6) | `k_head.py` | — | Per row: `lse`, entropy `log s − t/s`, loss `lse − x_t`, accuracy hit, and `d(hidden) = acc/(1024 s) + (p_t−1)·E_t` written straight into token order. The target term is added in fp32 from the gathered row `E_t`, not through the fp16 `P` matrix: for a confident prediction the gradient is a small difference of two nearly equal vectors. |
| `nf4_layer_dequant_kernel` | `k_nf4.py` | 7× bitsandbytes `dequantize_4bit` per layer per pass | At setup the 7 NF4 weights of a layer are repacked into one buffer `[nibbles \| absmax codes \| 2nd-level scales \| offsets]` (in pinned host memory in v6). One launch per layer half (attention part, MLP part) decodes `fp16(NF4[nibble] · (code2[q]·absmax2 + offset))` for 8192 weights per program (`tl.join` interleaves the two nibbles of a byte), writing the fused Q\|K\|V, O, gate\|up, down matrices. Launched with FMA fusion off: bit-exact vs bitsandbytes. |
| `rms_fwd_kernel` | `k_norm.py` | Qwen2RMSNorm (fp32, ~6 kernels) + residual add + PEFT's two `lora_dropout` + autocast casts | One program per token: `x = x_prev + delta`, `rstd`, `y = w·(x·rstd)`, writes the fp32 residual (the layer checkpoint), fp16 `y` for the GEMMs and both dropout-masked LoRA inputs. Masks come from Philox (`tl.rand`) keyed by token×column, so backward regenerates them instead of storing them. |
| `rms_bwd_kernel` | `k_norm.py` | autograd of the above + dropout backward + residual-grad accumulation | `dx = rstd·(g − x̂·mean(g·x̂))`, `g = w·(dy + mask_q·dLoRA_q + mask_v·dLoRA_v)`, accumulated in place into the fp32 residual gradient. |
| `qkv_epilogue_fwd_kernel` | `k_rope.py` | PEFT's LoRA add for q/v, `apply_rotary_pos_emb` (`rotate_half`, `cat`), the fp16 casts for SDPA | In place on the fused QKV GEMM output: `q += 2·lora_q`, rotate q and k with the model's cos/sin, `v += 2·lora_v`. |
| `qkv_epilogue_bwd_kernel` | `k_rope.py` | RoPE backward + GQA reduction | Transposed rotation, summing the 6 per-query-head dK/dV partials of each KV head, writes the fp16 grad of the fused QKV output. |
| `gated_silu_fwd_kernel`, `gated_silu_bwd_kernel` | `k_mlp.py` | `act_fn(gate) * up` in fp32 + its autograd | Works on the fused `[gate \| up]` GEMM output; the backward overwrites it in place with `[d gate \| d up]`, so the fused weight-grad GEMM reads one buffer and no extra `[N, 17920]` tensor exists. |
| `lm_head_rows_kernel` *(v5 head, `HEAD_IMPL="rows"`)* | `k_head.py` | as `head_chunk_kernel` | 256-token chunks against the whole vocabulary: max/argmax, softmax sums, in-place `1024·(softmax − onehot)`, one GEMM for d(hidden). |
| `attn_fwd_kernel`, `attn_bwd_*` *(off: `ATTN_IMPL="triton"`)* | `k_attn.py` | SDPA | Varlen causal GQA FlashAttention-2 over a (sequence, block) table. Correct, but Triton 3.6 runs `tl.dot` on the FMA units below sm_80 (see *What did not work*). |

## How the step is organised

* **Loading (`solution/integration.py`, `_patch_loading`)** — only while Soup loads the model of a
  run the fast path implements (a context variable is set around Soup's `_setup_transformers`):
  `transformers.modeling_utils.caching_allocator_warmup` is skipped, `core_model_loading.get_device`
  places `model.embed_tokens.weight` on the CPU (it is materialised there as fp16, the same IEEE
  bf16→fp16 rounding as on the GPU), and after `core_model_loading.set_param_for_module` has set a
  4-bit weight that bitsandbytes has just quantised **on the GPU** (its own kernel, as in the
  base), the packed data and quantisation state are moved to host memory. `dev/check_fast_path.py`
  loads the model both ways and checks all 196 weights, their quantisation states and the
  embedding are bit-identical.
* **Placement (`FastQwen2`)** — validation first (nothing is modified until every check passed),
  LoRA parameters to the GPU (PEFT created them next to the base weights, i.e. on the host;
  this happens before TRL casts them and before the optimizer exists), then in a background thread
  while Soup tokenises the data and builds the trainer: the 4-bit layers are repacked into one
  pinned buffer (`hostmem`, allocated by apply() in the background during Soup's imports), the
  embedding is copied into pinned memory, tables are prepared. A model that fails validation →
  every tensor is put back on the GPU and stock Soup runs.
* **Start-up self-test and fallbacks (v6, `FastQwen2.self_test`)** — still in the background
  build, before training: (1) the LM head on 8 random rows (one without a target) against an
  independent fp32 reference written with plain torch ops (online softmax over the vocabulary,
  the embedding copied chunk by chunk, not through the ring): loss, lse, entropy and d(hidden)
  must agree to fp16-logit precision; (2) one forward + backward of a small synthetic batch through
  every kernel, stream and copy path of the configured placement: finite loss, statistics and
  all 112 LoRA gradients. It uses its own random generators and restores the dropout generator,
  step counter and gradients, so nothing reaches training. If the streamed placement fails it (an
  exception, a Triton compile error, a mismatch, a non-finite value), the model is rebuilt in the
  **v5 placement** (weights and fp16 embedding on the GPU, v5 head), which is tested the same way;
  if that fails too, stock Soup runs. If Soup's model setup itself fails with our host-loading
  hooks active, it is run once more without them. The log line `fast path ready … self-test: …`
  says which placement trains.
* **One autograd node for the whole model** (`FusedCausalLM`): inputs are the 112 LoRA tensors,
  output the summed CE. Forward: 28 layers on the packed batch (the real tokens of the 4 rows
  concatenated, `cu_seqlens` attention, no padding); backward by hand (fp16 cuBLAS GEMMs on the
  dequantised weights, our kernels for everything else). LoRA gradients follow autocast's fp16
  GEMM semantics and land in the fp32 `.grad`.
* **Weight pipe (`k_nf4.WeightPipe`)** — layer j's packed bytes go to GPU slot j % 2 (consecutive
  layers of a walk never share a slot) on their own copy stream, two layers ahead: issued right
  after both halves of the slot's previous layer have been queued for dequantisation, ordered after
  everything already queued on the main and dequantisation streams. The dequantisation stream
  waits for the slot's copy, dequantises the next layer's attention / MLP half into the fp16
  workspace while the current layer's GEMMs run (v4), and the layer at each turn is reused.
  `dev/test_ws_schedule.py` checks, on a model of CUDA streams and events, that every dequant
  reads its slot after the right layer's copy and that no copy into a slot is unordered with a
  read of it (in 8 configurations; removing the ordering is caught: 82 races).
* **Activations through host memory** (`_ActStore`): per layer the forward queues GPU→host copies of
  the fp32 residual, gate\|up output, post-RoPE QKV, attention output and O output into pinned
  arenas; the backward queues the copies back one layer ahead into the blocks the previous layer
  just released. Ordering by events only (`dev/test_act_store.py`).
* **LM head (`k_head.streamed_head`)** — see the kernel table; the embedding chunks arrive on their
  own copy stream into a 3-buffer ring.
* **Trainer side** — the trainer's `compute_loss` is replaced on the instance; it packs the batch,
  runs the step and records exactly what HF `Trainer`, TRL's `SFTTrainer` and Soup's `RewindTrainer`
  record (loss normalisation by `num_items_in_batch`, entropy over attention-mask tokens,
  `num_tokens`, token accuracy over supervised tokens, per-row rewind losses popped from the
  sampler FIFO). v6 copies the per-token statistics to the host asynchronously and records them at
  the end of the step's backward (and in any case before the trainer logs), so the CPU no longer
  waits for the forward before queuing the backward.
* **Compilation** (`solution/precompile.py`): `apply()` starts compiling every kernel in a
  background thread with `JITFunction.warmup` (exact dtypes/constexprs, runtime integers
  `do_not_specialize`) while Soup imports, tokenises and loads; Triton's MLIR passes, LLVM and
  ptxas release the GIL. No cache is shipped; every run compiles from scratch.

## Correctness

* `solution/test_kernels.py` (`pytest solution`, run by the notebook): every kernel against an
  independent PyTorch / bitsandbytes reference — NF4 bit-exact incl. the split dequant and the weight
  pipe walking three layers forward and back (with and without a resident layer), RMSNorm vs fp64
  autograd, RoPE vs HF's formula, attention vs SDPA, gated SiLU, the v5 head, and the v6 streamed
  head (resident and streamed embedding) vs fp64 torch on the same fp16 logits **and** vs the v5
  head, plus the fp32-accumulating GEMM.
* `dev/test_emu.py` (CPU): every kernel body executed in a NumPy emulator of `triton.language`
  (`dev/tl_emu.py`) against NumPy references, incl. the vocabulary-chunk head (cross-chunk argmax
  ties, rows without targets, a single chunk, and confident predictions where the gradient's
  precision is checked against what the base computes for the same rows).
* `dev/test_ws_schedule.py`, `dev/test_act_store.py` (CPU): the dequant schedule, the weight pipe
  and the activation store driven through a model of streams, events and the caching allocator;
  each catches a mutated (unordered) version.
* `dev/check_fast_path.py` (`dev/kaggle_dev_check.ipynb`): the fast step vs the stock
  transformers+PEFT+bitsandbytes step on the real model and rows (loss, entropy, accuracy, rewind
  row losses, all 112 gradients); the hooked load vs a normal load (bit-exact); the whole
  integration path `soup train` takes (hooked load → patched `prepare_model_for_kbit_training` →
  LoRA created on the host → `FastQwen2`) against the runner built from a normal load, then the
  v5-placement fallback rebuilt from that streamed state; the self-test's result and time; the
  placement variants (time, peak, gradient agreement); PCIe both ways; where the GPU waits.
* Static checks: `ruff --select F` (pyflakes rules) is clean on `solution/` and `dev/`; it caught a
  name used out of scope in the v6 build path during this release, which the background build
  would have turned into a silent fallback to stock Soup.
* The head kernels never re-read a logit they may already have overwritten: the target logit is
  picked out of the first pass's loads (one lane per row holds it) instead of a scalar re-load,
  and the cross-warp reductions after that pass order every read before the in-place writes of
  the next one (the v5 head kernel got the same change).
* Differences from the base that remain: summation order inside GEMMs/reductions (fused QKV and
  gate\|up GEMMs, a different M per GEMM, vocabulary chunks in the head), and the LoRA dropout masks
  come from a different random stream (Philox keyed by our own seeded generator instead of
  PyTorch's CUDA RNG). With `lora_B = 0` at init the first-step loss is identical; afterwards the
  run is a different dropout draw of the same training, well within checks 5–7.

## Rules compliance

| Rule | How |
|---|---|
| Own Triton kernels run and take ≥ 5 % | 9 kernels launched every step (dequant, norms, QKV epilogue, SwiGLU, head); the v5 head and the Triton attention are off by default |
| No ready-made kernel libraries | only torch, triton, transformers, peft, trl, bitsandbytes, soup-cli; nothing from the banned list is imported (checked by `check_rules.py`'s scan) |
| Same task | model, data, batch order (the trainer's sampler/collator are untouched), steps, LoRA r/alpha/dropout/targets and init, optimizer, schedule, loss. The model is the same tensors: bitsandbytes quantises every weight on the GPU exactly as in the base; we only change where the results are stored (host memory), which the dev check verifies bit for bit |
| Same Soup outputs | the metrics and rewind records are computed as TRL/Soup do and recorded within the same step, before the trainer logs |
| Soup's built-in modes | untouched: `gradient_checkpointing`, `activation_offloading`, `stream_layers`, `packing`, `use_cut_ce`, `fast_lora` stay off; the offloading, streaming, recomputation and padding removal are our own code inside our forward/backward |
| Clean measurement | no CUDA graphs, no extra processes (background *threads* of the measured process only: compilation, pinned allocation, the model repack and the start-up self-test), no caches, the harness's allocator settings unchanged; behaviour does not depend on being measured or tested (the self-test and its fallbacks run identically in every run) |
| Soup's tests | patches are guarded (fake modules in the tests never qualify), loading patches are pass-through unless the fast plan is active |

## How to run

On Kaggle (what the jury does): import `templates/kaggle_notebook.ipynb`, accelerator **GPU T4
x2**, internet on, set `REPO_URL`/`COMMIT`, **Run All** (~10 min). It installs the pinned packages,
runs `pytest solution`, `harness/run_task.py` (base and ours) and `harness/check_rules.py`.

Developer check: `dev/kaggle_dev_check.ipynb` (CPU model tests, unit tests, the end-to-end
comparison with the placement variants, one measured run of ours, ~12 min).

Locally on a Linux CUDA machine:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python dev/test_emu.py && python dev/test_ws_schedule.py && python dev/test_act_store.py   # CPU
python -m pytest solution -q
python dev/check_fast_path.py
python harness/run_task.py --repeats 2 && python harness/check_rules.py
```

Switches are constants in `solution/config.py`: what lives in host memory (`OFFLOAD_WEIGHTS`,
`RESIDENT_LAYERS`, `OFFLOAD_EMBEDDING`, `OFFLOAD_CHECKPOINTS`, `LOAD_TO_HOST`), the head
(`HEAD_IMPL`, `HEAD_VOCAB_CHUNK`, `HEAD_RING`), which activations are kept, background
compilation/allocation, deferred metrics, the start-up self-test (`SELF_TEST`,
`FALLBACK_RESIDENT`). Setting `OFFLOAD_WEIGHTS`, `OFFLOAD_EMBEDDING`, `OFFLOAD_CHECKPOINTS` and
`LOAD_TO_HOST` to False and `HEAD_IMPL="rows"` gives the measured v5 layout.

## Limitations, what did not work, next steps

* The fast path is specific to this configuration (Qwen2-architecture, NF4 double-quant, LoRA on
  q/v, fp16, SFT, single GPU); anything else silently uses stock Soup.
* Streaming trades GPU memory for PCIe traffic (~3 GB host→GPU and ~1.2 GB GPU→host per
  900-token step). The forward and the head fit under the compute; the backward moves activations
  *and* weights in the same direction and can be link-bound for small batches — on a slower or
  shared PCIe link (another tenant, a ×8 slot) the step slows down rather than failing.
  `RESIDENT_LAYERS` trades memory back for time.
* Pinned host memory: ~4 GB (weights 0.68, embedding 0.47, activation arenas 2.9 for the task's
  maximum batch of 2048 tokens). Allocation is in the background; releasing it at exit is part of
  the measured time.
* LoRA dropout masks are not bit-identical to PyTorch's stream, so the loss curve is not
  bit-identical to the base after step 1 (it is a different, equally valid dropout draw).
* **v6 has not been measured yet.** Every v6 number in this README is an estimate from v5's
  measurements and the PCIe budget above, not a result; the v6 row of the table stays *pending*
  until `dev/kaggle_dev_check.ipynb` and the full notebook have run on a T4. The self-test guards
  against crashes and gross numerical errors of the new code (it falls back to the measured v5
  placement), not against small ones; those are what the dev check's comparisons are for.
* **Our Triton flash attention on the T4.** `k_attn.py` (varlen, causal, GQA without K/V copies,
  FlashAttention-2 schedule) passes all correctness tests, but in the first measured run it took
  ~0.9 s of a 1.14 s step. In Triton 3.6 `AccelerateMatmul` maps `tl.dot` to tensor cores (MMA)
  only for compute capability ≥ 8.0; for sm_75 it emits *"Dot op using MMA for compute capability
  75 has been deprecated. It falls back to the FMA path"*, so every `tl.dot` runs on the fp32 FMA
  units (8 TFLOP/s peak instead of 65). That rules out Triton GEMMs of any kind on this GPU (a fused
  NF4 GEMM, a Triton LM-head GEMM): our kernels do the memory-bound work and every matmul stays on
  tensor cores (cuBLAS, and PyTorch's memory-efficient attention kernel called varlen).
* Next: compress what crosses the link (the backward's weights could be fetched in the forward's
  idle host→GPU direction into a small cache), chunk the MLP over tokens to shrink the largest
  transient (the gate\|up output, 35 KB per token).

## Sources and credits

Ideas (the code is ours): FlashAttention / FlashAttention-2 (Tri Dao et al., 2022/2023,
arXiv:2205.14135, arXiv:2307.08691) for the online softmax used both in attention and over the
vocabulary, and Triton's fused-attention tutorial for the attention schedule; the fused
linear–cross-entropy idea of Liger-Kernel and Apple's Cut Cross-Entropy (arXiv:2411.09009) for
computing the head gradient in the forward without full logits; NF4 / double quantisation from
QLoRA (Dettmers et al., 2023) and bitsandbytes' `kDequantizeBlockwise` for the exact format;
ZeRO-Offload/ZeRO-Infinity (Ren et al., 2021; Rajbhandari et al., 2021) and Unsloth for offloading
weights and activations to host memory; gradient checkpointing (Chen et al., 2016). Data:
AmanMussa/kazakh-instruction-v2 (MIT), model Qwen2.5-1.5B-Instruct (Apache-2.0).
