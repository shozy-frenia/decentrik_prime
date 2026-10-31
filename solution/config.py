"""Switches of the solution. Plain constants: the same values in every run (measured or not).

Change them here and push; nothing reads environment variables.
"""

# Use the fast path (our own forward/backward with Triton kernels). False = stock Soup + example.
FAST_PATH = True

# Attention:
#   "varlen" = PyTorch's memory-efficient attention kernel (the one SDPA uses on a T4, i.e. what
#              base Soup runs) called on the packed tokens with cu_seqlens: no padding, no
#              scatter/gather. Checked against "sdpa" at start-up; falls back to it on any mismatch.
#   "sdpa"   = the same kernel through SDPA on a zero-padded [B, H, Tmax, 128] view of the batch.
#   "triton" = our own varlen GQA flash-attention kernels (k_attn.py): correct, but Triton 3.6 runs
#              tl.dot on the FMA units below sm_80 (no tensor cores on the T4), ~40x slower.
ATTN_IMPL = "varlen"

# Dequantise the next layer's weights on a side stream while the current layer's GEMMs run.
DEQUANT_OVERLAP = True

# ---------------------------------------------------------------------------- where things live
# v6: by default nothing of the model stays on the GPU between uses. Each of these falls back to
# the v5 placement (GPU-resident) if its pinned host buffer cannot be allocated.
#
# Keep the repacked 4-bit weights (680 MB) in pinned host memory; layer j is copied into GPU slot
# j % 2 two layers ahead of its dequantisation (forward and backward), on its own copy stream.
OFFLOAD_WEIGHTS = True
# ... except for the first RESIDENT_LAYERS layers, which stay on the GPU (24 MB each). The backward
# moves both weights and saved activations host -> GPU; resident layers take PCIe load off it.
RESIDENT_LAYERS = 0
#
# Keep the tied fp16 embedding / lm_head (467 MB) in pinned host memory. The input embedding is
# gathered on the CPU (N rows); the LM head streams the matrix through HEAD_RING GPU buffers of
# HEAD_VOCAB_CHUNK rows each (k_head.streamed_head).
OFFLOAD_EMBEDDING = True
#
# Keep the per-layer fp32 residual checkpoints (6 KB/token/layer) in a pinned host arena too.
OFFLOAD_CHECKPOINTS = True
#
# While the fast path's model is loaded (inside Soup's model setup): skip transformers'
# caching-allocator warm-up (it reserves the whole model size on the GPU up front), materialise the
# embedding on the CPU, and move every 4-bit weight to host memory right after bitsandbytes has
# quantised it on the GPU. The quantisation itself is bitsandbytes' own, on the GPU, as in the base.
LOAD_TO_HOST = True
# ... and convert the checkpoint's bf16 tensors to fp16 on the GPU instead of the CPU (bit-exact).
LOAD_CONVERT_ON_GPU = True

# ---------------------------------------------------------------------------- LM head
# "chunks" = v6 streamed one-pass head over vocabulary chunks (works with a resident embedding
#            too); "rows" = v5 head (token chunks against the whole vocabulary, embedding on GPU).
HEAD_IMPL = "chunks"
HEAD_VOCAB_CHUNK = 2048     # embedding rows per chunk (6.3 MB fp16); every width is a multiple of 16
HEAD_RING = 2               # GPU buffers the streamed embedding cycles through (v10: 2 x 6.3 MB, was 3 x 12.6 MB)
# v5 head: tokens per chunk (logits are materialised for this many rows: CHUNK x 151936 x 2 bytes).
HEAD_CHUNK = 256

# Scale applied to the fp16 head gradient (p - onehot) before the fp16 GEMM; undone in fp32.
HEAD_GRAD_SCALE = 1024.0

# What the backward gets from the forward instead of recomputing it. Each is copied to a pinned
# host arena on a copy stream while the forward computes the next layer, and copied back one
# layer ahead in the backward (the copy engines run alongside the kernels; GPU memory only holds
# the layers in flight):
#   SAVE_GU   the gate|up GEMM output (35 KB/token/layer): saves the largest recomputed GEMM;
#   SAVE_ATTN the post-RoPE QKV, the attention output and log-sum-exp, and the O GEMM output
#             (10 KB/token/layer): saves norm-1 -> QKV GEMM -> epilogue -> attention -> O GEMM.
# A batch with more tokens than HOST_CAPACITY (4 rows x 512 tokens is the task's maximum) is
# recomputed instead, with the same result.
SAVE_GU = True
SAVE_ATTN = True
HOST_CAPACITY = 2048

# Allocate the pinned host buffers (weights, embedding, activation arenas) in a background thread
# as soon as apply() runs.
EARLY_PINNED = True

# Unregister (unpin) our page-locked host buffers once the training loop is over, in a
# background thread while Soup saves and wraps up (hostmem.release_async).
EARLY_RELEASE = True

# Before training (in the background build, while Soup builds the trainer): run the LM head
# against an fp32 reference on a few random rows and one forward + backward of a small synthetic
# batch through every kernel and copy path (fast_qwen.FastQwen2.self_test). If the streamed
# placement fails it (an exception, a mismatch, a non-finite value), the run continues with the
# v5 placement (FALLBACK_RESIDENT), which is tested the same way; if that fails too, with stock
# Soup. Nothing of the self-test reaches the training state.
SELF_TEST = True
FALLBACK_RESIDENT = True

# Record the step's metrics (loss, entropy, accuracy, rewind row losses) after the backward has
# been queued instead of synchronising on them between the forward and the backward.
DEFER_METRICS = True

# Compile every Triton kernel in background threads as soon as apply() runs, so compilation
# overlaps Soup's imports, data preparation and model loading.
PRECOMPILE = True
PRECOMPILE_THREADS = 1

# Keep the tied embedding / lm_head in fp16 (bit-exact: the checkpoint is loaded as fp16 on a T4
# and only upcast to fp32 by prepare_model_for_kbit_training).
FP16_EMBEDDING = True

# Print one line per phase (setup, first steps) to the training log.
VERBOSE = True

# Sample the main thread's Python stack every 20 ms outside the training steps (from apply() to
# the first step, and from the end of the training loop to exit) and print where the time went
# (solution/profile_setup.py). Log only; it changes nothing the run computes.
SETUP_PROFILE = False

# Python's cyclic GC: a higher generation-0 threshold from apply() on (fewer full passes over the
# millions of objects Soup's imports create), and gc.freeze() at the first training step. Changes
# when garbage cycles are collected, nothing else (reference counting frees tensors as before).
GC_TUNE = True

# Experimental (A/B in dev/check_fast_path.py): run the layer GEMMs through cuBLASLt instead of
# cuBLAS (torch.backends.cuda.preferred_blas_library), set around our forward and backward only.
BLAS_LT = False
