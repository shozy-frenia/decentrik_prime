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

# Tokens per chunk of the fused LM head (logits are only ever materialised for this many rows:
# CHUNK x 151936 x 2 bytes, 78 MB at 256).
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

# Keep the per-layer fp32 residual checkpoints (6 KB/token/layer) in a pinned host arena too,
# instead of on the GPU.
OFFLOAD_CHECKPOINTS = False

# Keep the repacked 4-bit weights (680 MB) in pinned host memory and stream each layer into one of
# two GPU slots one layer ahead (H2D on a side stream, overlapped with the previous layer's work).
OFFLOAD_WEIGHTS = False

# Compile every Triton kernel in background threads as soon as apply() runs, so compilation
# overlaps Soup's imports, data preparation and model loading.
PRECOMPILE = True
PRECOMPILE_THREADS = 1

# Keep the tied embedding / lm_head in fp16 (bit-exact: the checkpoint is loaded as fp16 on a T4
# and only upcast to fp32 by prepare_model_for_kbit_training).
FP16_EMBEDDING = True

# Print one line per phase (setup, first steps) to the training log.
VERBOSE = True
