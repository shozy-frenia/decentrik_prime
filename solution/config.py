"""Switches of the solution. Plain constants: the same values in every run (measured or not).

Change them here and push; nothing reads environment variables.
"""

# Use the fast path (our own forward/backward with Triton kernels). False = stock Soup + example.
FAST_PATH = True

# Tokens per chunk of the fused LM head (logits are only ever materialised for this many rows:
# CHUNK x 151936 x 2 bytes, 78 MB at 256).
HEAD_CHUNK = 256

# Scale applied to the fp16 head gradient (p - onehot) before the fp16 GEMM; undone in fp32.
HEAD_GRAD_SCALE = 1024.0

# Keep the per-layer residual checkpoints in pinned host memory instead of on the GPU
# (copied out on a side stream during forward, prefetched one layer ahead during backward).
OFFLOAD_CHECKPOINTS = False

# Compile every Triton kernel in background threads as soon as apply() runs, so compilation
# overlaps Soup's imports, data preparation and model loading.
PRECOMPILE = True
PRECOMPILE_THREADS = 2

# Keep the tied embedding / lm_head in fp16 (bit-exact: the checkpoint is loaded as fp16 on a T4
# and only upcast to fp32 by prepare_model_for_kbit_training).
FP16_EMBEDDING = True

# Print one line per phase (setup, first steps) to the training log.
VERBOSE = True
