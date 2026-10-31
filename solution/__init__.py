"""Soup Kernel Challenge solution: a padding-free, fused-kernel QLoRA training step for the 4-bit
Qwen2 run that keeps the whole model in pinned host memory and streams it through the GPU,
plugged into Soup through apply().

See README.md for what the kernels do and why. Files:
    k_nf4.py      NF4 double-quant dequantisation of a layer into fused QKV / gate|up buffers, and
                  the weight pipe that streams repacked layers from host memory into two GPU slots
    k_norm.py     residual add + RMSNorm + both LoRA dropout masks, forward and backward
    k_rope.py     QKV epilogue: LoRA add + RoPE, forward and backward (with GQA dk/dv reduction)
    k_attn.py     variable-length causal GQA flash attention, forward and backward
    k_mlp.py      gated SiLU on the fused gate|up buffer, forward and in-place backward
    k_head.py     LM head: CE, entropy, argmax accuracy and the logit gradient without full logits,
                  streamed over vocabulary chunks (the embedding stays in host memory)
    fast_qwen.py  the model's forward/backward (one autograd node), activation offloading
    integration.py  the patches into transformers' loading, PEFT and Soup, the replacement compute_loss
    hostmem.py    pinned host buffers allocated in the background as soon as apply() runs
    precompile.py   background compilation of the kernels while Soup loads
    profile_setup.py  sampling profile of the main thread outside the training steps (log only)
    config.py     switches
"""



def apply():
    from . import config, integration, precompile, profile_setup

    profile_setup.start()        # where the time outside the training steps goes (log only)
    if config.FAST_PATH and config.CUDA_STACK_BYTES:
        from . import cudactx
        integration._CTX["stack"] = cudactx.shrink_stack(int(config.CUDA_STACK_BYTES))
    if config.FAST_PATH and config.CUBLAS_SMALL_WORKSPACE:
        import os
        # read by PyTorch when it creates the first cuBLAS workspace (no GEMM has run yet)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    if config.GC_TUNE:
        import gc
        # fewer cyclic-GC passes while Soup imports its stack and builds the run (every pass of
        # the oldest generation walks millions of objects); frozen at the first training step
        gc.set_threshold(max(gc.get_threshold()[0], 50000), *gc.get_threshold()[1:])
    precompile.start()           # compile our kernels while Soup imports, tokenises and loads
    if config.FAST_PATH and config.EARLY_PINNED:
        try:
            from . import hostmem
            from .fast_qwen import pinned_specs

            hostmem.start(pinned_specs())    # ~4 GB of pinned host buffers, in the background
        except Exception:  # noqa: BLE001 - FastQwen2 allocates them itself later
            pass
    from . import distmap
    distmap.share()              # until the first training step (see distmap.py)
    integration.install()        # fast path: model load to host, fused compute_loss in train()
    if not config.FAST_PATH:
        from .swiglu import patch_qwen2_mlp
        patch_qwen2_mlp()        # the example (stock path only); its kernels never count
