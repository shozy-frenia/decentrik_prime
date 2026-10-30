"""Soup Kernel Challenge solution: a padding-free, recomputing, fused-kernel training step for the
4-bit QLoRA Qwen2 run, plugged into Soup through apply().

See README.md for what the kernels do and why. Files:
    k_nf4.py      NF4 double-quant dequantisation of a whole layer into fused QKV / gate|up buffers
    k_norm.py     residual add + RMSNorm + both LoRA dropout masks, forward and backward
    k_rope.py     QKV epilogue: LoRA add + RoPE, forward and backward (with GQA dk/dv reduction)
    k_attn.py     variable-length causal GQA flash attention, forward and backward
    k_mlp.py      gated SiLU on the fused gate|up buffer, forward and in-place backward
    k_head.py     LM head: CE, entropy, argmax accuracy and the logit gradient without full logits
    fast_qwen.py  the model's forward/backward with per-layer recomputation (one autograd node)
    integration.py  the patches into Soup / PEFT and the replacement compute_loss
    precompile.py   background compilation of the kernels while Soup loads
    config.py     switches
"""



def apply():
    from . import config, integration, precompile

    precompile.start()           # compile our kernels while Soup imports, tokenises and loads
    integration.install()        # fast path: model prep after load, fused compute_loss in train()
    if not config.FAST_PATH:
        from .swiglu import patch_qwen2_mlp
        patch_qwen2_mlp()        # the example (stock path only); its kernels never count
