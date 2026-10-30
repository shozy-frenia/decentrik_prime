# Submission: <team name>

Copy this file to the root of your repository as `SUBMISSION.md` and fill it in.

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_layer_dequant_kernel` | `solution/k_nf4.py` | 7x bitsandbytes dequantize per layer and pass | % |
| `rms_fwd_kernel`, `rms_bwd_kernel` | `solution/k_norm.py` | Qwen2RMSNorm + residual add + LoRA dropout (fwd/bwd) | % |
| `qkv_epilogue_fwd_kernel`, `qkv_epilogue_bwd_kernel` | `solution/k_rope.py` | LoRA add + RoPE + GQA reduction (fwd/bwd) | % |
| `attn_fwd_kernel`, `attn_bwd_delta_kernel`, `attn_bwd_dkdv_kernel`, `attn_bwd_dq_kernel` | `solution/k_attn.py` | SDPA on padded, repeat_kv-expanded heads (fwd/bwd) | % |
| `gated_silu_fwd_kernel`, `gated_silu_bwd_kernel` | `solution/k_mlp.py` | silu(gate)*up in fp32 (fwd/bwd) | % |
| `lm_head_rows_kernel` | `solution/k_head.py` | full fp32 logits, CE, log_softmax, TRL entropy, argmax, rewind CE | % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
FlashAttention-2 (https://arxiv.org/abs/2307.08691) and the Triton fused-attention tutorial
(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html); fused linear
cross-entropy (Liger-Kernel https://github.com/linkedin/Liger-Kernel, Cut Cross-Entropy
https://arxiv.org/abs/2411.09009); NF4 format from bitsandbytes `kDequantizeBlockwise`
(https://github.com/bitsandbytes-foundation/bitsandbytes) and QLoRA (https://arxiv.org/abs/2305.14314).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
What does not work yet, where the speedup stops, what you would do next.
