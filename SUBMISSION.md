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
| `head_chunk_kernel`, `head_finalize_kernel` | `solution/k_head.py` | full fp32 logits, CE, log_softmax, TRL entropy, argmax, rewind CE, the logits gradient (one-pass online softmax over vocabulary chunks; the embedding streams from host memory) | % |
| `nf4_layer_dequant_kernel` | `solution/k_nf4.py` | 7x bitsandbytes dequantize per layer and pass (reads the layer from its streamed GPU slot) | % |
| `rms_fwd_kernel`, `rms_bwd_kernel` | `solution/k_norm.py` | Qwen2RMSNorm + residual add + LoRA dropout (fwd/bwd) | % |
| `qkv_epilogue_fwd_kernel`, `qkv_epilogue_bwd_kernel` | `solution/k_rope.py` | LoRA add + RoPE + GQA reduction (fwd/bwd) | % |
| `gated_silu_fwd_kernel`, `gated_silu_bwd_kernel` | `solution/k_mlp.py` | silu(gate)*up in fp32 (fwd/bwd) | % |
| `lm_head_rows_kernel` (v5 head, off by default) | `solution/k_head.py` | as above, token chunks with the embedding on the GPU | – |
| `attn_fwd_kernel`, `attn_bwd_delta_kernel`, `attn_bwd_dkdv_kernel`, `attn_bwd_dq_kernel` (off by default) | `solution/k_attn.py` | SDPA on padded, repeat_kv-expanded heads (fwd/bwd); FMA-only on sm_75 in Triton 3.6 | – |

Besides the kernels: the model is kept in pinned host memory and streamed through the GPU (4-bit
layers through two slots, the embedding through a ring of three chunks, activations through host
arenas), and it is loaded without ever residing on the GPU (bitsandbytes still quantises every
weight on the GPU; the results are moved to host memory as they are produced). Before training, a
self-test checks the streamed LM head against an fp32 reference and runs one synthetic step
through every kernel and copy path; on any failure the run falls back to the measured v5
placement (model on the GPU), then to stock Soup.

Ideas taken from other kernels and systems, with links (we wrote the code ourselves):
FlashAttention-2 (https://arxiv.org/abs/2307.08691) and the Triton fused-attention tutorial
(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) - also the
online softmax our head runs over the vocabulary; fused linear cross-entropy (Liger-Kernel
https://github.com/linkedin/Liger-Kernel, Cut Cross-Entropy https://arxiv.org/abs/2411.09009);
NF4 format from bitsandbytes `kDequantizeBlockwise`
(https://github.com/bitsandbytes-foundation/bitsandbytes) and QLoRA (https://arxiv.org/abs/2305.14314);
host-memory offloading from ZeRO-Offload (https://arxiv.org/abs/2101.06840) and ZeRO-Infinity
(https://arxiv.org/abs/2104.07857).

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
- Specific to this run's configuration (Qwen2, NF4 double quant, LoRA on q/v, fp16, SFT, one GPU);
  anything else uses stock Soup.
- Streaming moves ~3 GB host->GPU per 900-token step; the backward can be PCIe-bound on small
  batches (`RESIDENT_LAYERS` trades GPU memory back for time). ~4 GB of pinned host memory.
- Triton 3.6 has no tensor-core `tl.dot` on sm_75, so all matmuls stay in cuBLAS / PyTorch's
  memory-efficient attention; our kernels do the memory-bound work.
- LoRA dropout masks come from our Philox stream, not PyTorch's: same distribution, not the same
  draw, so the loss curve matches the base within tolerance but not bit for bit after step 1.
- v6/v7 (the streamed placement, the trimmed backward) have not been measured on a T4 yet; fill
  the tables above from the notebook run, not from the README's estimates.
