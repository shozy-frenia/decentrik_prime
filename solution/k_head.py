# solution/k_head.py
import torch
import triton
import triton.language as tl


@triton.jit
def lm_head_rows_kernel(
    logits_ptr,          # [R, V] fp16, overwritten in-place with d_logits
    labels_ptr,          # [R] int64 or int32 (valid labels in [0, V))
    losses_ptr,          # [R] fp32 output per-token NLL
    entropies_ptr,       # [R] fp32 output per-token entropy
    correct_ptr,         # [R] fp32 or int32 output (1.0 if argmax == label else 0.0)
    grad_scale,          # fp32 scalar: loss_scale / total_supervised_tokens
    stride_r,            # int: row stride of logits_ptr (typically V)
    V: tl.constexpr,     # 151936 for Qwen2.5-1.5B
    BLOCK_V: tl.constexpr = 4096,
):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row * stride_r
    y = tl.load(labels_ptr + row).to(tl.int32)

    # Load target logit once before scanning the vocabulary
    target_logit = tl.load(row_ptr + y).to(tl.float32)

    # Pass 1: Online single-pass reduction for (m, d, h) + exact tie-aware accuracy check
    m_vec = tl.full([BLOCK_V], -float("inf"), dtype=tl.float32)
    d_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
    h_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
    beaten_vec = tl.zeros([BLOCK_V], dtype=tl.int32)

    for start_v in range(0, V, BLOCK_V):
        cols = start_v + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

        # First-index tie-breaking argmax check without tl.argmax:
        # Another token beats `y` iff x > target_logit, or (x == target_logit and col < y)
        is_better = mask & ((x > target_logit) | ((x == target_logit) & (cols < y)))
        beaten_vec = beaten_vec | is_better.to(tl.int32)

        m_new = tl.maximum(m_vec, x)
        alpha = tl.exp(m_vec - m_new)
        w = tl.where(mask, tl.exp(x - m_new), 0.0)
        d_vec = d_vec * alpha + w
        h_vec = h_vec * alpha + tl.where(mask, x * w, 0.0)
        m_vec = m_new

    # Cross-CTA reduction across BLOCK_V
    m_row = tl.max(m_vec, axis=0)
    alpha_row = tl.exp(m_vec - m_row)
    d_row = tl.sum(d_vec * alpha_row, axis=0)
    h_row = tl.sum(h_vec * alpha_row, axis=0)
    any_beaten = tl.max(beaten_vec, axis=0)

    lse = m_row + tl.log(d_row)
    loss_val = lse - target_logit
    entropy_val = lse - (h_row / d_row)
    acc_val = tl.where(any_beaten == 0, 1.0, 0.0)

    tl.store(losses_ptr + row, loss_val)
    tl.store(entropies_ptr + row, entropy_val)
    tl.store(correct_ptr + row, acc_val)

    # Pass 2: In-place backward d_logits = grad_scale * (softmax(x) - 1_{v == y})
    for start_v in range(0, V, BLOCK_V):
        cols = start_v + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(x - lse)
        grad = (p - tl.where(cols == y, 1.0, 0.0)) * grad_scale
        tl.store(row_ptr + cols, grad.to(tl.float16), mask=mask)
