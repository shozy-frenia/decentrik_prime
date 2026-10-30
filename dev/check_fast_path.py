"""End-to-end check of the fast path against the stock transformers + PEFT + bitsandbytes path on
the real model and real data (a T4 with the challenge's packages; ~2 minutes).

    python dev/check_fast_path.py            # after python task/prepare_data.py

Builds the model the way Soup does (fp16 load, NF4 double quant, prepare_model_for_kbit_training,
LoRA r=16 alpha=32 on q_proj/v_proj), gives lora_B random values so that every LoRA gradient is
non-zero, and compares on the same right-padded batch:
    loss (sum CE / supervised tokens), TRL entropy, TRL token accuracy, rewind per-row losses,
    and the gradients of all 112 LoRA tensors (relative L2 error and cosine),
with LoRA dropout 0 (so both sides see the same network). Then it times a few steps of each
path, measures peak memory, and checks the host-offloaded checkpoints give identical grads.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build(dropout):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map={"": 0},
                                                 dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, target_modules=["q_proj", "v_proj"],
                     bias="none", task_type="CAUSAL_LM")
    torch.manual_seed(0)
    model = get_peft_model(model, cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02)
    model.train()
    return tok, model


def make_batch(tok, rows, max_len=512):
    seqs = []
    for row in rows:
        extra = row.get("input") or ""
        user = f"{row['instruction']}\n{extra}".strip() if extra else row["instruction"]
        msgs = [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": row["output"]}],
                                       tokenize=True, return_dict=False)[:max_len]
        lab = [-100] * min(len(prompt), len(full)) + full[len(prompt):]
        seqs.append((full, lab[:len(full)]))
    T = max(len(s[0]) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    labels = torch.full((len(seqs), T), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (f, l) in enumerate(seqs):
        ids[i, :len(f)] = torch.tensor(f)
        labels[i, :len(l)] = torch.tensor(l)
        mask[i, :len(f)] = 1
    return ids.cuda(), mask.cuda(), labels.cuda()


def lora_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad]


def stock_step(model, ids, mask, labels):
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits.float()
    shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
    valid = shift_labels != -100
    n_items = valid.sum()
    ce = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, logits.shape[-1]), shift_labels.reshape(-1),
                                           ignore_index=-100, reduction="none").view(shift_labels.shape)
    loss = ce.sum() / n_items
    with torch.no_grad():
        ents = []
        for chunk in logits.reshape(-1, logits.shape[-1]).split(128):
            lp = torch.log_softmax(chunk, -1)
            ents.append(-(lp.exp() * lp).sum(-1))
        ent = torch.cat(ents).view(mask.shape)
        entropy = float((ent * mask).sum() / mask.sum())
        acc = float(((shift_logits.argmax(-1) == shift_labels) & valid).sum() / n_items)
        row_loss = [float((ce[i] * valid[i]).sum() / valid[i].sum().clamp(min=1)) for i in range(ids.shape[0])]
    del logits, shift_logits, out
    loss.backward()
    return dict(loss=float(loss), entropy=entropy, accuracy=acc, row_loss=row_loss)


def fast_step(runner, ids, mask, labels):
    from solution.fast_qwen import batch_metrics, run_step

    batch = runner.pack(ids, mask, labels)
    loss_sum, stats = run_step(runner, batch)
    loss = loss_sum / batch.n_lab
    m = batch_metrics(batch, stats.detach().cpu())
    loss.backward()
    return dict(loss=float(loss), entropy=m["entropy"], accuracy=m["accuracy"], row_loss=m["row_loss"],
                n=batch.n, n_pad=ids.numel(), n_lab=batch.n_lab)


def grads_of(model):
    return [p.grad.detach().float().clone() for p in lora_params(model)]


def compare_grads(a, b):
    worst_rel, worst_cos = 0.0, 1.0
    for x, y in zip(a, b):
        rel = float((x - y).norm() / y.norm().clamp(min=1e-30))
        cos = float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))
        worst_rel, worst_cos = max(worst_rel, rel), min(worst_cos, cos)
    return worst_rel, worst_cos


def timed(fn, reps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    from solution import config, precompile

    report = {}
    rows = [json.loads(line) for line in open(REPO / "data" / "train.jsonl", encoding="utf-8")]
    tok, model = build(dropout=0.0)
    params = lora_params(model)
    batches = [make_batch(tok, rows[i:i + 4]) for i in (0, 4, 8, 40)]
    ids, mask, labels = batches[0]

    torch.cuda.reset_peak_memory_stats()
    ref = stock_step(model, ids, mask, labels)
    ref_grads = grads_of(model)
    report["stock_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    for p in params:
        p.grad = None

    from solution.fast_qwen import FastQwen2
    precompile.start()
    precompile.wait()
    runner = FastQwen2(model, seed=1)
    torch.cuda.reset_peak_memory_stats()
    got = fast_step(runner, ids, mask, labels)
    fast_grads = grads_of(model)
    report["fast_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    rel, cos = compare_grads(fast_grads, ref_grads)
    report["stock"] = ref
    report["fast"] = got
    report["loss_rel"] = abs(got["loss"] - ref["loss"]) / ref["loss"]
    report["entropy_rel"] = abs(got["entropy"] - ref["entropy"]) / ref["entropy"]
    report["accuracy_abs"] = abs(got["accuracy"] - ref["accuracy"])
    report["row_loss_rel_max"] = max(abs(a - b) / b for a, b in zip(got["row_loss"], ref["row_loss"]))
    report["grad_rel_l2_worst"] = rel
    report["grad_cos_worst"] = cos

    # loss scaling like the fp16 GradScaler: grads stay finite
    for p in params:
        p.grad = None
    from solution.fast_qwen import run_step
    batch = runner.pack(ids, mask, labels)
    loss_sum, _ = run_step(runner, batch)
    (loss_sum / batch.n_lab * 65536.0).backward()
    report["scaled_grads_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in params)

    # host-offloaded checkpoints: identical gradients
    for p in params:
        p.grad = None
    config.OFFLOAD_CHECKPOINTS = True
    torch.cuda.reset_peak_memory_stats()
    fast_step(runner, ids, mask, labels)
    report["offload_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    off_grads = grads_of(model)
    report["offload_grads_identical"] = all(torch.equal(a, b) for a, b in zip(off_grads, fast_grads))
    config.OFFLOAD_CHECKPOINTS = False

    # timing over a few different batches (forward + backward)
    def run_fast():
        for b in batches:
            for p in params:
                p.grad = None
            fast_step(runner, *b)

    def run_stock():
        for b in batches:
            for p in params:
                p.grad = None
            stock_step(model, *b)

    run_fast()
    report["fast_s_per_step"] = timed(run_fast, 2) / len(batches)
    config.OFFLOAD_CHECKPOINTS = True
    run_fast()
    report["fast_offload_s_per_step"] = timed(run_fast, 2) / len(batches)
    config.OFFLOAD_CHECKPOINTS = False
    # the stock path now sees the fp16 embedding (same values), fine for timing
    report["stock_s_per_step"] = timed(run_stock, 1) / len(batches)
    report["tokens_real_vs_padded"] = [int(b[1].sum()) for b in batches], [b[1].numel() for b in batches]

    ok = (report["loss_rel"] < 2e-3 and report["entropy_rel"] < 5e-3 and report["accuracy_abs"] < 0.02
          and report["row_loss_rel_max"] < 5e-3 and report["grad_cos_worst"] > 0.995
          and report["scaled_grads_finite"] and report["offload_grads_identical"])
    report["PASS"] = ok
    print(json.dumps(report, indent=2))
    print("\nFAST PATH CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
