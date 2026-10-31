"""End-to-end check of the fast path against the stock transformers + PEFT + bitsandbytes path on
the real model and real data (a T4 with the challenge's packages; ~2 minutes).

    python dev/check_fast_path.py            # after python task/prepare_data.py

Builds the model the way Soup does (fp16 load, NF4 double quant, prepare_model_for_kbit_training,
LoRA r=16 alpha=32 on q_proj/v_proj), gives lora_B random values so that every LoRA gradient is
non-zero, and compares on the same right-padded batch:
    loss (sum CE / supervised tokens), TRL entropy, TRL token accuracy, rewind per-row losses,
    and the gradients of all 112 LoRA tensors (relative L2 error and cosine),
with LoRA dropout 0 (so both sides see the same network). Then it times a few steps of each
path, measures peak memory, and checks the host-offloaded variants give the same gradients.
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
SCALE = 65536.0   # the fp16 GradScaler's initial loss scale: gradients compared as training sees them


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


def stock_step(model, ids, mask, labels, scale=SCALE):
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
    (loss * scale).backward()
    return dict(loss=loss.item(), entropy=entropy, accuracy=acc, row_loss=row_loss)


def fast_step(runner, ids, mask, labels, scale=SCALE):
    from solution.fast_qwen import batch_metrics, run_step

    batch = runner.pack(ids, mask, labels)
    loss_sum, stats = run_step(runner, batch)
    loss = loss_sum / batch.n_lab
    m = batch_metrics(batch, stats.detach().cpu())
    (loss * scale).backward()
    return dict(loss=loss.item(), entropy=m["entropy"], accuracy=m["accuracy"], row_loss=m["row_loss"],
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


def grad_groups(a, b):
    """Worst (relative L2, cosine, layer) per LoRA tensor kind (A_q, B_q, A_v, B_v)."""
    out = {}
    for i, (x, y) in enumerate(zip(a, b)):
        kind = ("A_q", "B_q", "A_v", "B_v")[i % 4]
        rel = float((x - y).norm() / y.norm().clamp(min=1e-30))
        cos = float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))
        cur = out.get(kind)
        if cur is None or rel > cur[0]:
            out[kind] = (round(rel, 5), round(cos, 6), i // 4)
    return out


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
    report["grad_groups_vs_stock (rel, cos, layer)"] = grad_groups(fast_grads, ref_grads)
    # noise floor: the same fast step twice (SDPA's backward is not deterministic)
    for p in params:
        p.grad = None
    fast_step(runner, ids, mask, labels)
    rel2, cos2 = compare_grads(grads_of(model), fast_grads)
    report["fast_vs_fast_rel_l2_worst"] = rel2
    report["fast_vs_fast_cos_worst"] = cos2
    report["stock"] = ref
    report["fast"] = got
    report["loss_rel"] = abs(got["loss"] - ref["loss"]) / ref["loss"]
    report["entropy_rel"] = abs(got["entropy"] - ref["entropy"]) / ref["entropy"]
    report["accuracy_abs"] = abs(got["accuracy"] - ref["accuracy"])
    report["row_loss_rel_max"] = max(abs(a - b) / b for a, b in zip(got["row_loss"], ref["row_loss"]))
    report["grad_rel_l2_worst"] = rel
    report["grad_cos_worst"] = cos
    report["attention"] = runner.attn_note
    report["dequant_overlap"] = runner.overlap
    d0 = runner.n_dequant
    for p in params:
        p.grad = None
    fast_step(runner, ids, mask, labels)
    report["dequant_parts_per_step (2 per layer pass; 108 = 54 layers)"] = runner.n_dequant - d0

    ok = (report["loss_rel"] < 2e-3 and report["entropy_rel"] < 5e-3 and report["accuracy_abs"] < 0.02
          and report["row_loss_rel_max"] < 5e-3 and report["grad_cos_worst"] > 0.995)
    report["CORRECTNESS_PASS"] = ok
    print(json.dumps(report, indent=2), flush=True)
    del ref_grads
    torch.cuda.empty_cache()
    extra = {}

    def section(name, fn):
        started = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            extra[name + "_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[section {name} done in {time.perf_counter() - started:.1f} s]", flush=True)

    from solution.fast_qwen import run_step

    def scaled():
        for p in params:
            p.grad = None
        batch = runner.pack(ids, mask, labels)
        loss_sum, _ = run_step(runner, batch)
        (loss_sum / batch.n_lab * SCALE).backward()
        extra["scaled_grads_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in params)

    def run_fast():
        for b in batches:
            for p in params:
                p.grad = None
            fast_step(runner, *b)

    def pcie():
        """Pinned host <-> GPU copy bandwidth, alone and next to a GEMM (what the host arenas use)."""
        mb = 256
        host = torch.empty(mb << 20, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
        side = torch.cuda.Stream()
        out = {}
        for name, fn in (("h2d", lambda: dev.copy_(host, non_blocking=True)),
                         ("d2h", lambda: host.copy_(dev, non_blocking=True))):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(4):
                fn()
            torch.cuda.synchronize()
            out[name + "_GBps"] = round(4 * mb / 1024 / (time.perf_counter() - t0), 2)
        a = torch.randn(1024, 1536, device="cuda").half()
        w = torch.randn(17920, 1536, device="cuda").half()
        gemm_s = timed(lambda: torch.mm(a, w.t()), 20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(side):
            for _ in range(4):
                host.copy_(dev, non_blocking=True)
        n = 0
        while not side.query():
            torch.mm(a, w.t())
            torch.cuda.current_stream().synchronize()
            n += 1
        torch.cuda.synchronize()
        out["d2h_GBps_next_to_gemm"] = round(4 * mb / 1024 / (time.perf_counter() - t0), 2)
        out["gemm_ms_alone"] = round(1000 * gemm_s, 3)
        out["gemms_during_copy"] = n
        extra["pcie"] = out
        print("pcie", out, flush=True)

    def variants():
        """The same four batches through variants of the fast path, in this one process, in
        interleaved rounds (the T4's clock drifts as it heats, so A then B is biased): median
        step time, peak memory, and gradient agreement with the default."""
        runner.start_host_alloc(int(config.HOST_CAPACITY), kinds={"x", "gu", "qkv", "out", "op"})
        runner._host_ready()
        extra["host_arenas"] = runner.host_note
        keys = ("SAVE_GU", "SAVE_ATTN", "OFFLOAD_CHECKPOINTS")
        base_cfg = {k: getattr(config, k) for k in keys}
        ov0 = runner.overlap
        plan = [("default", {}),
                ("recompute_all (v4)", {"SAVE_GU": False, "SAVE_ATTN": False}),
                ("save_gu_only", {"SAVE_GU": True, "SAVE_ATTN": False}),
                ("save_gu_attn", {"SAVE_GU": True, "SAVE_ATTN": True}),
                ("save_all_ckpt_on_host", {"SAVE_GU": True, "SAVE_ATTN": True, "OFFLOAD_CHECKPOINTS": True}),
                ("no_dequant_overlap", {"overlap": False})]

        def apply(change):
            torch.cuda.synchronize()
            for k in keys:
                setattr(config, k, change.get(k, base_cfg[k]))
            runner.overlap = change.get("overlap", ov0)

        def grads_b0():
            for p in params:
                p.grad = None
            fast_step(runner, ids, mask, labels)
            return grads_of(model)

        apply({})
        ref = grads_b0()
        times = {name: [] for name, _ in plan}
        peaks, coss = {}, {}
        for name, change in plan:
            apply(change)
            try:
                coss[name] = round(compare_grads(grads_b0(), ref)[1], 6)
                run_fast()
                torch.cuda.reset_peak_memory_stats()
                run_fast()
                peaks[name] = round(torch.cuda.max_memory_allocated() / 2**20)
            except Exception as exc:  # noqa: BLE001
                coss[name] = f"{type(exc).__name__}: {exc}"
        for _ in range(3):
            for name, change in plan:
                if not isinstance(coss.get(name), float):
                    continue
                apply(change)
                times[name].append(timed(run_fast, 1) / len(batches))
        apply({})
        out = {}
        for name, _ in plan:
            ts = sorted(times[name])
            out[name] = {"s_per_step_median": round(ts[len(ts) // 2], 4) if ts else None,
                         "rounds": [round(t, 4) for t in times[name]],
                         "peak_mib": peaks.get(name), "grad_cos_vs_default": coss.get(name)}
            print(name, out[name], flush=True)
        extra["variants"] = out

    def profile():
        """Where the GPU waits inside our step: busy time (union of kernel intervals on all
        streams) against wall time, and the largest gaps with the kernel before them."""
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile

        run_fast()
        torch.cuda.synchronize()
        with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            run_fast()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        ev = [e for e in prof.events() if e.device_type == DeviceType.CUDA and e.time_range.end > e.time_range.start]
        ev.sort(key=lambda e: e.time_range.start)
        busy, gaps, cur_s, cur_e, prev = 0.0, [], None, None, None
        for e in ev:
            st, en = e.time_range.start, e.time_range.end
            if cur_e is None or st > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                    gaps.append((st - cur_e, prev))
                cur_s, cur_e = st, en
            else:
                cur_e = max(cur_e, en)
            prev = e.name
        if cur_e is not None:
            busy += cur_e - cur_s
        gaps.sort(key=lambda g: -g[0])
        span = (ev[-1].time_range.end - ev[0].time_range.start) if ev else 0
        steps = len(batches)
        extra["profile"] = {"wall_ms_per_step": round(1000 * wall / steps, 1),
                            "gpu_span_ms_per_step": round(span / 1000 / steps, 1),
                            "gpu_busy_ms_per_step": round(busy / 1000 / steps, 1),
                            "idle_gaps_over_50us_ms_per_step": round(sum(g for g, _ in gaps if g > 50) / 1000 / steps, 1),
                            "largest_gaps_us_after": [(round(g), str(n)[:70]) for g, n in gaps[:12]]}
        print("profile", extra["profile"], flush=True)

    section("scaled", scaled)
    section("pcie", pcie)
    section("variants", variants)
    section("profile", profile)
    extra["tokens_real_vs_padded"] = [[int(b[1].sum()) for b in batches], [b[1].numel() for b in batches]]

    print("\n======== SUMMARY ========")
    keys = ["loss_rel", "entropy_rel", "accuracy_abs", "row_loss_rel_max", "grad_rel_l2_worst", "grad_cos_worst",
            "fast_vs_fast_cos_worst", "grad_groups_vs_stock (rel, cos, layer)", "stock_peak_mib", "fast_peak_mib",
            "attention", "dequant_overlap"]
    print(json.dumps({k: report.get(k) for k in keys}, indent=1))
    print(json.dumps(extra, indent=1))
    print("\nFAST PATH CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
