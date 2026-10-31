"""End-to-end check of the fast path against the stock transformers + PEFT + bitsandbytes path on
the real model and real data (a T4 with the challenge's packages; ~2 minutes).

    python dev/check_fast_path.py            # after python task/prepare_data.py

Builds the model the way Soup does (fp16 load, NF4 double quant, prepare_model_for_kbit_training,
LoRA r=16 alpha=32 on q_proj/v_proj), gives lora_B random values so that every LoRA gradient is
non-zero, and compares on the same right-padded batch:
    loss (sum CE / supervised tokens), TRL entropy, TRL token accuracy, rewind per-row losses,
    and the gradients of all 112 LoRA tensors (relative L2 error and cosine),
with LoRA dropout 0 (so both sides see the same network). Then:
  * loads the model a second time with the solution's load hooks active (what happens inside
    `soup train`): every 4-bit weight, its quantisation state and the fp16 embedding must be
    bit-identical to the normal load, and the GPU peak of that load is reported;
  * times the step and measures the peak memory of the v6 placement (everything streamed from
    pinned host memory) against variants (embedding on the GPU, a few layers resident, the v5
    layout), in this one process, on the same four batches, with gradient agreement;
  * measures PCIe bandwidth (each direction alone and both at once), pinned-memory allocation and
    release, and profiles where the GPU waits inside the default step.
Ends with a SUMMARY block. Please send back the whole output.
"""

import json
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


def load_with_hooks():
    """from_pretrained the way `soup train` runs it with the solution: the load hooks of
    integration.py active. Returns (model, GPU reserved MiB peak above the level before)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from solution import integration

    integration._patch_loading()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    token = integration._PLAN.set(True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                     device_map="auto", dtype=torch.float16)
    finally:
        integration._PLAN.reset(token)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_reserved() - before) / 2**20
    return model, peak, dict(integration._LOAD)


def compare_loaded(ref_model, hooked):
    """Bit-exactness of the hooked load against the normal one (before the runner repacks)."""
    out = {"weights_compared": 0, "weights_on_host": 0, "mismatch": []}
    ref_mods = dict(ref_model.get_base_model().named_modules())
    for name, mod in hooked.named_modules():
        if type(mod).__name__ != "Linear4bit":
            continue
        ref = ref_mods.get(name) or ref_mods.get(name + ".base_layer")
        if ref is None:
            out["mismatch"].append(f"{name}: missing in reference")
            continue
        w, rw = mod.weight, ref.weight
        out["weights_on_host"] += int(not w.data.is_cuda)
        qs, rq = w.quant_state, rw.quant_state
        same = (torch.equal(w.data.cpu(), rw.data.cpu()) and torch.equal(qs.absmax.cpu(), rq.absmax.cpu())
                and torch.equal(qs.state2.absmax.cpu(), rq.state2.absmax.cpu())
                and torch.equal(qs.offset.cpu(), rq.offset.cpu())
                and torch.equal(qs.state2.code.cpu(), rq.state2.code.cpu())
                and torch.equal(qs.code.cpu(), rq.code.cpu()))
        out["weights_compared"] += 1
        if not same:
            out["mismatch"].append(name)
    e1 = hooked.get_input_embeddings().weight
    e0 = ref_model.get_base_model().get_input_embeddings().weight
    out["embedding_device"] = str(e1.device)
    out["embedding_dtype"] = str(e1.dtype)
    out["embedding_bit_exact"] = bool(torch.equal(e1.data.cpu().half(), e0.data.cpu().half()))
    out["norms_on_gpu"] = all(p.is_cuda for n, p in hooked.named_parameters() if "norm" in n)
    out["mismatch"] = out["mismatch"][:5]
    return out


def main():
    from solution import config, precompile

    report, extra = {}, {}
    rows = [json.loads(line) for line in open(REPO / "data" / "train.jsonl", encoding="utf-8")]
    torch.cuda.reset_peak_memory_stats()
    tok, model = build(dropout=0.0)
    report["stock_load_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    params = lora_params(model)
    batches = [make_batch(tok, rows[i:i + 4]) for i in (0, 4, 8, 40)]
    ids, mask, labels = batches[0]

    torch.cuda.reset_peak_memory_stats()
    ref = stock_step(model, ids, mask, labels)
    ref_grads = grads_of(model)
    report["stock_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    for p in params:
        p.grad = None
    torch.cuda.empty_cache()

    def section(name, fn):
        started = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            import traceback
            extra[name + "_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        print(f"[section {name} done in {time.perf_counter() - started:.1f} s]", flush=True)

    kept = {}

    def load_hooks():
        t0 = time.perf_counter()
        hooked, peak, note = load_with_hooks()
        extra["hooked_load_s"] = round(time.perf_counter() - t0, 2)
        extra["hooked_load_gpu_peak_reserved_above_baseline_mib"] = round(peak, 1)
        extra["hooked_load_note"] = note
        extra["hooked_load_vs_normal"] = compare_loaded(model, hooked)
        kept["hooked"] = hooked          # for the integration check below (it lives in host memory)
        print("load hooks", extra["hooked_load_vs_normal"], flush=True)

    section("load_hooks", load_hooks)

    from solution.fast_qwen import FastQwen2
    precompile.start()
    precompile.wait()
    t0 = time.perf_counter()
    runner = FastQwen2(model, seed=1)
    runner._host_ready()
    report["runner_build_s"] = round(time.perf_counter() - t0, 2)
    report["placement"] = runner.placement()
    report["gpu_allocated_after_build_mib"] = round(torch.cuda.memory_allocated() / 2**20)
    torch.cuda.reset_peak_memory_stats()
    got = fast_step(runner, ids, mask, labels)
    fast_grads = grads_of(model)
    report["fast_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    rel, cos = compare_grads(fast_grads, ref_grads)
    report["grad_groups_vs_stock (rel, cos, layer)"] = grad_groups(fast_grads, ref_grads)
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
    c0 = runner.pipe.copies if runner.pipe is not None else 0
    e0 = getattr(runner.emb_source, "copies", 0)
    for p in params:
        p.grad = None
    fast_step(runner, ids, mask, labels)
    report["dequant_parts_per_step (2 per layer pass; 108 = 54 layers)"] = runner.n_dequant - d0
    report["weight_copies_per_step (52 = 2 x 26 layers)"] = (runner.pipe.copies - c0) if runner.pipe else 0
    report["embedding_chunks_per_step"] = getattr(runner.emb_source, "copies", 0) - e0

    ok = (report["loss_rel"] < 2e-3 and report["entropy_rel"] < 5e-3 and report["accuracy_abs"] < 0.02
          and report["row_loss_rel_max"] < 5e-3 and report["grad_cos_worst"] > 0.995)
    hv = extra.get("hooked_load_vs_normal", {})
    load_ok = bool(hv) and not hv.get("mismatch") and hv.get("embedding_bit_exact") is True \
        and hv.get("weights_compared") == 196
    report["CORRECTNESS_PASS"] = ok
    report["LOAD_HOOKS_BIT_EXACT"] = load_ok
    print(json.dumps(report, indent=2), flush=True)
    del ref_grads

    def self_test():
        """The start-up self-test `soup train` runs in the background build (time and result)."""
        t0 = time.perf_counter()
        extra["self_test"] = runner.self_test()
        extra["self_test_s"] = round(time.perf_counter() - t0, 3)
        print("self-test", extra["self_test"], flush=True)

    def integration_path():
        """What `soup train` builds: the hooked load (4-bit weights and embedding loaded to host),
        the patched prepare_model_for_kbit_training (fp16 embedding), PEFT LoRA created on the
        host, FastQwen2 moving LoRA to the GPU; same LoRA values as the reference model. Its step
        must match the runner built from the normally loaded model; then the v5-placement
        fallback (rebuilt from the streamed state) must match too."""
        import peft
        from peft import LoraConfig, get_peft_model

        from solution import integration

        hooked = kept.pop("hooked")
        integration._patch_prepare_kbit()
        token = integration._PLAN.set(True)
        try:
            hooked = peft.prepare_model_for_kbit_training(hooked, use_gradient_checkpointing=False)
        finally:
            integration._PLAN.reset(token)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                         bias="none", task_type="CAUSAL_LM")
        hp = get_peft_model(hooked, cfg)
        hp.train()
        mine = dict((n, p) for n, p in hp.named_parameters() if p.requires_grad)
        theirs = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        out = {"lora_device_before": str(next(iter(mine.values())).device),
               "embedding_dtype": str(hp.get_base_model().get_input_embeddings().weight.dtype)}
        with torch.no_grad():
            for n, p in mine.items():
                p.data = theirs[n].data.to(p.device).clone()
        cap = config.HOST_CAPACITY
        config.HOST_CAPACITY = 0
        try:
            r2 = FastQwen2(hp, seed=1)
        finally:
            config.HOST_CAPACITY = cap
        r2.host, r2.host_cap = runner.host, runner.host_cap
        out["placement"] = r2.placement()
        hparams = list(mine.values())
        for p in hparams:
            p.grad = None
        g2 = fast_step(r2, ids, mask, labels)
        grads2 = [p.grad.detach().float().clone() for p in hparams]
        out["loss_vs_runner"] = abs(g2["loss"] - got["loss"]) / got["loss"]
        out["grad_rel_l2_worst_vs_runner"], out["grad_cos_worst_vs_runner"] = compare_grads(grads2, fast_grads)
        t0 = time.perf_counter()
        out["self_test"] = r2.self_test()
        out["self_test_s"] = round(time.perf_counter() - t0, 3)
        # the fallback: v5 placement rebuilt from the streamed state, tested and compared
        t0 = time.perf_counter()
        r2.use_resident_layout("dev check")
        out["fallback_rebuild_s"] = round(time.perf_counter() - t0, 2)
        r2.host, r2.host_cap = runner.host, runner.host_cap
        out["fallback_placement"] = r2.placement()
        out["fallback_self_test"] = r2.self_test()
        for p in hparams:
            p.grad = None
        g3 = fast_step(r2, ids, mask, labels)
        grads3 = [p.grad.detach().float().clone() for p in hparams]
        out["fallback_loss_vs_runner"] = abs(g3["loss"] - got["loss"]) / got["loss"]
        out["fallback_grad_cos_worst_vs_runner"] = compare_grads(grads3, fast_grads)[1]
        out["PASS"] = (out["loss_vs_runner"] < 1e-4 and out["grad_cos_worst_vs_runner"] > 0.9999
                       and out["fallback_loss_vs_runner"] < 2e-3
                       and out["fallback_grad_cos_worst_vs_runner"] > 0.999)
        extra["integration_path"] = out
        print("integration path", out, flush=True)
        del r2, hp, hooked, mine, hparams, grads2, grads3
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    section("self_test", self_test)
    if "hooked" in kept:
        section("integration_path", integration_path)
    kept.clear()

    from solution.fast_qwen import run_step

    def scaled():
        for p in params:
            p.grad = None
        batch = runner.pack(ids, mask, labels)
        loss_sum, _ = run_step(runner, batch)
        (loss_sum / batch.n_lab * SCALE).backward()
        extra["scaled_grads_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in params)

    def run_fast(r):
        for b in batches:
            for p in params:
                p.grad = None
            fast_step(r, *b)

    def pcie():
        """Pinned host <-> GPU copy bandwidth: each direction alone, both at once (what the v6
        forward does: activations out, weights in), and next to a GEMM; pinned alloc/free cost."""
        mb = 256
        host = torch.empty(mb << 20, dtype=torch.uint8, pin_memory=True)
        host2 = torch.empty(mb << 20, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
        dev2 = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(4):
            with torch.cuda.stream(s1):
                dev.copy_(host, non_blocking=True)
            with torch.cuda.stream(s2):
                host2.copy_(dev2, non_blocking=True)
        torch.cuda.synchronize()
        out["both_directions_GBps_each"] = round(4 * mb / 1024 / (time.perf_counter() - t0), 2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(4):
            with torch.cuda.stream(s1):
                dev.copy_(host, non_blocking=True)
            with torch.cuda.stream(s2):
                dev2.copy_(host2, non_blocking=True)
        torch.cuda.synchronize()
        out["two_h2d_streams_GBps_total"] = round(8 * mb / 1024 / (time.perf_counter() - t0), 2)
        a = torch.randn(1024, 1536, device="cuda").half()
        w = torch.randn(17920, 1536, device="cuda").half()
        gemm_s = timed(lambda: torch.mm(a, w.t()), 20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s1):
            for _ in range(4):
                host.copy_(dev, non_blocking=True)
        n = 0
        while not s1.query():
            torch.mm(a, w.t())
            torch.cuda.current_stream().synchronize()
            n += 1
        torch.cuda.synchronize()
        out["d2h_GBps_next_to_gemm"] = round(4 * mb / 1024 / (time.perf_counter() - t0), 2)
        out["gemm_ms_alone"] = round(1000 * gemm_s, 3)
        out["gemms_during_copy"] = n
        host = host2 = dev = dev2 = None   # (not del: the lambdas above close over them)
        t0 = time.perf_counter()
        big = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
        out["pin_alloc_1GiB_s"] = round(time.perf_counter() - t0, 3)
        del big
        t0 = time.perf_counter()
        try:
            torch._C._host_emptyCache()
            out["pin_free_cached_s"] = round(time.perf_counter() - t0, 3)
        except Exception as exc:  # noqa: BLE001
            out["pin_free_cached_s"] = f"{type(exc).__name__}"
        extra["pcie"] = out
        print("pcie", out, flush=True)

    def variants():
        """The v6 placement against variants, each its own runner on the same model, built and
        measured one after the other (the activation arenas are shared), the default once more at
        the end (the T4's clock drifts as it heats). Median s/step over 3 rounds of the 4 batches,
        allocator peaks, PCIe copies per step, gradient agreement with the default."""
        import gc

        keys = ("OFFLOAD_WEIGHTS", "OFFLOAD_EMBEDDING", "OFFLOAD_CHECKPOINTS", "RESIDENT_LAYERS",
                "HEAD_IMPL", "SAVE_GU", "SAVE_ATTN", "DEQUANT_OVERLAP", "HOST_CAPACITY")
        base_cfg = {k: getattr(config, k) for k in keys}
        plan = [("v6 default (all in host memory)", {}),
                ("embedding on GPU", {"OFFLOAD_EMBEDDING": False}),
                ("6 layers resident", {"RESIDENT_LAYERS": 6}),
                ("checkpoints on GPU", {"OFFLOAD_CHECKPOINTS": False}),
                ("v5 layout (all on GPU, rows head)", {"OFFLOAD_WEIGHTS": False, "OFFLOAD_EMBEDDING": False,
                                                      "OFFLOAD_CHECKPOINTS": False, "HEAD_IMPL": "rows"}),
                ("v6 default again", {})]
        out, ref = {}, None
        for name, change in plan:
            for k in keys:
                setattr(config, k, change.get(k, base_cfg[k]))
            config.HOST_CAPACITY = 0
            r = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                r = FastQwen2(model, seed=1)
                config.HOST_CAPACITY = base_cfg["HOST_CAPACITY"]
                r._host_ready()
                r.host, r.host_cap = shared
                for p in params:
                    p.grad = None
                fast_step(r, ids, mask, labels)
                g = grads_of(model)
                if ref is None:
                    ref = g
                cos = round(compare_grads(g, ref)[1], 6)
                run_fast(r)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                c0 = r.pipe.copies if r.pipe is not None else 0
                e0 = getattr(r.emb_source, "copies", 0)
                run_fast(r)
                peak_a = round(torch.cuda.max_memory_allocated() / 2**20)
                peak_r = round(torch.cuda.max_memory_reserved() / 2**20)
                copies = ((r.pipe.copies - c0) if r.pipe is not None else 0) / len(batches)
                echunks = (getattr(r.emb_source, "copies", 0) - e0) / len(batches)
                ts = sorted(timed(lambda: run_fast(r), 1) / len(batches) for _ in range(3))
                free, total = torch.cuda.mem_get_info()
                out[name] = {"s_per_step_median": round(ts[1], 4), "rounds": [round(t, 4) for t in ts],
                             "peak_allocated_mib": peak_a, "peak_reserved_mib": peak_r,
                             "device_used_now_mib": round((total - free) / 2**20),
                             "weight_copies_per_step": copies, "embedding_chunks_per_step": echunks,
                             "grad_cos_vs_default": cos, "placement": r.placement()}
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                out[name] = f"{type(exc).__name__}: {exc}"
            finally:
                config.HOST_CAPACITY = base_cfg["HOST_CAPACITY"]
                r = None
            print(name, out[name], flush=True)
        for k in keys:
            setattr(config, k, base_cfg[k])
        extra["variants"] = out

    def profile():
        """Where the GPU waits inside the default step: busy time (union of kernel intervals on all
        streams) against wall time, and the largest gaps with the kernel before them."""
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile

        import gc
        gc.collect()
        cap = config.HOST_CAPACITY
        config.HOST_CAPACITY = 0
        try:
            r = FastQwen2(model, seed=1)
        finally:
            config.HOST_CAPACITY = cap
        r.host, r.host_cap = shared
        run_fast(r)
        torch.cuda.synchronize()
        with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            run_fast(r)
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
        totals = {}
        for e in ev:
            key = "memcpy HtoD" if "HtoD" in e.name else "memcpy DtoH" if "DtoH" in e.name else (
                "gemm" if ("gemm" in e.name or "Kernel2" in e.name) else e.name[:40])
            totals[key] = totals.get(key, 0.0) + (e.time_range.end - e.time_range.start)
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:14]
        extra["profile"] = {"wall_ms_per_step": round(1000 * wall / steps, 1),
                            "gpu_span_ms_per_step": round(span / 1000 / steps, 1),
                            "gpu_busy_ms_per_step": round(busy / 1000 / steps, 1),
                            "idle_gaps_over_50us_ms_per_step": round(sum(g for g, _ in gaps if g > 50) / 1000 / steps, 1),
                            "largest_gaps_us_after": [(round(g), str(n)[:70]) for g, n in gaps[:12]],
                            "ms_per_step_by_kind": [(k, round(v / 1000 / steps, 2)) for k, v in top]}
        print("profile", extra["profile"], flush=True)
        del r

    section("scaled", scaled)
    section("pcie", pcie)
    # one set of activation arenas for every runner below; the first runner goes away so that its
    # GPU buffers do not count in the variants' peaks
    runner.start_host_alloc(int(config.HOST_CAPACITY), kinds={"x", "gu", "qkv", "out", "op"})
    runner._host_ready()
    shared = (runner.host, runner.host_cap)
    extra["host_arenas"] = runner.host_note
    runner = None                    # (not del: the sections above close over it)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    section("variants", variants)
    section("profile", profile)
    extra["tokens_real_vs_padded"] = [[int(b[1].sum()) for b in batches], [b[1].numel() for b in batches]]

    print("\n======== SUMMARY ========")
    keys = ["loss_rel", "entropy_rel", "accuracy_abs", "row_loss_rel_max", "grad_rel_l2_worst", "grad_cos_worst",
            "fast_vs_fast_cos_worst", "grad_groups_vs_stock (rel, cos, layer)", "stock_peak_mib", "fast_peak_mib",
            "stock_load_peak_reserved_mib", "gpu_allocated_after_build_mib", "runner_build_s", "placement",
            "attention", "dequant_overlap", "dequant_parts_per_step (2 per layer pass; 108 = 54 layers)",
            "weight_copies_per_step (52 = 2 x 26 layers)", "embedding_chunks_per_step",
            "LOAD_HOOKS_BIT_EXACT"]
    print(json.dumps({k: report.get(k) for k in keys}, indent=1))
    print(json.dumps(extra, indent=1, default=str))
    integ = extra.get("integration_path", {})
    integ_ok = isinstance(integ, dict) and integ.get("PASS") is True
    print("\nFAST PATH CHECK:", "PASS" if ok else "FAIL", "| LOAD HOOKS:", "BIT-EXACT" if load_ok else "CHECK",
          "| INTEGRATION + FALLBACK:", "PASS" if integ_ok else "CHECK",
          "| SELF-TEST:", "OK" if "self_test" in extra else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
