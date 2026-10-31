"""How the fast path plugs into Soup 0.77.0 / TRL 0.29.1 / transformers 5.16.1.

1. peft.prepare_model_for_kbit_training (looked up by Soup at call time): when the run is one the
   fast path implements, the tied embedding is hidden from its fp16 -> fp32 upcast, so it stays
   fp16 (467 MB instead of 934 MB, and no 1.4 GB load-time spike). This is bit-exact: on a T4 Soup
   loads the checkpoint in fp16, so the fp32 copy only ever held fp16 values, and the lm_head
   matmul runs in fp16 under autocast either way.
2. SFTTrainerWrapper._setup_transformers: after Soup built the 4-bit PEFT model, FastQwen2 checks
   it (anything unexpected -> stock path), repacks the 4-bit weights and prepares the tables.
3. SFTTrainerWrapper.train: before training starts, the trainer's compute_loss is replaced on the
   instance by `fast_compute_loss`, which runs the fused step and records exactly what the stock
   chain (HF Trainer.compute_loss -> TRL SFTTrainer.compute_loss -> Soup's RewindTrainer) records:
     loss          = summed token CE / num_items_in_batch (HF's ForCausalLMLoss with num_items)
     entropy       = mean over attention-mask tokens of the per-token entropy (TRL)
     num_tokens    = running sum of attention_mask (TRL)
     mean_token_accuracy = argmax hits / supervised tokens (TRL)
     rewind record = per-row mean CE and token counts, popped from the sampler FIFO (Soup)
   Anything it does not handle (eval, return_outputs, extra input keys, other loss settings)
   goes to the original method.
Every patch is narrow and guarded: Soup's regression tests call these methods with fake
transformers/peft modules, which never satisfy the checks, so they see the stock behaviour.
"""

import contextvars
import functools
import sys
import time
import types

from . import config

_PLAN = contextvars.ContextVar("soup_fast_plan", default=None)
_STATE = {"installed": False}


def _process_age():
    """Seconds since this process started (Linux /proc), or None."""
    try:
        import os

        with open("/proc/self/stat") as fh:
            start_ticks = int(fh.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return None


def _log(msg):
    if config.VERBOSE:
        age = _process_age()
        stamp = f"t+{age:6.2f}s " if age is not None else ""
        print(f"[solution] {stamp}{msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- config gate
def _plan_for(wrapper, cfg, tcfg):
    """True when this SFT run is the configuration the fast path implements."""
    try:
        import torch

        if not config.FAST_PATH or getattr(wrapper, "device", None) != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        checks = [
            getattr(cfg, "task", None) == "sft",
            getattr(cfg, "backend", "transformers") == "transformers",
            getattr(cfg, "modality", "text") == "text",
            getattr(tcfg, "quantization", None) == "4bit",
            int(getattr(tcfg.lora, "r", 0) or 0) > 0,
            not getattr(tcfg.lora, "use_dora", False),
            not getattr(tcfg.lora, "use_vera", False),
            not getattr(tcfg, "use_liger", False),
            not getattr(tcfg, "use_cut_ce", False),
            not getattr(tcfg, "packing", False),
            not getattr(tcfg, "multipack", False),
            not getattr(tcfg, "fast_lora", False),
            not getattr(tcfg, "stream_layers", False),
            getattr(tcfg, "neftune_alpha", None) is None,
            not getattr(tcfg, "use_flash_attn", False),
            not getattr(tcfg, "unfrozen_parameters", None),
            not getattr(tcfg, "lisa_enabled", False),
            getattr(tcfg, "freeze_layers", None) is None,
            getattr(tcfg, "freeze_ratio", None) is None,
            getattr(tcfg, "quantization_aware", None) in (None, False, "none"),
            getattr(cfg.data, "format", None) != "raft",
        ]
        return all(checks)
    except Exception:  # noqa: BLE001 - fakes in tests or unknown configs: stock path
        return False


# --------------------------------------------------------------------------- patches
def _patch_prepare_kbit():
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_soup_fast", False):
        return

    @functools.wraps(original)
    def prepare_model_for_kbit_training(model, *args, **kwargs):
        hidden = None
        try:
            import torch

            if (config.FP16_EMBEDDING and _PLAN.get() and isinstance(model, torch.nn.Module)
                    and type(model).__name__ == "Qwen2ForCausalLM"
                    and getattr(model, "is_loaded_in_4bit", False)):
                emb = model.get_input_embeddings().weight
                head = model.get_output_embeddings()
                tied = head is not None and head.weight is emb
                if tied and emb.dtype == torch.float16 and emb.is_cuda:
                    hidden = (emb, emb.data)
                    emb.data = torch.empty(0, dtype=torch.float32, device=emb.device)
        except Exception:  # noqa: BLE001
            hidden = None
        try:
            return original(model, *args, **kwargs)
        finally:
            if hidden is not None:
                hidden[0].data = hidden[1]
                _log("tied embedding kept in fp16 (bit-exact)")

    prepare_model_for_kbit_training._soup_fast = True
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training


def _patch_wrapper():
    from soup_cli.trainer import sft

    cls = sft.SFTTrainerWrapper
    if getattr(cls._setup_transformers, "_soup_fast", False):
        return
    orig_setup, orig_train = cls._setup_transformers, cls.train

    @functools.wraps(orig_setup)
    def _setup_transformers(self, cfg, tcfg):
        plan = _plan_for(self, cfg, tcfg)
        if plan:
            _log("model setup starts (tokenizer, 4-bit load, PEFT)")
        token = _PLAN.set(plan)
        try:
            result = orig_setup(self, cfg, tcfg)
        finally:
            _PLAN.reset(token)
        self._soup_fast = None
        if plan:
            _log("model loaded")
            self._soup_fast = _build_runner(self, cfg, tcfg)
        return result

    @functools.wraps(orig_train)
    def train(self, *args, **kwargs):
        runner = getattr(self, "_soup_fast", None)
        trainer = getattr(self, "trainer", None)
        if runner is not None and trainer is not None:
            if _install_on_trainer(trainer, runner):
                from . import precompile
                precompile.wait()
            else:
                runner.restore_stock()
                self._soup_fast = None
            _log("training starts")
            try:
                return orig_train(self, *args, **kwargs)
            finally:
                _log("training, final save and Soup's wrap-up done")
        return orig_train(self, *args, **kwargs)

    _setup_transformers._soup_fast = True
    train._soup_fast = True
    cls._setup_transformers = _setup_transformers
    cls.train = train


def _build_runner(wrapper, cfg, tcfg):
    import torch

    model = getattr(wrapper, "model", None)
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "get_base_model"):
        return None
    started = time.perf_counter()
    try:
        from .fast_qwen import FastQwen2

        seed = getattr(tcfg, "seed", None)
        runner = FastQwen2(model, seed=0 if seed is None else int(seed))
    except Exception as exc:  # noqa: BLE001 - any mismatch: keep the stock path
        _log(f"fast path disabled: {type(exc).__name__}: {exc}")
        _restore_fp32_embedding(model)
        return None
    _log(f"fast path ready in {time.perf_counter() - started:.2f} s "
         f"(peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated so far); "
         f"attention: {runner.attn_note}; dequant overlap: {runner.overlap}")
    return runner


def _restore_fp32_embedding(model):
    """If the fast path is not used, give the stock path the fp32 embedding it expects."""
    try:
        import torch

        emb = model.get_input_embeddings().weight
        if emb.dtype == torch.float16:
            emb.data = emb.data.float()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- trainer
def _trainer_supported(trainer):
    args = trainer.args
    checks = [
        getattr(trainer, "compute_loss_func", None) is None,
        getattr(trainer, "label_smoother", None) is None,
        not getattr(args, "use_liger_kernel", False),
        getattr(args, "loss_type", "nll") in (None, "nll"),
        getattr(args, "neftune_noise_alpha", None) is None,
        not getattr(args, "padding_free", False),
        not getattr(trainer, "padding_free", False),
        int(getattr(args, "n_gpu", 1) or 1) <= 1,
        int(getattr(trainer.accelerator, "num_processes", 1) or 1) == 1,
        int(getattr(trainer, "num_virtual_tokens", 0) or 0) == 0,
        not getattr(trainer, "aux_loss_enabled", False),
        bool(getattr(args, "fp16", False)) and not bool(getattr(args, "bf16", False)),
        hasattr(trainer, "_metrics") and hasattr(trainer, "_total_train_tokens"),
    ]
    return all(checks)


def _install_on_trainer(trainer, runner):
    if not _trainer_supported(trainer):
        _log("fast path not installed: trainer settings differ from the task")
        return False
    original = trainer.compute_loss
    trainer._soup_orig_compute_loss = original
    trainer._soup_fast_runner = runner
    trainer.compute_loss = types.MethodType(fast_compute_loss, trainer)
    _log("fast compute_loss installed")
    return True


_INPUT_KEYS = {"input_ids", "attention_mask", "labels"}


def fast_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    runner = self._soup_fast_runner
    if (return_outputs or not getattr(model, "training", False) or not self.model.training
            or not set(inputs) <= _INPUT_KEYS | {"_prediction_loss_only"}
            or not _INPUT_KEYS <= set(inputs)):
        return self._soup_orig_compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
    import torch

    from .fast_qwen import batch_metrics, run_step

    batch = runner.pack(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
    if batch is None:
        return self._soup_orig_compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
    inputs.pop("_prediction_loss_only", None)
    t0 = time.perf_counter()
    loss_sum, stats = run_step(runner, batch)

    # HF Trainer.compute_loss + ForCausalLMLoss: sum / num_items_in_batch, else mean
    if getattr(self, "model_accepts_loss_kwargs", False) and num_items_in_batch is not None:
        denom = num_items_in_batch
        if torch.is_tensor(denom):
            denom = denom.to(loss_sum.device)
        loss = loss_sum / denom
    else:
        loss = loss_sum / max(batch.n_lab, 1)

    # TRL SFTTrainer.compute_loss metrics and Soup's rewind record
    m = batch_metrics(batch, stats.detach().to("cpu"))
    metrics = self._metrics["train"]
    metrics["entropy"].append(m["entropy"])
    self._total_train_tokens += int(batch.n_real)
    metrics["num_tokens"] = [self._total_train_tokens]
    metrics["mean_token_accuracy"].append(m["accuracy"])
    state = getattr(self, "_soup_rewind_state", None)
    if state is not None and not state.failed:
        try:
            state.on_micro_batch(step=int(self.state.global_step) + 1, row_loss=m["row_loss"],
                                 row_tokens=m["row_tokens"])
        except Exception as exc:  # noqa: BLE001 - the recorder never raises into training
            state.fail(exc)
    if runner.steps <= 3:
        _log(f"step {runner.steps}: {batch.B} rows, {batch.n_real} tokens packed of {batch.B * batch.Tm} "
             f"({batch.n_lab} supervised), forward {time.perf_counter() - t0:.3f} s, "
             f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
             f"{runner.n_dequant} half-layer dequants so far, kept from forward: "
             f"{_plan_text(runner)}")
        if runner.steps == 1:
            _log(f"host arenas: {runner.host_note}")
    return loss


def _plan_text(runner):
    plan = getattr(runner, "last_plan", None)
    if plan is None:
        return "?"
    kept = [name for name, on in (("gate|up", plan.gu), ("attention", plan.attn),
                                  ("checkpoints on host", plan.x)) if on]
    return ", ".join(kept) or "nothing (recompute)"


# --------------------------------------------------------------------------- entry point
def install():
    if _STATE["installed"]:
        return
    _log("apply()")
    _patch_prepare_kbit()
    _patch_wrapper()
    _STATE["installed"] = True
