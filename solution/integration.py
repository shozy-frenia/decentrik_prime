"""How the fast path plugs into Soup 0.77.0 / TRL 0.29.1 / transformers 5.16.1.

1. Model loading (transformers' core_model_loading, looked up by name at call time), only while
   Soup loads the model of a run the fast path implements (the _PLAN context variable is set
   around Soup's _setup_transformers): transformers' caching-allocator warm-up is skipped (it
   reserves the size of the whole model on the GPU before loading), the embedding is
   materialised on the CPU instead of the GPU, and every 4-bit weight is moved to host memory
   right after bitsandbytes has quantised it on the GPU (the quantisation itself is unchanged:
   bitsandbytes' own kernel on the GPU, as in the base run). The GPU never holds the model.
2. peft.prepare_model_for_kbit_training (looked up by Soup at call time): when the run is one the
   fast path implements, the tied embedding is hidden from its fp16 -> fp32 upcast, so it stays
   fp16 (467 MB instead of 934 MB). This is bit-exact: on a T4 Soup loads the checkpoint in fp16,
   so the fp32 copy only ever held fp16 values, and the lm_head matmul runs in fp16 under
   autocast either way.
3. SFTTrainerWrapper._setup_transformers: after Soup built the 4-bit PEFT model, FastQwen2 checks
   it (anything unexpected -> stock path, with everything moved back to the GPU), repacks the
   4-bit weights into pinned host memory and prepares the tables.
4. SFTTrainerWrapper.train: before training starts, the trainer's compute_loss is replaced on the
   instance by `fast_compute_loss`, which runs the fused step and records exactly what the stock
   chain (HF Trainer.compute_loss -> TRL SFTTrainer.compute_loss -> Soup's RewindTrainer) records:
     loss          = summed token CE / num_items_in_batch (HF's ForCausalLMLoss with num_items)
     entropy       = mean over attention-mask tokens of the per-token entropy (TRL)
     num_tokens    = running sum of attention_mask (TRL)
     mean_token_accuracy = argmax hits / supervised tokens (TRL)
     rewind record = per-row mean CE and token counts, popped from the sampler FIFO (Soup)
   The per-token statistics come back to the host with an asynchronous copy and are recorded at
   the end of the step's backward (and in any case before the trainer logs), so the CPU never
   waits between the forward and the backward.
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
_CTX = {}                       # CUDA context settings made in apply() (for the log)
_LOAD = {"host_weights": 0, "host_bytes": 0, "embedding_cpu": False, "warmup_skipped": False,
         "error": None, "disabled": False}


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
            getattr(tcfg, "rope_scaling_type", None) in (None, False, "none"),
            not getattr(tcfg, "relora_steps", None),
        ]
        return all(checks)
    except Exception:  # noqa: BLE001 - fakes in tests or unknown configs: stock path
        return False


def _host_load():
    """The model being loaded right now is the fast path's, and it should stay off the GPU."""
    return bool(config.LOAD_TO_HOST and _PLAN.get() and not _LOAD["disabled"]
                and (config.OFFLOAD_WEIGHTS or config.OFFLOAD_EMBEDDING))


# --------------------------------------------------------------------------- patches
def _is_embedding_key(name):
    return isinstance(name, str) and (name.endswith("embed_tokens.weight") or name == "lm_head.weight")


def _weight_to_host(model, target_name):
    """After transformers set `target_name`: if it is a 4-bit weight bitsandbytes has just
    quantised on the GPU, move its packed data and quantisation state to host memory."""
    module_path, _, pname = target_name.rpartition(".")
    if pname != "weight":
        return
    mod = model.get_submodule(module_path) if module_path else model
    w = getattr(mod, "weight", None)
    if type(w).__name__ != "Params4bit" or not getattr(w, "bnb_quantized", False) or not w.is_cuda:
        return
    w.data = w.data.to("cpu")
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        qs.to("cpu")
    _LOAD["host_weights"] += 1
    _LOAD["host_bytes"] += int(w.data.numel() * w.data.element_size())


def _embedding_to_pinned(tensor, dtype):
    """The only tensor our hooks load to the CPU is the (tied) embedding. Read it from the
    checkpoint in blocks of 8192 rows, convert bf16 -> fp16 on the GPU (same rounding as the CPU
    conversion) and write it straight into the page-locked buffer the fast path streams the LM head
    from: the table never sits on the GPU as a whole, and no pageable fp16 copy is made first.
    None (-> the stock path) for anything that is not a 2-D float table."""
    import torch

    from . import hostmem

    shape = tuple(tensor.get_shape()) if hasattr(tensor, "get_shape") else tuple(tensor.shape)
    if len(shape) != 2 or shape[0] < 65536 or dtype != torch.float16:
        return None
    rows, cols = (int(v) for v in shape)
    dst = hostmem.take("embedding", (rows, cols), torch.float16)
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev)
    step = 8192
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        part = tensor[r0:r1]
        if not torch.is_tensor(part) or tuple(part.shape) != (r1 - r0, cols) \
                or not part.is_floating_point():
            stream.synchronize()
            return None
        g = part.to(dev)
        dst[r0:r1].copy_(g.to(dtype), non_blocking=True)
        del g
    stream.synchronize()
    _LOAD["embedding_pinned"] = True
    return dst


def _patch_loading():
    try:
        from transformers import core_model_loading as cml
        from transformers import modeling_utils as mu
    except Exception:  # noqa: BLE001 - another transformers layout: load as usual
        return
    if getattr(getattr(cml, "set_param_for_module", None), "_soup_fast", False):
        return
    orig_get_device = getattr(cml, "get_device", None)
    orig_set = getattr(cml, "set_param_for_module", None)
    orig_warm = getattr(mu, "caching_allocator_warmup", None)
    if orig_get_device is None or orig_set is None:
        return

    @functools.wraps(orig_get_device)
    def get_device(device_map, param_name, *args, **kwargs):
        device = orig_get_device(device_map, param_name, *args, **kwargs)
        try:
            if config.OFFLOAD_EMBEDDING and _host_load() and _is_embedding_key(param_name) \
                    and device != "disk":
                _LOAD["embedding_cpu"] = True
                return "cpu"
        except Exception:  # noqa: BLE001
            pass
        return device

    @functools.wraps(orig_set)
    def set_param_for_module(model, target_name, *args, **kwargs):
        result = orig_set(model, target_name, *args, **kwargs)
        if config.OFFLOAD_WEIGHTS and _host_load():
            try:
                _weight_to_host(model, target_name)
            except Exception as exc:  # noqa: BLE001 - leave it where it is; FastQwen2 copes
                _LOAD["error"] = f"{type(exc).__name__}: {exc}"
        return result

    orig_mat = getattr(cml, "_materialize_copy", None)
    if orig_mat is not None and config.LOAD_CONVERT_ON_GPU:
        @functools.wraps(orig_mat)
        def _materialize_copy(tensor, device=None, dtype=None):
            # transformers' blocking .to(cuda, fp16) converts bf16 -> fp16 on the CPU before the
            # copy; on the GPU it is the same rounding (bf16 -> fp32 is exact, fp32 -> fp16 is
            # round-to-nearest-even on both), just much faster. Only while our model loads.
            if _host_load() and device is not None and dtype is not None:
                try:
                    import torch

                    dev = torch.device(device)
                    if dev.type == "cpu" and config.OFFLOAD_EMBEDDING and config.EMBEDDING_DIRECT:
                        pinned = _embedding_to_pinned(tensor, dtype)
                        if pinned is not None:
                            return pinned
                    t = tensor[...]
                    if (dev.type == "cuda" and t.device.type == "cpu" and t.is_floating_point()
                            and t.dtype != dtype):
                        return t.to(device=dev).to(dtype)
                    return orig_mat(t, device, dtype)
                except Exception:  # noqa: BLE001
                    pass
            return orig_mat(tensor, device, dtype)

        _materialize_copy._soup_fast = True
        cml._materialize_copy = _materialize_copy
    get_device._soup_fast = True
    set_param_for_module._soup_fast = True
    cml.get_device = get_device
    cml.set_param_for_module = set_param_for_module
    if orig_warm is not None:
        @functools.wraps(orig_warm)
        def caching_allocator_warmup(*args, **kwargs):
            if _host_load():
                _LOAD["warmup_skipped"] = True
                return None
            return orig_warm(*args, **kwargs)

        caching_allocator_warmup._soup_fast = True
        mu.caching_allocator_warmup = caching_allocator_warmup


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
                if tied and emb.dtype == torch.float16 and emb.device.type in ("cuda", "cpu"):
                    hidden = (emb, emb.data)
                    emb.data = torch.empty(0, dtype=torch.float32, device=emb.device)
        except Exception:  # noqa: BLE001
            hidden = None
        try:
            return original(model, *args, **kwargs)
        finally:
            if hidden is not None:
                hidden[0].data = hidden[1]
                _log(f"tied embedding kept in fp16 (bit-exact), on {hidden[1].device.type}")

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
            try:
                result = orig_setup(self, cfg, tcfg)
            except Exception as exc:  # noqa: BLE001
                if not (plan and _host_load()):
                    raise
                # the host-loading hooks are the only change to Soup's model setup: run it again
                # without them (the fast path then starts from a GPU-resident model)
                _log(f"model setup with the host-loading hooks failed ({type(exc).__name__}: {exc});"
                     " loading again without them")
                _LOAD["disabled"] = True
                self.model = None
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
                result = orig_setup(self, cfg, tcfg)
        except BaseException:
            if plan:
                _restore_quietly(getattr(self, "model", None))
            raise
        finally:
            _PLAN.reset(token)
        self._soup_fast = None
        if plan:
            _log("model loaded" + _load_note())
            self._soup_fast = _build_runner(self, cfg, tcfg)
        return result

    @functools.wraps(orig_train)
    def train(self, *args, **kwargs):
        runner = getattr(self, "_soup_fast", None)
        trainer = getattr(self, "trainer", None)
        if runner is not None:
            from . import precompile
            precompile.wait()
            supported = trainer is not None and _trainer_supported(trainer)
            if not supported:
                _log("fast path not installed: trainer settings differ from the task")
            installed = supported and _runner_ready(runner) and _install_on_trainer(trainer, runner)
            if not installed:
                runner.restore_stock()
                self._soup_fast = None
            _log("training starts")
            try:
                return orig_train(self, *args, **kwargs)
            finally:
                if installed:
                    try:
                        runner.flush()
                    except Exception:  # noqa: BLE001
                        pass
                    undo = getattr(trainer, "_soup_undo_traversals", None)
                    if undo is not None:
                        undo()
                _log("training, final save and Soup's wrap-up done")
        return orig_train(self, *args, **kwargs)

    _setup_transformers._soup_fast = True
    train._soup_fast = True
    cls._setup_transformers = _setup_transformers
    cls.train = train


def _load_note():
    parts = []
    if _LOAD["warmup_skipped"]:
        parts.append("allocator warm-up skipped")
    if _LOAD.get("embedding_pinned"):
        parts.append("embedding read into page-locked memory (fp16 conversion on the GPU)")
    elif _LOAD["embedding_cpu"]:
        parts.append("embedding loaded on the CPU")
    if _LOAD["host_weights"]:
        parts.append(f"{_LOAD['host_weights']} 4-bit weights moved to host after quantisation "
                     f"({_LOAD['host_bytes'] / 2**20:.0f} MiB)")
    if _LOAD["error"]:
        parts.append(f"load hook error: {_LOAD['error']}")
    try:
        import torch

        parts.append(f"GPU peak so far {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
                     f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved")
    except Exception:  # noqa: BLE001
        pass
    return (": " + "; ".join(parts)) if parts else ""


def _restore_quietly(model):
    if model is None:
        return
    try:
        import torch

        from .fast_qwen import restore_to_gpu

        restore_to_gpu(model, torch.device("cuda", torch.cuda.current_device()))
    except Exception as exc:  # noqa: BLE001
        _log(f"could not move the model back to the GPU: {type(exc).__name__}: {exc}")


def _build_runner(wrapper, cfg, tcfg):
    import torch

    model = getattr(wrapper, "model", None)
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "get_base_model"):
        _restore_quietly(model)
        return None
    started = time.perf_counter()
    try:
        from .fast_qwen import FastQwen2

        seed = getattr(tcfg, "seed", None)
        runner = FastQwen2(model, seed=0 if seed is None else int(seed), background=True)
    except Exception as exc:  # noqa: BLE001 - any mismatch: keep the stock path
        _log(f"fast path disabled: {type(exc).__name__}: {exc}")
        _restore_quietly(model)
        return None
    runner._t_build = started
    # transformers 5 sets no hf_device_map for a single-device load, so accelerate's prepare_model
    # would call model.to("cuda") and copy the host-resident 4-bit weights and the embedding to
    # the GPU (+1.07 GB of unused duplicates in the v6 run). The map is the documented marker of
    # a model that manages its own placement (Soup's layer-stream runtime sets it the same way).
    try:
        base = model.get_base_model()
        if getattr(base, "hf_device_map", None) is None:
            index = runner.device.index
            base.hf_device_map = {"": torch.cuda.current_device() if index is None else int(index)}
    except Exception as exc:  # noqa: BLE001 - the first-step check still reclaims the copies
        _log(f"hf_device_map not set: {type(exc).__name__}: {exc}")
    _log(f"fast path validated in {time.perf_counter() - started:.2f} s; placement and repacking "
         f"continue in the background while Soup builds the trainer")
    return runner


def _runner_ready(runner):
    """Join the background build; False (after putting the model back on the GPU) if it failed."""
    import torch

    try:
        runner.ready()
    except Exception as exc:  # noqa: BLE001
        _log(f"fast path disabled (build failed: {type(exc).__name__}: {exc})")
        runner.restore_stock()
        return False
    _log(f"fast path ready (GPU {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated, peak so far "
         f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated / "
         f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved); {runner.placement()}; "
         f"attention: {runner.attn_note}; dequant overlap: {runner.overlap}; "
         f"self-test: {runner.test_note}")
    return True


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


def _cache_traversals(model):
    """HF Trainer walks the whole model at every step: model.parameters() for gradient clipping
    (twice, inside accelerate) and model.zero_grad(). The parameter set cannot change during
    training, and only the trainable (LoRA) parameters can hold a gradient, so both use lists
    taken once: the same parameters in the same order; zero_grad(set_to_none=True) clears exactly
    the gradients the full walk would. Returns an undo function, or None if not applicable."""
    params = list(model.parameters())
    trainable = [p for p in params if p.requires_grad]
    if not trainable or any(p.grad is not None for p in params if not p.requires_grad):
        return None
    orig_parameters, orig_zero_grad = model.parameters, model.zero_grad
    frozen_ids = {id(p) for p in params if not p.requires_grad}

    def parameters(recurse=True):
        if not recurse:
            return orig_parameters(recurse)
        return iter(params)

    def zero_grad(set_to_none=True):
        if not set_to_none:
            return orig_zero_grad(set_to_none)
        for p in trainable:
            p.grad = None
        return None

    object.__setattr__(model, "parameters", parameters)
    object.__setattr__(model, "zero_grad", zero_grad)

    def undo():
        model.__dict__.pop("parameters", None)
        model.__dict__.pop("zero_grad", None)

    undo.frozen = frozen_ids
    return undo


def _install_on_trainer(trainer, runner):
    if not _trainer_supported(trainer):
        _log("fast path not installed: trainer settings differ from the task")
        return False
    original = trainer.compute_loss
    trainer._soup_orig_compute_loss = original
    trainer._soup_fast_runner = runner
    trainer.compute_loss = types.MethodType(fast_compute_loss, trainer)
    # the step's metrics are recorded at the end of its backward; whatever happens, they are in
    # place before the trainer averages and logs them
    orig_log = trainer.log

    @functools.wraps(orig_log)
    def log(*args, **kwargs):
        try:
            runner.flush()
        except Exception:  # noqa: BLE001
            pass
        return orig_log(*args, **kwargs)

    trainer.log = log
    # timeline of the wrap-up (and the setup profiler's phase switch); nothing else changes
    try:
        import transformers

        class _Timeline(transformers.TrainerCallback):
            def on_train_end(self, args, state, control, **kwargs):
                from . import hostmem, profile_setup
                _log("training loop done")
                profile_setup.phase("wrapup")
                if config.EARLY_RELEASE:
                    try:
                        hostmem.release_async(runner.device)   # unpin while Soup wraps up
                    except Exception:  # noqa: BLE001
                        pass

        trainer.add_callback(_Timeline())
        orig_save = trainer.save_model

        @functools.wraps(orig_save)
        def save_model(*args, **kwargs):
            _log("save_model starts")
            try:
                return orig_save(*args, **kwargs)
            finally:
                _log("save_model done")

        trainer.save_model = save_model
    except Exception as exc:  # noqa: BLE001 - logging only
        _log(f"timeline hooks not installed: {type(exc).__name__}: {exc}")
    if config.CACHED_TRAVERSALS:
        try:
            trainer._soup_undo_traversals = _cache_traversals(trainer.model)
        except Exception as exc:  # noqa: BLE001 - the full walks stay
            _log(f"traversal lists not installed: {type(exc).__name__}: {exc}")
    _log("fast compute_loss installed")
    return True


_INPUT_KEYS = {"input_ids", "attention_mask", "labels"}


def _record_metrics(trainer, runner, batch, stats_cpu, ready, step):
    """Append the step's TRL metrics and Soup's rewind record from the per-token stats."""
    from .fast_qwen import batch_metrics

    try:
        if ready is not None:
            ready.synchronize()
        m = batch_metrics(batch, stats_cpu)
        metrics = trainer._metrics["train"]
        metrics["entropy"].append(m["entropy"])
        metrics["mean_token_accuracy"].append(m["accuracy"])
        state = getattr(trainer, "_soup_rewind_state", None)
        if state is not None and not state.failed:
            try:
                state.on_micro_batch(step=step, row_loss=m["row_loss"], row_tokens=m["row_tokens"])
            except Exception as exc:  # noqa: BLE001 - the recorder never raises into training
                state.fail(exc)
    except Exception as exc:  # noqa: BLE001
        _log(f"metrics of step {step} not recorded: {type(exc).__name__}: {exc}")


def fast_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    runner = self._soup_fast_runner
    if (return_outputs or not getattr(model, "training", False) or not self.model.training
            or not set(inputs) <= _INPUT_KEYS | {"_prediction_loss_only"}
            or not _INPUT_KEYS <= set(inputs)):
        return _stock_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
    import torch

    from .fast_qwen import run_step

    runner.flush()
    if runner.steps == 0:
        from . import profile_setup
        profile_setup.phase(None)         # the setup profile ends at the first step
        profile_setup.report("setup")
        if config.GC_TUNE:
            import gc
            gc.freeze()                   # setup's objects never get scanned again by the GC
        try:
            freed = runner.reclaim_gpu_copies()
            if freed:
                _log(f"model weights had been copied to the GPU ({freed:.0f} MiB); moved back to "
                     f"the host buffers and freed")
        except Exception as exc:  # noqa: BLE001
            _log(f"placement check failed: {type(exc).__name__}: {exc}")
        from . import distmap, hostmem
        distmap.restore()
        try:
            # the build thread's self-test made its own cuBLAS handle and workspace; the training
            # threads re-create theirs at their next GEMM
            torch._C._cuda_clearCublasWorkspaces()
        except Exception:  # noqa: BLE001
            pass
        torch.cuda.empty_cache()          # the self-test's and the load's leftovers, before the peak
        if "stack" in _CTX:
            from . import cudactx
            _log(f"CUDA stack per thread (bytes): before, after apply() = {_CTX['stack']}; "
                 f"now {cudactx.stack_now()}")
        _log(f"first step starts (GPU {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated; "
             f"host buffers: {hostmem.describe()})")
    batch = runner.pack(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
    if batch is None:
        return _stock_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
    inputs.pop("_prediction_loss_only", None)
    if config.EMPTY_CACHE_ON_GROWTH and batch.n > getattr(runner, "_max_n", 0):
        runner._max_n = batch.n
        torch.cuda.empty_cache()        # the GPU is idle here (the harness synchronises each step)
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
    self._total_train_tokens += int(batch.n_real)
    self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
    step = int(self.state.global_step) + 1
    if config.DEFER_METRICS:
        host = torch.empty(stats.shape, dtype=stats.dtype, pin_memory=True)
        host.copy_(stats.detach(), non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        runner.defer(functools.partial(_record_metrics, self, runner, batch, host, ready, step))
    else:
        _record_metrics(self, runner, batch, stats.detach().to("cpu"), None, step)
    if runner.steps <= 3:
        _log(f"step {runner.steps}: {batch.B} rows, {batch.n_real} tokens packed of {batch.B * batch.Tm} "
             f"({batch.n_lab} supervised), forward queued in {time.perf_counter() - t0:.3f} s, "
             f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated / "
             f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved, "
             f"{runner.n_dequant} half-layer dequants so far, kept from forward: {_plan_text(runner)}")
        if runner.steps == 1:
            _log(f"host arenas: {runner.host_note}")
    return loss


def _stock_compute_loss(trainer, model, inputs, return_outputs, num_items_in_batch):
    """The original compute_loss. With the model off the GPU the stock forward cannot run, so the
    fast path is taken down first (everything back on the GPU, the original method reinstalled)."""
    runner = getattr(trainer, "_soup_fast_runner", None)
    if runner is not None and (runner.pipe is not None or runner.emb_on_host):
        _log("a batch the fast path does not handle: switching this run back to the stock path")
        try:
            runner.flush()
        except Exception:  # noqa: BLE001
            pass
        runner.restore_stock()
        trainer.compute_loss = trainer._soup_orig_compute_loss
        trainer._soup_fast_runner = None
    return trainer._soup_orig_compute_loss(model, inputs, return_outputs=return_outputs,
                                           num_items_in_batch=num_items_in_batch)


def _plan_text(runner):
    plan = getattr(runner, "last_plan", None)
    if plan is None:
        return "?"
    kept = [name for name, on in (("gate|up", plan.gu), ("attention", plan.attn),
                                  ("checkpoints on host", plan.x),
                                  ("MLP outputs (residual stream rebuilt backwards)",
                                   getattr(plan, "rx", False))) if on]
    return ", ".join(kept) or "nothing (recompute)"


# --------------------------------------------------------------------------- entry point
def install():
    if _STATE["installed"]:
        return
    _log("apply()")
    _patch_loading()
    _patch_prepare_kbit()
    _patch_wrapper()
    _STATE["installed"] = True
