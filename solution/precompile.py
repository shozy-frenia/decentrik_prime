"""Ahead-of-time compilation of every Triton kernel of the fast path, in background threads.

Triton compiles a kernel the first time it is launched; for a dozen kernels that is several
seconds of the measured wall time (plus a gcc build of its C launcher at the first launch). apply()
starts the compilation right away, in daemon threads,
using Triton's own `JITFunction.warmup` with the exact dtypes, constexprs and launch options the
real launches use (all runtime integers are `do_not_specialize`, pointers are 16-byte aligned on
both sides), so each later launch finds its binary in the in-memory kernel cache. The heavy
phases of Triton's compiler (MLIR passes, LLVM, ptxas) release the GIL, so this overlaps with
Soup importing its stack, tokenising the data and loading/quantising the model. `wait()` joins
the threads before the first training step. Nothing is cached on disk beyond what Triton does
for any run (the harness gives every run an empty TRITON_CACHE_DIR).
"""

import atexit
import sys
import threading
import time

from . import config

_threads = []
_errors = []
_timing = {}


def _compile(specs):
    for kernel, args, kwargs in specs:
        name = getattr(getattr(kernel, "fn", None), "__name__", str(kernel))
        start = time.perf_counter()
        try:
            compiled = kernel.warmup(*args, grid=(1,), **kwargs)
            # Triton builds each kernel's C launcher (a gcc run) and loads the cubin only at the
            # first real launch (CompiledKernel._init_handles); doing it here keeps ~0.3 s per
            # kernel signature out of the first training step.
            init = getattr(compiled, "_init_handles", None)
            if init is not None:
                init()
        except Exception as exc:  # noqa: BLE001 - the real launch will compile (and report) it
            _errors.append(f"{name}: {type(exc).__name__}: {exc}")
        _timing[name] = round(time.perf_counter() - start, 2)


def start():
    if not config.PRECOMPILE or _threads:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        from .fast_qwen import warmup_specs

        specs = warmup_specs()
    except Exception as exc:  # noqa: BLE001
        _errors.append(f"precompile setup: {type(exc).__name__}: {exc}")
        return
    # heaviest kernels first, spread round-robin over the threads
    order = sorted(specs, key=lambda s: -_weight(s[0]))
    n = max(1, int(config.PRECOMPILE_THREADS))
    buckets = [order[i::n] for i in range(n)]
    for b in buckets:
        t = threading.Thread(target=_compile, args=(b,), daemon=True, name="soup-precompile")
        t.start()
        _threads.append(t)
    # never let the interpreter shut down while a thread is inside the compiler
    atexit.register(_join_quietly)


def _join_quietly():
    for t in _threads:
        t.join(120.0)


def _weight(kernel):
    name = getattr(getattr(kernel, "fn", None), "__name__", "")
    return {"attn_bwd_dkdv_kernel": 9, "attn_bwd_dq_kernel": 8, "attn_fwd_kernel": 7,
            "lm_head_rows_kernel": 6, "qkv_epilogue_bwd_kernel": 5}.get(name, 1)


def wait(timeout=300.0):
    start = time.perf_counter()
    for t in _threads:
        t.join(max(0.0, timeout - (time.perf_counter() - start)))
    if config.VERBOSE and _threads:
        waited = time.perf_counter() - start
        print(f"[solution] precompiled {len(_timing)} kernels (waited {waited:.2f} s): {_timing}",
              file=sys.stderr, flush=True)
        for err in _errors:
            print(f"[solution] precompile warning: {err}", file=sys.stderr, flush=True)
