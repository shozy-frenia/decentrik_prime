"""Pinned host memory for everything the fast path keeps off the GPU, allocated early.

With the default configuration the GPU holds no model weights between uses: the repacked 4-bit
layers (680 MB), the fp16 embedding (467 MB) and the per-layer activations kept for the backward
(about 2.9 GB for a 2048-token batch) live in pinned (page-locked) host memory, which the GPU's
copy engines read and write at PCIe speed while the kernels run. cudaHostAlloc of ~4 GB takes a
noticeable fraction of a second, so `start()` (called from apply()) allocates the buffers in a
background thread while Soup imports its stack, reads the data and loads the model; `take()`
hands a buffer over when the fast path is built (waiting for the thread if it is not done) or
allocates a fresh one if the requested shape differs. The thread is joined at interpreter exit so
the process never shuts down while it is inside the driver.
"""

import atexit
import threading
import time

_POOL = {}
_THREAD = [None]
NOTE = {}


def start(specs):
    """specs: [(name, shape, torch dtype)], allocated in this order in one daemon thread."""
    if _THREAD[0] is not None or not specs:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except Exception:  # noqa: BLE001
        return

    def work():
        for name, shape, dtype in specs:
            started = time.perf_counter()
            try:
                _POOL[name] = torch.empty(shape, dtype=dtype, pin_memory=True)
            except Exception as exc:  # noqa: BLE001 - take() allocates (or the caller falls back)
                _POOL[name] = exc
            NOTE[name] = round(time.perf_counter() - started, 3)

    thread = threading.Thread(target=work, daemon=True, name="soup-pinned-pool")
    thread.start()
    _THREAD[0] = thread
    atexit.register(wait)


def wait():
    thread = _THREAD[0]
    if thread is not None:
        thread.join()


def take(name, shape, dtype):
    """The pre-allocated pinned tensor `name` if it has exactly this shape and dtype, otherwise a
    newly allocated pinned tensor (an allocation failure raises: callers fall back to the GPU)."""
    import torch

    wait()
    t = _POOL.pop(name, None)
    if isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(shape) and t.dtype == dtype:
        return t
    return torch.empty(tuple(shape), dtype=dtype, pin_memory=True)


def pinned_bytes():
    """Bytes currently held by the pool (not yet taken)."""
    import torch

    return sum(t.numel() * t.element_size() for t in _POOL.values() if isinstance(t, torch.Tensor))
