"""Pinned (page-locked) host memory for everything the fast path keeps off the GPU.

With the default configuration the GPU holds no model weights between uses: the repacked 4-bit
layers (680 MB), the fp16 embedding (467 MB) and the per-layer activations kept for the backward
(about 2.9 GB for a 2048-token batch) live in page-locked host memory, which the GPU's copy engines
read and write at PCIe speed while the kernels run.

How the memory is page-locked (v8). PyTorch's pinned allocator (cudaHostAlloc behind a caching
allocator) rounds every request up to a power of two: the default buffers (3.76 GiB) would pin
4.75 GiB, and both pinning at start-up and unpinning at process exit take time in proportion. So
each buffer is instead an exact-size anonymous mapping (huge pages where the kernel allows,
excluded from fork()), its pages are touched once in the allocating thread, and the range is
registered with the CUDA driver (cudaHostRegister). For the copy engines this is the same
page-locked memory as cudaHostAlloc's. If registration fails for any reason, PyTorch's pinned
allocator is used for that buffer and every later one.

`start()` (called from apply()) allocates the buffers in a background thread while Soup imports
its stack, reads the data and loads the model; `take()` hands a buffer over when the fast path is
built (waiting for the thread if it is not done) or allocates a fresh one if the requested shape
differs. `release_async()` (called when the training loop has finished) unregisters our buffers
in a background thread while Soup saves the adapter and wraps up: they stay valid as ordinary
host memory, and the driver has nothing left to unpin when the process exits. Every thread is
joined at interpreter exit, so the process never shuts down while one is inside the driver.
"""

import atexit
import threading
import time

_POOL = {}
_THREAD = [None]
NOTE = {}
_REG = []            # [uint8 tensor, mmap, registered] for every buffer we registered ourselves
_STATE = {"mode": "register", "why": None, "release": None, "released_s": None, "bytes": 0,
          "torch_bytes": 0}
_LOCK = threading.Lock()
_HUGE = 2 << 20


def _register(nbytes):
    """Exact-size page-locked uint8 buffer of nbytes (an anonymous mapping registered with the
    driver). Raises on any failure (the caller falls back)."""
    import mmap

    import torch

    size = max(_HUGE, -(-int(nbytes) // _HUGE) * _HUGE)
    mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    for name in ("MADV_DONTFORK", "MADV_HUGEPAGE"):
        advice = getattr(mmap, name, None)
        if advice is not None:
            try:
                mm.madvise(advice)
            except OSError:
                pass
    buf = torch.frombuffer(mm, dtype=torch.uint8)
    buf.numpy()[::4096] = 0                      # fault every page in here, not inside the driver
    err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), size, 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister returned error {int(err)}")
    rec = [buf, mm, True]
    with _LOCK:
        _REG.append(rec)
    if not buf.is_pinned():
        _unregister(rec)
        raise RuntimeError("registered memory is not reported as page-locked")
    _STATE["bytes"] += size
    return buf[:nbytes]


def _unregister(rec):
    import torch

    if rec[2]:
        rec[2] = False
        try:
            torch.cuda.cudart().cudaHostUnregister(rec[0].data_ptr())
        except Exception:  # noqa: BLE001 - the driver unpins it at exit anyway
            pass


def alloc(nbytes):
    """A page-locked uint8 tensor of exactly nbytes (an allocation failure raises: callers fall
    back to the GPU)."""
    import torch

    if _STATE["mode"] == "register" and not _STATE["release"]:
        try:
            return _register(nbytes)
        except Exception as exc:  # noqa: BLE001 - PyTorch's pinned allocator from now on
            _STATE["mode"] = "torch"
            _STATE["why"] = f"{type(exc).__name__}: {exc}"
    t = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
    _STATE["torch_bytes"] += int(nbytes)
    return t


def empty(shape, dtype):
    """A page-locked tensor of this shape and dtype."""
    import math

    import torch

    numel = math.prod(int(s) for s in shape)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    return alloc(nbytes).view(dtype).view(tuple(int(s) for s in shape))


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
                _POOL[name] = empty(shape, dtype)
            except Exception as exc:  # noqa: BLE001 - take() allocates (or the caller falls back)
                _POOL[name] = exc
            NOTE[name] = round(time.perf_counter() - started, 3)

    thread = threading.Thread(target=work, daemon=True, name="soup-pinned-pool")
    thread.start()
    _THREAD[0] = thread
    atexit.register(_at_exit)


def wait():
    thread = _THREAD[0]
    if thread is not None:
        thread.join()


def take(name, shape, dtype):
    """The pre-allocated page-locked tensor `name` if it has exactly this shape and dtype,
    otherwise a newly allocated one (an allocation failure raises: callers fall back)."""
    import torch

    wait()
    t = _POOL.pop(name, None)
    if isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(shape) and t.dtype == dtype:
        return t
    return empty(shape, dtype)


def pinned_bytes():
    """Bytes currently held by the pool (not yet taken)."""
    import torch

    return sum(t.numel() * t.element_size() for t in _POOL.values() if isinstance(t, torch.Tensor))


def describe():
    """One phrase for the log: how the host buffers were page-locked."""
    reg = _STATE["bytes"] / 2**30
    tor = _STATE["torch_bytes"] / 2**30
    parts = []
    if reg:
        parts.append(f"{reg:.2f} GiB registered (exact size)")
    if tor:
        parts.append(f"{tor:.2f} GiB from PyTorch's pinned allocator")
    if _STATE["why"]:
        parts.append(f"registration unavailable: {_STATE['why']}")
    return ", ".join(parts) or "none"


def release_async(device=None):
    """Training is over: unregister our buffers in a background thread (they remain valid host
    memory), after the GPU has finished every queued copy."""
    if _STATE["release"] is not None:
        return
    with _LOCK:
        recs = [r for r in _REG if r[2]]
    if not recs:
        _STATE["release"] = "nothing"
        return

    def work():
        started = time.perf_counter()
        try:
            import torch

            torch.cuda.synchronize(device)
            for rec in recs:
                _unregister(rec)
        except Exception:  # noqa: BLE001 - the driver unpins at exit anyway
            pass
        _STATE["released_s"] = round(time.perf_counter() - started, 2)

    t = threading.Thread(target=work, daemon=True, name="soup-pinned-release")
    _STATE["release"] = t
    t.start()


def _at_exit():
    wait()
    t = _STATE["release"]
    if isinstance(t, threading.Thread):
        t.join()
