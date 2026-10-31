"""Per-context CUDA settings that cost GPU memory without serving this run (v13).

Thread stack. The CUDA context reserves local memory for the call stack of every thread the GPU
can hold at once: stack size x 1024 threads x 40 SMs on a T4, i.e. 40 MiB at the default 1 KiB,
counted by nvidia-smi. None of the kernels of this run needs more than a few bytes of stack, and
the driver raises the per-thread stack size by itself for any kernel that does ("The driver
automatically increases the per-thread stack size for each kernel launch as needed", cuCtxSetLimit
documentation), so a small limit changes no result. Set once in apply(), on the context the
harness created, through the driver API (cuCtxSetLimit(CU_LIMIT_STACK_SIZE, ...)); the allocator
settings of the harness are not touched.
"""

CU_LIMIT_STACK_SIZE = 0x00


def _driver():
    import ctypes

    lib = ctypes.CDLL("libcuda.so.1")
    get = lib.cuCtxGetLimit
    get.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    get.restype = ctypes.c_int
    put = lib.cuCtxSetLimit
    put.argtypes = [ctypes.c_int, ctypes.c_size_t]
    put.restype = ctypes.c_int
    return ctypes, get, put


def shrink_stack(size):
    """Returns (before, after) bytes per thread, or a string saying why nothing changed."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return "CUDA not initialised"
        ctypes, get, put = _driver()
        before = ctypes.c_size_t(0)
        res = get(ctypes.byref(before), CU_LIMIT_STACK_SIZE)
        if res != 0:
            return f"cuCtxGetLimit returned {res}"
        if before.value <= size:
            return (before.value, before.value)
        res = put(CU_LIMIT_STACK_SIZE, int(size))
        if res != 0:
            return f"cuCtxSetLimit returned {res}"
        after = ctypes.c_size_t(0)
        get(ctypes.byref(after), CU_LIMIT_STACK_SIZE)
        return (before.value, after.value)
    except Exception as exc:  # noqa: BLE001 - the default stays
        return f"{type(exc).__name__}: {exc}"


def stack_now():
    try:
        ctypes, get, _ = _driver()
        v = ctypes.c_size_t(0)
        return v.value if get(ctypes.byref(v), CU_LIMIT_STACK_SIZE) == 0 else None
    except Exception:  # noqa: BLE001
        return None
