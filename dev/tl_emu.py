"""A tiny NumPy emulator of the subset of triton / triton.language used by solution/k_*.py.

It runs a @triton.jit function body program-by-program on CPU with numpy arrays, so kernel
indexing, masking and math can be checked without a GPU. Install with `install()` before
importing the kernel modules (it registers fake `triton`, `triton.language` and a minimal
`torch` stand-in if torch is missing).
"""

import math
import sys
import types

import numpy as np

_PID = [0, 0, 0]


class T:
    """Block tensor (wraps a numpy array / scalar)."""
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = np.asarray(a)

    # conversions -------------------------------------------------------
    def to(self, dtype, bitcast=False):
        return T(self.a.astype(dtype.np))

    @property
    def shape(self):
        return self.a.shape

    @property
    def dtype(self):
        return _BY_NP[self.a.dtype.type]

    def __index__(self):
        return int(self.a)

    def __int__(self):
        return int(self.a)

    def __float__(self):
        return float(self.a)

    def __bool__(self):
        assert self.a.size == 1
        return bool(self.a)

    def __getitem__(self, idx):
        return T(self.a[idx])

    # arithmetic --------------------------------------------------------
    def _b(self, other, op):
        o = other.a if isinstance(other, T) else other
        return T(op(self.a, o))

    def _r(self, other, op):
        o = other.a if isinstance(other, T) else other
        return T(op(o, self.a))

    __add__ = lambda s, o: s._b(o, np.add)
    __radd__ = lambda s, o: s._r(o, np.add)
    __sub__ = lambda s, o: s._b(o, np.subtract)
    __rsub__ = lambda s, o: s._r(o, np.subtract)
    __mul__ = lambda s, o: s._b(o, np.multiply)
    __rmul__ = lambda s, o: s._r(o, np.multiply)
    __truediv__ = lambda s, o: s._b(o, _div)
    __rtruediv__ = lambda s, o: s._r(o, _div)
    __floordiv__ = lambda s, o: s._b(o, _floordiv)
    __rfloordiv__ = lambda s, o: s._r(o, _floordiv)
    __mod__ = lambda s, o: s._b(o, np.remainder)
    __rshift__ = lambda s, o: s._b(o, np.right_shift)
    __lshift__ = lambda s, o: s._b(o, np.left_shift)
    __and__ = lambda s, o: s._b(o, np.bitwise_and)
    __rand__ = lambda s, o: s._r(o, np.bitwise_and)
    __or__ = lambda s, o: s._b(o, np.bitwise_or)
    __xor__ = lambda s, o: s._b(o, np.bitwise_xor)
    __lt__ = lambda s, o: s._b(o, np.less)
    __le__ = lambda s, o: s._b(o, np.less_equal)
    __gt__ = lambda s, o: s._b(o, np.greater)
    __ge__ = lambda s, o: s._b(o, np.greater_equal)
    __eq__ = lambda s, o: s._b(o, np.equal)
    __ne__ = lambda s, o: s._b(o, np.not_equal)
    __neg__ = lambda s: T(-s.a)
    __invert__ = lambda s: T(~s.a)
    __hash__ = None


def _div(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.issubdtype(np.result_type(a, b), np.integer):
        raise TypeError("integer true division in kernel")
    return np.true_divide(a, b)


def _floordiv(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.issubdtype(np.result_type(a, b), np.integer):
        q = np.trunc(a / b).astype(np.result_type(a, b))   # C semantics
        return q
    return np.floor_divide(a, b)


class Ptr:
    def __init__(self, buf, off):
        self.buf, self.off = buf, off

    def __add__(self, o):
        o = o.a if isinstance(o, T) else o
        return Ptr(self.buf, np.asarray(self.off) + np.asarray(o).astype(np.int64))

    __radd__ = __add__

    def __sub__(self, o):
        o = o.a if isinstance(o, T) else o
        return Ptr(self.buf, np.asarray(self.off) - np.asarray(o).astype(np.int64))


class DType:
    def __init__(self, name, np_type):
        self.name, self.np = name, np_type

    def __repr__(self):
        return f"tl.{self.name}"


float16 = DType("float16", np.float16)
float32 = DType("float32", np.float32)
int32 = DType("int32", np.int32)
int64 = DType("int64", np.int64)
uint8 = DType("uint8", np.uint8)
uint32 = DType("uint32", np.uint32)
_BY_NP = {np.float16: float16, np.float32: float32, np.int32: int32, np.int64: int64,
          np.uint8: uint8, np.uint32: uint32}


def _un(x):
    return x.a if isinstance(x, T) else x


class constexpr:  # used only as an annotation
    pass


def program_id(axis):
    return T(np.int32(_PID[axis]))


def arange(start, end):
    return T(np.arange(start, end, dtype=np.int32))


def zeros(shape, dtype):
    return T(np.zeros(shape, dtype=dtype.np))


def full(shape, value, dtype):
    return T(np.full(shape, value, dtype=dtype.np))


def load(ptr, mask=None, other=None):
    off = np.asarray(ptr.off)
    buf = ptr.buf
    if mask is None:
        if off.size and (off.min() < 0 or off.max() >= buf.size):
            raise IndexError(f"unmasked out-of-bounds load {off.min()}..{off.max()} of {buf.size}")
        return T(buf[off])
    m = np.broadcast_to(_un(mask), np.broadcast_shapes(np.shape(_un(mask)), off.shape))
    off = np.broadcast_to(off, m.shape)
    safe = np.where(m, off, 0)
    if safe.size and (safe.min() < 0 or safe.max() >= buf.size):
        raise IndexError(f"masked-in out-of-bounds load {safe.min()}..{safe.max()} of {buf.size}")
    vals = buf[safe]
    fill = 0 if other is None else _un(other)
    return T(np.where(m, vals, np.asarray(fill).astype(buf.dtype)))


def store(ptr, val, mask=None):
    off = np.asarray(ptr.off)
    v = np.asarray(_un(val))
    shape = np.broadcast_shapes(off.shape, v.shape, () if mask is None else np.shape(_un(mask)))
    off = np.broadcast_to(off, shape)
    v = np.broadcast_to(v, shape).astype(ptr.buf.dtype)
    if mask is None:
        m = np.ones(shape, dtype=bool)
    else:
        m = np.broadcast_to(_un(mask), shape)
    sel = off[m]
    if sel.size and (sel.min() < 0 or sel.max() >= ptr.buf.size):
        raise IndexError(f"out-of-bounds store {sel.min()}..{sel.max()} of {ptr.buf.size}")
    ptr.buf[sel] = v[m]


def where(c, a, b):
    a, b = _un(a), _un(b)
    if isinstance(a, float) and not isinstance(b, float):
        a = np.asarray(a, dtype=np.asarray(b).dtype)
    if isinstance(b, float) and not isinstance(a, float):
        b = np.asarray(b, dtype=np.asarray(a).dtype)
    return T(np.where(_un(c), a, b))


def _red(fn):
    def r(x, axis=None, **kw):
        return T(fn(_un(x), axis=axis))
    return r


sum = _red(np.sum)
max = _red(np.max)
min = _red(np.min)


def argmax(x, axis, tie_break_left=True):
    return T(np.argmax(_un(x), axis=axis).astype(np.int32))


def _f(fn):
    def g(x):
        a = _un(x)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return T(fn(a).astype(np.asarray(a).dtype if np.asarray(a).dtype.kind == "f" else np.float32))
    return g


exp = _f(np.exp)
exp2 = _f(np.exp2)
log = _f(np.log)
log2 = _f(np.log2)
sqrt = _f(np.sqrt)
rsqrt = _f(lambda a: 1.0 / np.sqrt(a))
sigmoid = _f(lambda a: 1.0 / (1.0 + np.exp(-a)))
cos = _f(np.cos)
sin = _f(np.sin)


def maximum(a, b):
    return T(np.maximum(_un(a), _un(b)))


def minimum(a, b):
    return T(np.minimum(_un(a), _un(b)))


def dot(a, b, acc=None, **kw):
    r = _un(a).astype(np.float32) @ _un(b).astype(np.float32)
    if acc is not None:
        r = r + _un(acc)
    return T(r.astype(np.float32))


def trans(x, *dims):
    return T(np.swapaxes(_un(x), -1, -2))


def join(a, b):
    return T(np.stack([_un(a), _un(b)], axis=-1))


def reshape(x, *shape, can_reorder=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return T(np.reshape(_un(x), shape))


def rand(seed, offset):
    off = np.asarray(_un(offset)).astype(np.uint64)
    s = np.uint64(int(_un(seed)) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        x = off * np.uint64(0x9E3779B97F4A7C15) + s * np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(29)
    return T(((x >> np.uint64(40)).astype(np.float64) / float(1 << 24)).astype(np.float32))


def cdiv(a, b):
    return -(-a // b)


def next_power_of_2(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


class Kernel:
    def __init__(self, fn, **opts):
        self.fn = fn
        self.opts = opts
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kw = {k: v for k, v in kwargs.items() if k not in ("num_warps", "num_stages", "enable_fp_fusion")}
            conv = [Ptr(a.reshape(-1), 0) if isinstance(a, np.ndarray) else a for a in args]
            g = tuple(grid) + (1,) * (3 - len(grid))
            self.launches += 1
            for z in range(g[2]):
                for y in range(g[1]):
                    for x in range(g[0]):
                        _PID[:] = [x, y, z]
                        self.fn(*conv, **kw)
        return launch

    def warmup(self, *a, **k):
        return None


def jit(fn=None, **opts):
    if fn is None:
        return lambda f: Kernel(f, **opts)
    return Kernel(fn)


def install():
    tl = types.ModuleType("triton.language")
    for name, val in list(globals().items()):
        if not name.startswith("_") and name not in ("install", "Kernel", "jit", "np", "math", "sys", "types"):
            setattr(tl, name, val)
    tri = types.ModuleType("triton")
    tri.jit = jit
    tri.language = tl
    tri.cdiv = cdiv
    tri.next_power_of_2 = next_power_of_2
    sys.modules["triton"] = tri
    sys.modules["triton.language"] = tl
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            fake = types.ModuleType("torch")
            for n in ("float16", "float32", "int32", "int64", "uint8"):
                setattr(fake, n, n)
            sys.modules["torch"] = fake
    return tri, tl
