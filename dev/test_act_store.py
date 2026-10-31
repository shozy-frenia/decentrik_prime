"""CPU check of fast_qwen._ActStore (activations kept from the forward in pinned host arenas).

Models CUDA streams, events and the caching allocator (a freed block is handed to the next
allocation on the main stream immediately, which is what PyTorch does for a block without
record_stream), then drives the real _ActStore methods in the order FastQwen2.forward/backward use
them, with the main-stream kernels that read and write each tensor. It checks that every pair of
accesses to the same GPU block (at least one a write) issued in program order is also ordered on
the GPU by streams and events, that the store keeps at most IN_FLIGHT + 1 layers of saved
tensors alive in the forward, and that a mutated store (the GPU-side wait dropped) is caught.
    python dev/test_act_store.py
"""

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dev"))

OPS, ACCESS = [], []           # ops in enqueue order; (op, block, is_write)


class Op:
    def __init__(self, stream, what, deps):
        self.stream, self.what, self.deps = stream, what, deps


class Stream:
    def __init__(self, name):
        self.name, self.last, self.pending = name, None, []

    def add(self, what, reads=(), writes=()):
        op = Op(self, what, ([self.last] if self.last else []) + self.pending)
        self.pending = []
        self.last = op
        OPS.append(op)
        for b in reads:
            ACCESS.append((op, b, False))
        for b in writes:
            ACCESS.append((op, b, True))
        return op

    def wait_event(self, ev):
        if ev.op is not None:
            self.pending.append(ev.op)

    def wait_stream(self, other):
        if other.last is not None:
            self.pending.append(other.last)


MAIN = Stream("main")
CUR = [MAIN]


class Event:
    def __init__(self, *a, **k):
        self.op = None

    def record(self, stream=None):
        self.op = (stream or CUR[0]).add(("record",))


class _Ctx:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        self.prev, CUR[0] = CUR[0], self.s

    def __exit__(self, *a):
        CUR[0] = self.prev


class Allocator:
    def __init__(self):
        self.free, self.next_id, self.live = [], 0, 0
        self.peak = 0

    def alloc(self):
        self.live += 1
        self.peak = max(self.peak, self.live)
        if self.free:
            return self.free.pop()
        self.next_id += 1
        return self.next_id

    def release(self, block):
        self.live -= 1
        self.free.append(block)


ALLOC = Allocator()


class T:
    """A GPU tensor owning one allocator block."""

    def __init__(self, shape, dtype="f16"):
        self.block, self.shape, self.dtype = ALLOC.alloc(), tuple(shape), dtype

    def __del__(self):
        ALLOC.release(self.block)

    def copy_(self, src, non_blocking=False):
        CUR[0].add(("h2d",), writes=[self.block])
        return self


class HostSlice:
    def copy_(self, src, non_blocking=False):
        CUR[0].add(("d2h",), reads=[src.block])
        return self


class Arena:
    def __getitem__(self, idx):
        return HostSlice()


def kernel(reads=(), writes=()):
    MAIN.add(("kernel",), reads=[t.block for t in reads], writes=[t.block for t in writes])


def before(a, b):
    seen, todo = set(), [b]
    while todo:
        x = todo.pop()
        for d in x.deps:
            if d is a:
                return True
            if id(d) not in seen:
                seen.add(id(d))
                todo.append(d)
    return False


def install_fake_torch():
    import tl_emu
    tl_emu.install()
    torch = sys.modules["torch"]
    torch.no_grad = lambda: (lambda f: f)
    torch.autograd = types.SimpleNamespace(Function=object)
    torch.cuda = types.SimpleNamespace(Event=Event, Stream=lambda device=None: Stream("side"),
                                       stream=_Ctx, current_stream=lambda device=None: CUR[0])
    torch.empty = lambda shape, dtype=None, device=None: T(shape, dtype)


def run(store_cls, L=6, n=4, steps=2, save_x=True):
    model = types.SimpleNamespace(L=L, device=None, copy_stream=Stream("copy"),
                                  host={k: Arena() for k in ("x", "gu", "qkv", "out", "op")})
    alive_max = 0
    for _ in range(steps):
        st = store_cls(model, n)
        # ---- forward
        x = T((n, 8))
        kernel(writes=[x])
        for i in range(L):
            st.retire(i - st.IN_FLIGHT)
            if i > 0:
                x_new = T((n, 8))
                kernel(reads=[x], writes=[x_new])
                x = x_new
            st.put(i, "x", x, save_x)
            qkv = T((n, 16))
            kernel(writes=[qkv])
            out = T((n, 12))
            kernel(reads=[qkv], writes=[out])
            op = T((n, 8))
            kernel(reads=[out], writes=[op])
            st.put(i, "qkv", qkv, True)
            st.put(i, "out", out, True)
            st.put(i, "op", op, True)
            del qkv, out
            h1 = T((n, 8))
            kernel(reads=[x, op], writes=[h1])
            del op
            gu = T((n, 64))
            kernel(writes=[gu])
            act = T((n, 32))
            kernel(reads=[gu], writes=[act])
            st.put(i, "gu", gu, True)
            del gu
            d16 = T((n, 8))
            kernel(reads=[act], writes=[d16])
            del act
            alive_max = max(alive_max, len(st.inflight))
            x = h1
            kernel(reads=[d16])
            del d16
        st.retire(L)
        del x
        MAIN.add(("head",))
        # ---- backward
        for i in reversed(range(L)):
            x = st.get(i, "x")
            if i > 0:
                st.prefetch(i - 1, "x")
                st.prefetch(i - 1, "op")
            kernel(reads=[x])
            op = st.get(i, "op")
            h1 = T((n, 8))
            kernel(reads=[x, op], writes=[h1])
            del op
            gu = st.get(i, "gu")
            kernel(reads=[gu], writes=[gu])        # silu backward, in place
            kernel(reads=[gu])                      # dy2 GEMM
            del gu
            if i > 0:
                st.prefetch(i - 1, "gu")
            kernel(reads=[h1])
            del h1
            qkv = st.get(i, "qkv")
            out = st.get(i, "out")
            kernel(reads=[qkv, out])                # attention backward
            del out
            kernel(writes=[qkv])                    # epilogue backward into the QKV buffer
            if i > 0:
                st.prefetch(i - 1, "qkv")
                st.prefetch(i - 1, "out")
            kernel(reads=[qkv, x])                  # dy1 GEMM, LoRA grads, norm-1 backward
            del qkv, x
        st.close()
        MAIN.add(("optimizer",))
    bad = 0
    for k, (op_a, blk, w_a) in enumerate(ACCESS):
        for op_b, blk_b, w_b in ACCESS[k + 1:]:
            if blk_b != blk or not (w_a or w_b) or op_a is op_b:
                continue
            if not before(op_a, op_b):
                bad += 1
    return bad, alive_max


def main():
    install_fake_torch()
    from solution import fast_qwen

    Store = fast_qwen._ActStore
    bad, alive = run(Store)
    print(f"_ActStore: unordered conflicting accesses {bad}, layers of saved tensors in flight <= {alive}, "
          f"peak live blocks {ALLOC.peak}")
    assert bad == 0 and alive <= Store.IN_FLIGHT + 1

    class NoWait(Store):                    # mutation: release without the GPU-side wait
        def retire(self, upto):
            for j in [k for k in self.inflight if k <= upto]:
                del self.inflight[j]

    OPS.clear()
    ACCESS.clear()
    bad2, _ = run(NoWait)
    print(f"mutated store (no wait before release): unordered conflicting accesses {bad2}")
    assert bad2 > 0
    print("ACTIVATION STORE ORDERING OK")


if __name__ == "__main__":
    main()
