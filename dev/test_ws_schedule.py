"""CPU check of the dequantisation schedule in fast_qwen.FastQwen2 (no GPU, no torch needed).

Runs the real _ws_load / _ws_use / _ws_release methods against a model of CUDA streams and
events: every kernel is an operation on a stream; an event wait makes a stream's later operations
come after the waited operation. It then checks, for several steps of forward + backward:
  * every GEMM reads the workspace part of the layer it expects (no read of stale weights),
  * every dequant writes a part only after the last GEMM that read the previous contents
    (no overwrite while in use), with both orders derived from the stream/event graph,
  * the number of dequantised halves per step (54 layers instead of 56).
    python dev/test_ws_schedule.py
"""

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dev"))


# ---------------------------------------------------------------- a model of streams and events
class Op:
    def __init__(self, stream, what, deps):
        self.stream, self.what, self.deps = stream, what, deps


class Stream:
    def __init__(self, name):
        self.name, self.last, self.pending = name, None, []

    def add(self, what):
        deps = ([self.last] if self.last else []) + self.pending
        self.pending = []
        op = Op(self, what, deps)
        self.last = op
        OPS.append(op)
        return op

    def wait_event(self, ev):
        if ev.op is not None:
            self.pending.append(ev.op)

    def wait_stream(self, other):
        if other.last is not None:
            self.pending.append(other.last)


class Event:
    def __init__(self, *a, **k):
        self.op = None

    def record(self, stream=None):
        stream = stream or CUR[0]
        self.op = stream.add(("record",))


OPS = []
MAIN = Stream("main")
CUR = [MAIN]


class _StreamCtx:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        self.prev = CUR[0]
        CUR[0] = self.s

    def __exit__(self, *a):
        CUR[0] = self.prev


def before(a, b):
    """True if op a is ordered before op b by the stream/event graph."""
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
    torch.Tensor = object
    cuda = types.SimpleNamespace(
        Event=Event, Stream=lambda device=None: Stream("side"),
        stream=_StreamCtx, current_stream=lambda device=None: CUR[0])
    torch.cuda = cuda


def main():
    install_fake_torch()
    from solution import fast_qwen

    L, SPLIT, NQB = 6, 1024, 4096
    m = object.__new__(fast_qwen.FastQwen2)
    m.streamer, m.overlap, m.split_q, m.n_dequant = None, True, SPLIT, 0
    m.ws_have = {"attn": None, "mlp": None}
    m.ev_free = {"attn": Event(), "mlp": Event()}
    m.ev_ready = {"attn": Event(), "mlp": Event()}
    m.dq_stream = Stream("side")
    m.device = None
    m.packed = [types.SimpleNamespace(n_qblocks=NQB, layer=i) for i in range(L)]
    m.code2 = m.nf4 = m.ws = None
    writes = []         # (op, part, layer)
    reads = []          # (op, part, layer expected)

    def fake_dequant_range(layer, q0, q1, code2, nf4, out):
        part = "attn" if q0 == 0 else "mlp"
        assert (q0, q1) in ((0, SPLIT), (SPLIT, NQB)), (q0, q1)
        writes.append((CUR[0].add(("dequant", part, layer.layer)), part, layer.layer))

    fast_qwen.dequant_range = fake_dequant_range

    def gemm(part, i):
        reads.append((CUR[0].add(("gemm", part, i)), part, i))

    for overlap in (True, False):
        m.overlap = overlap
        per_step = []
        for step in range(3):
            d0 = m.n_dequant
            if m.overlap:
                m.dq_stream.wait_stream(MAIN)
            for i in range(L):                                  # forward
                nxt = i + 1 if i + 1 < L else None
                m._ws_use("attn", i)
                gemm("attn", i)
                gemm("attn", i)
                m._ws_release("attn", nxt)
                MAIN.add(("norm",))
                m._ws_use("mlp", i)
                gemm("mlp", i)
                gemm("mlp", i)
                m._ws_release("mlp", nxt)
            MAIN.add(("head",))
            for i in reversed(range(L)):                        # backward
                prv = i - 1 if i > 0 else None
                m._ws_use("attn", i)
                gemm("attn", i)                                 # recompute qkv, o
                m._ws_use("mlp", i)
                gemm("mlp", i)                                  # recompute gu, d^T, gu^T
                m._ws_release("mlp", prv)
                gemm("attn", i)                                 # o^T, qkv^T
                m._ws_release("attn", prv)
                MAIN.add(("lora",))
            MAIN.add(("optimizer",))
            per_step.append(m.n_dequant - d0)
        # hazards: for each read, the latest write of that part ordered before it must be the
        # expected layer, and no write of that part may be unordered with the read
        bad = 0
        for rop, part, layer in reads:
            prior = [w for w in writes if w[1] == part and before(w[0], rop)]
            racing = [w for w in writes if w[1] == part and not before(w[0], rop) and not before(rop, w[0])]
            last = prior[-1] if prior else None
            if last is None or last[2] != layer or racing:
                bad += 1
        print(f"overlap={overlap}: dequantised halves per step {per_step} (layers {L}), "
              f"{len(reads)} weight reads, hazards: {bad}")
        assert bad == 0
        assert per_step[0] == 4 * L - 2 and all(x == 4 * L - 4 for x in per_step[1:]), per_step
        OPS.clear()
        reads.clear()
        writes.clear()
        m.ws_have = {"attn": None, "mlp": None}
        m.ev_free = {"attn": Event(), "mlp": Event()}
        m.ev_ready = {"attn": Event(), "mlp": Event()}
    print("WORKSPACE SCHEDULE OK")


if __name__ == "__main__":
    main()
