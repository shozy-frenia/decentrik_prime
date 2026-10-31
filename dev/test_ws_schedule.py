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

    L, SPLIT, SPLIT_D, NQB = 6, 1024, 3072, 4096
    m = object.__new__(fast_qwen.FastQwen2)
    m.pipe, m.overlap, m.split_q, m.n_dequant = None, True, SPLIT, 0
    m.split_d = SPLIT_D
    m.ws_have = {"attn": None, "mlp": None}
    m.ev_free = {"attn": Event(), "mlp": Event()}
    m.ev_ready = {"attn": Event(), "mlp": Event()}
    m.dq_stream = Stream("side")
    m.device = None
    m.packed = [types.SimpleNamespace(n_qblocks=NQB, layer=i) for i in range(L)]
    m.code2 = m.nf4 = m.ws = None
    writes = []         # (op, region, (part, layer))
    reads = []          # (op, region, (part, layer) expected)
    parts = {(0, SPLIT): "attn", (SPLIT, NQB): "mlp", (SPLIT, SPLIT_D): "gu", (SPLIT_D, NQB): "dn"}

    def region(part):
        return "attn" if part == "attn" else "mlp"

    def fake_dequant_range(layer, q0, q1, code2, nf4, out, dst=None):
        part = parts[(q0, q1)]
        assert (dst is not None) == (part == "dn"), (part, dst)
        writes.append((CUR[0].add(("dequant", part, layer.layer)), region(part), (part, layer.layer)))

    fast_qwen.dequant_range = fake_dequant_range

    def gemm(part, i):
        reads.append((CUR[0].add(("gemm", part, i)), region(part), (part, i)))

    for overlap, shared in ((True, False), (False, False), (True, True), (False, True)):
        m.overlap, m.share_mlp = overlap, shared
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
                if shared:
                    m._ws_use("gu", i)
                    gemm("gu", i)                               # two row blocks of gate|up
                    gemm("gu", i)
                    m._ws_release("gu", i, "dn")
                    MAIN.add(("silu",))
                    m._ws_use("dn", i)
                    gemm("dn", i)
                    m._ws_release("dn", nxt, "gu")
                else:
                    m._ws_use("mlp", i)
                    gemm("mlp", i)
                    gemm("mlp", i)
                    m._ws_release("mlp", nxt)
            MAIN.add(("head",))
            for i in reversed(range(L)):                        # backward
                prv = i - 1 if i > 0 else None
                m._ws_use("attn", i)
                gemm("attn", i)                                 # recompute qkv, o
                if shared:
                    m._ws_use("dn", i)
                    gemm("dn", i)                               # d^T
                    m._ws_release("dn", i, "gu")
                    MAIN.add(("silu_bwd",))
                    m._ws_use("gu", i)
                    gemm("gu", i)                               # gu^T
                    m._ws_release("gu", prv, "dn")
                else:
                    m._ws_use("mlp", i)
                    gemm("mlp", i)                              # recompute gu, d^T, gu^T
                    m._ws_release("mlp", prv)
                gemm("attn", i)                                 # o^T, qkv^T
                m._ws_release("attn", prv)
                MAIN.add(("lora",))
            MAIN.add(("optimizer",))
            per_step.append(m.n_dequant - d0)
        # hazards: for each read, the latest write of that part ordered before it must be the
        # expected layer, and no write of that part may be unordered with the read
        bad = 0
        for rop, reg, want in reads:
            prior = [w for w in writes if w[1] == reg and before(w[0], rop)]
            racing = [w for w in writes if w[1] == reg and not before(w[0], rop) and not before(rop, w[0])]
            last = prior[-1] if prior else None
            if last is None or last[2] != want or racing:
                bad += 1
        print(f"overlap={overlap} shared_mlp={shared}: dequants per step {per_step} (layers {L}), "
              f"{len(reads)} weight reads, hazards: {bad}")
        assert bad == 0
        if shared:   # forward: attn, gu, dn per layer; backward: dn, gu, attn; two reuses per turn
            assert per_step[0] == 6 * L - 2 and all(x == 6 * L - 4 for x in per_step[1:]), per_step
        else:
            assert per_step[0] == 4 * L - 2 and all(x == 4 * L - 4 for x in per_step[1:]), per_step
        OPS.clear()
        reads.clear()
        writes.clear()
        m.ws_have = {"attn": None, "mlp": None}
        m.ev_free = {"attn": Event(), "mlp": Event()}
        m.ev_ready = {"attn": Event(), "mlp": Event()}
    print("WORKSPACE SCHEDULE OK")
    pipe_main(fast_qwen)


class _FakeSlot:
    def __init__(self, k):
        self.k = k

    def copy_(self, src, non_blocking=False):
        COPIES.append((CUR[0].add(("h2d", self.k, src.layer)), self.k, src.layer))


COPIES = []


def pipe_main(fast_qwen):
    """The v6 weight pipe: 4-bit layers in host memory, layer j streamed into GPU slot j % 2 two
    layers ahead, dequantised from the slot. Drives the real _ws_use / _ws_release /
    _weights_ahead / WeightPipe.prefetch / acquire in the order of FastQwen2.forward/backward and
    checks, from the stream/event graph:
      * every dequant reads its slot after the copy of the expected layer, and no other copy into
        that slot is unordered with the read (no overwrite while a dequant may still read it),
      * every GEMM reads the workspace part of its layer (as in main()),
      * copies per steady-state step: 2 * (L - 2) - (layers kept resident) * 2."""
    from solution import k_nf4

    L, SPLIT, SPLIT_D, NQB = 8, 1024, 3072, 4096
    parts = {(0, SPLIT): "attn", (SPLIT, NQB): "mlp", (SPLIT, SPLIT_D): "gu", (SPLIT_D, NQB): "dn"}

    def region(part):
        return "attn" if part == "attn" else "mlp"

    for resident in ((), (0, 1, 2)):
      for shared in (False, True):  # noqa: E111
        for overlap in (True, False):
            for attn_saved in (True, False):
                OPS.clear()
                COPIES.clear()
                m = object.__new__(fast_qwen.FastQwen2)
                m.overlap, m.split_q, m.n_dequant = overlap, SPLIT, 0
                m.split_d, m.share_mlp = SPLIT_D, shared
                m.ws_have = {"attn": None, "mlp": None}
                m.ev_free = {"attn": Event(), "mlp": Event()}
                m.ev_ready = {"attn": Event(), "mlp": Event()}
                m.dq_stream = Stream("side") if overlap else None
                m.device = None
                host = [types.SimpleNamespace(n_qblocks=NQB, layer=i, flat=types.SimpleNamespace(layer=i),
                                              where="gpu" if i in resident else "host")
                        for i in range(L)]
                m.packed = host
                m.code2 = m.nf4 = m.ws = None
                pipe = object.__new__(k_nf4.WeightPipe)
                pipe.layers, pipe.device, pipe.side = host, None, m.dq_stream
                pipe.resident = set(resident)
                pipe.slots = [_FakeSlot(0), _FakeSlot(1)]
                pipe.views = [types.SimpleNamespace(n_qblocks=NQB, slot=0), types.SimpleNamespace(n_qblocks=NQB, slot=1)]
                pipe.owner, pipe.loaded = [None, None], [None, None]
                pipe.stream = Stream("wcopy")
                pipe.copies = 0
                m.pipe = pipe
                asked = []
                orig_acquire = k_nf4.WeightPipe.acquire

                def acquire(self, j, stream, _orig=orig_acquire):
                    asked.append(j)
                    return _orig(self, j, stream)

                pipe.acquire = types.MethodType(acquire, pipe)
                dq_reads = []   # (op, slot or None, layer)
                writes, reads = [], []

                def fake_dequant_range(src, q0, q1, code2, nf4, out, dst=None):
                    part = parts[(q0, q1)]
                    j = asked[-1]
                    op = CUR[0].add(("dequant", part, j))
                    writes.append((op, region(part), (part, j)))
                    dq_reads.append((op, getattr(src, "slot", None), j, getattr(src, "where", "slot")))

                fast_qwen.dequant_range = fake_dequant_range

                def gemm(part, i):
                    reads.append((CUR[0].add(("gemm", part, i)), region(part), (part, i)))

                pipe.prefetch(0)
                pipe.prefetch(1)
                per_step = []
                for step in range(3):
                    c0 = pipe.copies
                    if overlap:
                        m.dq_stream.wait_stream(MAIN)
                    for i in range(L):                                  # forward
                        nxt = i + 1 if i + 1 < L else None
                        m._ws_use("attn", i)
                        gemm("attn", i)
                        m._ws_release("attn", nxt)
                        MAIN.add(("norm",))
                        if shared:
                            m._ws_use("gu", i)
                            gemm("gu", i)
                            m._ws_release("gu", i, "dn")
                            m._weights_ahead(i + 2)
                            MAIN.add(("silu",))
                            m._ws_use("dn", i)
                            gemm("dn", i)
                            m._ws_release("dn", nxt, "gu")
                        else:
                            m._ws_use("mlp", i)
                            m._weights_ahead(i + 2)
                            gemm("mlp", i)
                            m._ws_release("mlp", nxt)
                    MAIN.add(("head",))
                    for i in reversed(range(L)):                        # backward
                        prv = i - 1 if i > 0 else None
                        if not attn_saved:
                            m._ws_use("attn", i)
                            gemm("attn", i)                             # recompute qkv, o
                        if shared:
                            m._ws_use("dn", i)
                            gemm("dn", i)
                            m._ws_release("dn", i, "gu")
                            MAIN.add(("silu_bwd",))
                            m._ws_use("gu", i)
                            gemm("gu", i)
                            m._ws_release("gu", prv, "dn")
                        else:
                            m._ws_use("mlp", i)
                            gemm("mlp", i)
                            m._ws_release("mlp", prv)
                        if attn_saved:
                            m._ws_use("attn", i)
                        m._weights_ahead(i - 2)
                        gemm("attn", i)                                 # o^T, qkv^T
                        m._ws_release("attn", prv)
                        MAIN.add(("lora",))
                    MAIN.add(("optimizer",))
                    per_step.append(pipe.copies - c0)
                bad_ws = 0
                for rop, reg, want in reads:
                    prior = [w for w in writes if w[1] == reg and before(w[0], rop)]
                    racing = [w for w in writes if w[1] == reg and not before(w[0], rop) and not before(rop, w[0])]
                    if not prior or prior[-1][2] != want or racing:
                        bad_ws += 1
                bad_slot = 0
                for rop, slot, layer, where in dq_reads:
                    if slot is None:
                        if layer not in resident or where != "gpu":
                            bad_slot += 1
                        continue
                    prior = [c for c in COPIES if c[1] == slot and before(c[0], rop)]
                    racing = [c for c in COPIES if c[1] == slot and not before(c[0], rop) and not before(rop, c[0])]
                    if not prior or prior[-1][2] != layer or racing or layer % 2 != slot:
                        bad_slot += 1
                streamed = L - len(resident)
                expect = 2 * (streamed - 2) if not resident else 2 * (streamed - 2) + 0
                print(f"pipe resident={list(resident)} shared_mlp={shared} overlap={overlap} attn_saved={attn_saved}: "
                      f"H2D copies per step {per_step}, workspace hazards {bad_ws}, slot hazards {bad_slot}")
                assert bad_ws == 0 and bad_slot == 0
                assert all(x == expect for x in per_step[1:]), (per_step, expect)
    print("WEIGHT PIPE SCHEDULE OK")


if __name__ == "__main__":
    main()
