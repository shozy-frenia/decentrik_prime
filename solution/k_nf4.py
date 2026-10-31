"""NF4 (bitsandbytes, double-quantised) -> fp16 dequantisation of a whole decoder layer per launch.

At setup the seven 4-bit weights of a layer (q, k, v, o, gate, up, down) are repacked into one
contiguous byte buffer, one uint8 absmax buffer and one per-256-block (scale, offset) table, in
exactly the order of the fp16 workspace they dequantise into:

    [ Wq ; Wk ; Wv ]  -> one [2048, 1536] matrix (fused QKV GEMM)
    [ Wo ]            -> [1536, 1536]
    [ Wgate ; Wup ]   -> one [17920, 1536] matrix (fused gate|up GEMM)
    [ Wdown ]         -> [1536, 8960]

so a single launch writes every fp16 weight of the layer, already concatenated for the fused
GEMMs. bitsandbytes' own Params4bit / QuantState tensors are re-pointed at views of the repacked
buffers, so no byte is stored twice.

Format (bitsandbytes 0.50.2, csrc/kernels.cu kDequantizeBlockwise<NF4> and
functional.dequantize_4bit with a nested state): element e of a weight lives in byte e // 2, the
even element in the high nibble; it dequantises to
    fp16( NF4[nibble] * absmax[e // 64] ),  absmax[b] = code2[absmax_u8[b]] * absmax2[b // 256] + offset
with the product and the sum rounded separately (we launch with enable_fp_fusion=False), so the
fp16 weights are bit-identical to bitsandbytes' dequantize_4bit.
"""

import torch
import triton
import triton.language as tl

# bitsandbytes csrc/kernels.cu: nf4_dequantization_lut
NF4_CODE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)

QBLOCKS_PER_PROGRAM = 128   # 128 quantisation blocks = 4096 packed bytes = 8192 weights
DEQUANT_WARPS = 4


@triton.jit(do_not_specialize=["n_qblocks"])
def nf4_layer_dequant_kernel(packed_ptr, qabs_ptr, scale2_ptr, offset2_ptr, code2_ptr, nf4_ptr,
                             out_ptr, n_qblocks, QB: tl.constexpr):
    pid = tl.program_id(0)
    qb = pid * QB + tl.arange(0, QB)                      # quantisation-block ids
    live = qb < n_qblocks
    sb = qb // 256                                        # second-level (absmax) block ids
    code = tl.load(qabs_ptr + qb, mask=live, other=0).to(tl.int32)
    first = tl.load(code2_ptr + code) * tl.load(scale2_ptr + sb, mask=live, other=0.0)
    absmax = first + tl.load(offset2_ptr + sb, mask=live, other=0.0)
    nib = tl.arange(0, 32)
    src = qb[:, None] * 32 + nib[None, :]                 # [QB, 32] packed bytes
    byte = tl.load(packed_ptr + src, mask=live[:, None], other=0).to(tl.int32)
    w_even = (tl.load(nf4_ptr + (byte >> 4)) * absmax[:, None]).to(tl.float16)
    w_odd = (tl.load(nf4_ptr + (byte & 15)) * absmax[:, None]).to(tl.float16)
    w = tl.reshape(tl.join(w_even, w_odd), (QB, 64))     # interleave: even, odd, even, ...
    dst = qb[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out_ptr + dst, w, mask=live[:, None])


# Order and roles of the weights inside a repacked layer.
LAYER_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class PackedLayer:
    """The repacked 4-bit weights of one decoder layer and their fp16 workspace layout.

    All four parts live in one flat uint8 buffer (`flat`), on the GPU or in pinned host memory:
        [ packed nibbles | absmax codes | second-level scales (fp32) | offsets (fp32) ]
    each section starting on a 64-byte boundary (`offsets`), so a host copy streams to the GPU
    with a single H2D transfer.
    """

    def __init__(self, flat, offsets, n_qblocks, segments):
        self.flat, self.offsets = flat, offsets
        self.n_qblocks = n_qblocks
        self.segments = segments          # name -> (start element, rows, cols)
        self.numel = n_qblocks * 64
        self.packed, self.qabs, self.scale2, self.offset2 = section_views(flat, offsets, n_qblocks)


def section_layout(n_qblocks):
    """Byte offsets of the four sections and the total size of a flat layer buffer."""
    sizes = (n_qblocks * 32, n_qblocks, (n_qblocks // 256) * 4, (n_qblocks // 256) * 4)
    offs, at = [], 0
    for size in sizes:
        offs.append(at)
        at += -(-size // 64) * 64
    return tuple(offs), at


def section_views(flat, offsets, n_qblocks):
    o_pk, o_qa, o_s2, o_o2 = offsets
    n2 = n_qblocks // 256
    return (flat[o_pk:o_pk + n_qblocks * 32], flat[o_qa:o_qa + n_qblocks],
            flat[o_s2:o_s2 + 4 * n2].view(torch.float32), flat[o_o2:o_o2 + 4 * n2].view(torch.float32))


def _quant_parts(linear):
    """(packed uint8 data, QuantState) of a bitsandbytes Linear4bit, checked for our format."""
    weight = linear.weight
    state = getattr(weight, "quant_state", None)
    if state is None:
        state = getattr(linear, "quant_state", None)
    if state is None:
        raise ValueError("4-bit weight without quant_state")
    if getattr(state, "quant_type", None) != "nf4" or int(state.blocksize) != 64:
        raise ValueError(f"unsupported 4-bit format {state.quant_type}/{state.blocksize}")
    if not getattr(state, "nested", False) or state.state2 is None or int(state.state2.blocksize) != 256:
        raise ValueError("expected double quantisation (nested absmax, blocksize 256)")
    if weight.data.dtype != torch.uint8:
        raise ValueError(f"packed storage {weight.data.dtype}, expected uint8")
    if state.dtype != torch.float16:
        raise ValueError(f"quant_state dtype {state.dtype}, expected float16")
    return weight, state


def layer_numel(linears):
    """Number of 4-bit weights of a layer (sum over its seven matrices)."""
    total = 0
    for name in LAYER_WEIGHTS:
        _, state = _quant_parts(linears[name])
        total += int(state.shape[0]) * int(state.shape[1])
    return total


def repack_layer(linears, device, host=False, out=None):
    """Repack the seven Linear4bit modules of a layer (dict name -> module) into one PackedLayer and
    point bitsandbytes' tensors at views of it (nothing is stored twice). With host=True the
    buffer is pinned host memory and the GPU copies are released. Returns (PackedLayer, code2)."""
    parts = [(name, *_quant_parts(linears[name])) for name in LAYER_WEIGHTS]
    total = 0
    segments = {}
    code2 = None
    for name, weight, state in parts:
        rows, cols = (int(state.shape[0]), int(state.shape[1]))
        n = rows * cols
        if n % (64 * 256) != 0:
            raise ValueError(f"{name}: {n} weights do not fill whole absmax blocks")
        if weight.data.numel() * 2 != n or state.absmax.numel() * 64 != n:
            raise ValueError(f"{name}: unexpected packed sizes")
        if state.state2.absmax.numel() * 256 * 64 != n:
            raise ValueError(f"{name}: unexpected second-level absmax size")
        if code2 is None:
            code2 = state.state2.code.to(device)
        elif not torch.equal(code2, state.state2.code.to(device)):
            raise ValueError("second-level code tables differ between weights")
        segments[name] = (total, rows, cols)
        total += n
    n_qblocks = total // 64
    offsets, nbytes = section_layout(n_qblocks)
    if out is not None:
        if out.numel() != nbytes or out.dtype != torch.uint8:
            raise ValueError("preallocated layer buffer has the wrong size")
        flat = out
    elif host:
        flat = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    else:
        flat = torch.empty(nbytes, dtype=torch.uint8, device=device)
    packed, qabs, scale2, offset2 = section_views(flat, offsets, n_qblocks)
    for name, weight, state in parts:
        start, rows, cols = segments[name]
        n = rows * cols
        b0, q0, s0 = start // 2, start // 64, start // (64 * 256)
        b1, q1, s1 = b0 + n // 2, q0 + n // 64, s0 + n // (64 * 256)
        packed[b0:b1].copy_(weight.data.reshape(-1))
        qabs[q0:q1].copy_(state.absmax.reshape(-1))
        scale2[s0:s1].copy_(state.state2.absmax.reshape(-1).float())
        offset2[s0:s1].copy_(state.offset.float().reshape(1).expand(s1 - s0))
        weight.data = packed[b0:b1].view(weight.data.shape)
        state.absmax = qabs[q0:q1].view(state.absmax.shape)
        state.state2.absmax = scale2[s0:s1].view(state.state2.absmax.shape)
        state.offset = state.offset.to("cpu" if host else device)
    return PackedLayer(flat, offsets, n_qblocks, segments), code2.float().contiguous()


def dequant_tensors(packed, qabs, scale2, offset2, n_qblocks, code2, nf4, out):
    grid = (triton.cdiv(n_qblocks, QBLOCKS_PER_PROGRAM),)
    nf4_layer_dequant_kernel[grid](packed, qabs, scale2, offset2, code2, nf4, out, n_qblocks,
                                   QB=QBLOCKS_PER_PROGRAM, num_warps=DEQUANT_WARPS,
                                   enable_fp_fusion=False)
    return out


def dequant_layer(layer, code2, nf4, out):
    """Dequantise every weight of `layer` (GPU-resident) into the flat fp16 buffer `out`."""
    return dequant_tensors(layer.packed, layer.qabs, layer.scale2, layer.offset2, layer.n_qblocks,
                           code2, nf4, out)


RANGE_ALIGN = 1024   # quantisation blocks: keeps every sliced section pointer 16-byte aligned


def dequant_range(layer, q0, q1, code2, nf4, out, dst=None):
    """Dequantise quantisation blocks [q0, q1) of `layer` into the same positions of `out` (the
    flat workspace). q0 and q1 must be multiples of RANGE_ALIGN (or q1 the layer end), so that the
    sliced packed / absmax / scale pointers stay 16-byte aligned like the ones the kernel was
    compiled for, and second-level blocks (256 quantisation blocks) are never split."""
    if q0 % RANGE_ALIGN or (q1 % RANGE_ALIGN and q1 != layer.n_qblocks):
        raise ValueError("dequant range not aligned")
    s0, s1 = q0 // 256, -(-q1 // 256)
    start = q0 * 64 if dst is None else int(dst)   # dst: element offset in `out` (shared region)
    if start % (RANGE_ALIGN * 64):
        raise ValueError("dequant destination not aligned")
    return dequant_tensors(layer.packed[q0 * 32:q1 * 32], layer.qabs[q0:q1], layer.scale2[s0:s1],
                           layer.offset2[s0:s1], q1 - q0, code2, nf4, out[start:start + (q1 - q0) * 64])


class SlotView:
    """The four sections of a packed layer held in a GPU slot (same attributes as PackedLayer that
    the dequantisation reads)."""

    def __init__(self, flat, offsets, n_qblocks):
        self.flat, self.n_qblocks = flat, n_qblocks
        self.packed, self.qabs, self.scale2, self.offset2 = section_views(flat, offsets, n_qblocks)


class WeightPipe:
    """Repacked 4-bit layers kept in pinned host memory, streamed into two GPU slots.

    Layer j always goes to slot j % 2: consecutive layers of a walk (forward 0..L-1, backward
    L-1..0, and the reuse of the same layer at each turn) never share a slot. The dequantisation of
    layer j's two halves reads the slot, on the dequantisation stream `reader`.
      prefetch(j)  queue the H2D copy of layer j into slot j % 2 on the pipe's own copy stream,
                   after everything already queued on the current stream and on the dequantisation
                   stream (in particular the last dequants that read the slot's previous layer,
                   whichever of the two streams runs them). The caller issues it only once both
                   halves of that previous layer have been queued, so nothing can still need the
                   old bytes.
      acquire(j, s) the GPU views of layer j for stream s, which waits for the copy (starting it
                   now if it was never prefetched, e.g. on the very first step).
    Layers in `resident` stay on the GPU (their PackedLayer views are returned directly).
    """

    def __init__(self, layers, device, side=None, resident=()):
        self.layers = layers
        self.device = device
        self.side = side
        self.resident = set(resident)
        n_qblocks = layers[0].n_qblocks
        self.offsets, nbytes = section_layout(n_qblocks)
        self.slots = [torch.empty(nbytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.views = [SlotView(s, self.offsets, n_qblocks) for s in self.slots]
        self.owner = [None, None]
        self.loaded = [None, None]       # event: the copy into the slot is done (copy stream)
        self.stream = torch.cuda.Stream(device=device)
        self.copies = 0

    def streamed(self, j):
        return j is not None and 0 <= j < len(self.layers) and j not in self.resident

    def prefetch(self, j):
        if not self.streamed(j):
            return
        k = j % 2
        if self.owner[k] == j:
            return
        guards = []
        for s in (torch.cuda.current_stream(self.device), self.side):
            if s is not None:
                ev = torch.cuda.Event()
                ev.record(s)
                guards.append(ev)
        with torch.cuda.stream(self.stream):
            for ev in guards:
                self.stream.wait_event(ev)
            self.slots[k].copy_(self.layers[j].flat, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.owner[k], self.loaded[k] = j, done
        self.copies += 1

    def acquire(self, j, stream):
        if not self.streamed(j):
            return self.layers[j]
        self.prefetch(j)
        k = j % 2
        stream.wait_event(self.loaded[k])
        return self.views[k]


def warmup_specs():
    """(kernel, args, kwargs) for compiling ahead of time with kernel.warmup()."""
    u8, f32, f16 = torch.uint8, torch.float32, torch.float16
    return [(nf4_layer_dequant_kernel, (u8, u8, f32, f32, f32, f32, f16, 1),
             dict(QB=QBLOCKS_PER_PROGRAM, num_warps=DEQUANT_WARPS, enable_fp_fusion=False))]
