# solution/k_nf4.py (core LUT helper + kernel structure)
import triton
import triton.language as tl


@triton.jit
def _nf4_lut_tree(nib):
    # Balanced 4-level binary tree (depth 4 instead of depth 15 sequential chain)
    # Works identically on both T4 GPU and dev/tl_emu.py CPU emulator
    b0 = (nib & 1) != 0
    b1 = (nib & 2) != 0
    b2 = (nib & 4) != 0
    b3 = (nib & 8) != 0

    # Level 1 (pairs)
    p0 = tl.where(b0, -0.6961928009986877, -1.0)
    p1 = tl.where(b0, -0.39491748809814453, -0.5250730514526367)
    p2 = tl.where(b0, -0.18477343022823334, -0.28444138169288635)
    p3 = tl.where(b0, 0.0, -0.09105003625154495)
    p4 = tl.where(b0, 0.16093020141124725, 0.07958029955625534)
    p5 = tl.where(b0, 0.33791524171829224, 0.24611230194568634)
    p6 = tl.where(b0, 0.5626170039176941, 0.44070982933044434)
    p7 = tl.where(b0, 1.0, 0.7229568362236023)

    # Level 2 (quads)
    q0 = tl.where(b1, p1, p0)
    q1 = tl.where(b1, p3, p2)
    q2 = tl.where(b1, p5, p4)
    q3 = tl.where(b1, p7, p6)

    # Level 3 (octets)
    o0 = tl.where(b2, q1, q0)
    o1 = tl.where(b2, q3, q2)

    # Level 4 (final 16-entry value)
    return tl.where(b3, o1, o0)
