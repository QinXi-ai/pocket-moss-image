"""Bit-exact packed integer reconstruction into a BF16 output matrix."""
import triton
import triton.language as tl


@triton.jit
def unpack_scale(Q, S, OUT, count: tl.constexpr, cols: tl.constexpr,
                 group: tl.constexpr, bits: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    row, col = i // cols, i % cols
    bit = col * bits
    byte = row * (cols * bits // 8) + bit // 8
    shift = bit % 8
    low = tl.load(Q + byte, valid, other=0).to(tl.int32)
    high = tl.load(Q + byte + 1, valid & (shift + bits > 8), other=0).to(tl.int32)
    code = ((low | (high << 8)) >> shift) & ((1 << bits) - 1)
    value = (code - ((1 << (bits - 1)) - 1)).to(tl.float32)
    scale = tl.load(S + row * (cols // group) + col // group, valid, other=0)
    tl.store(OUT + i, (value * scale).to(tl.bfloat16), valid)
