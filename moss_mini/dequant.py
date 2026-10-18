"""Optional CUDA unpack+scale kernel. GEMM and quantization remain unchanged."""
import torch


def dequantize_triton(qweight, scale, bits, group_size, in_features):
    """Write BF16 directly, without full-size unpacked uint8/FP32 intermediates."""
    if not qweight.is_cuda or not scale.is_cuda:
        raise ValueError("Triton dequantization requires CUDA tensors")
    if not qweight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("Triton dequantization requires contiguous weight and scale")
    # Lazy import: the reference/CPU path does not depend on Triton.
    from .dequant_kernel import unpack_scale
    output = torch.empty((qweight.shape[0], in_features), device=qweight.device, dtype=torch.bfloat16)
    count = output.numel()
    with torch.cuda.device(qweight.device):
        unpack_scale[((count + 1023) // 1024,)](
            qweight, scale, output, count, in_features, group_size, bits, BLOCK=1024)
    return output
