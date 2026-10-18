"""CUDA-only paired microbenchmark; reports allocator peaks, not NVML usage."""
import argparse
import json
from pathlib import Path
import torch
from moss_mini.encoder import pack_codes, unpack_codes
from moss_mini.dequant import dequantize_triton


def measure(fn, repeats=30):
    for _ in range(5):
        out = fn()
        del out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = fn()
        del out
    end.record()
    end.synchronize()
    return dict(milliseconds=start.elapsed_time(end)/repeats,
                incremental_allocated_peak_bytes=torch.cuda.max_memory_allocated()-baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(518)
    results = []
    for bits in (4, 6, 8):
        for rows, cols in ((3, 256), (512, 4096), (12288, 4096)):
            group = cols if bits == 8 else 64
            codes = torch.randint(0, 2**bits, (rows, cols), dtype=torch.uint8, device='cuda')
            packed = pack_codes(codes, bits)
            del codes
            scale = torch.rand(rows, cols//group, device='cuda', dtype=torch.float32) + 1e-5
            def reference():
                w = unpack_codes(packed, bits).float().sub_(2**(bits-1)-1)
                w = w.reshape(rows, -1, group)
                w.mul_(scale[..., None])
                return w.reshape(rows, cols).to(torch.bfloat16)
            def fused():
                return dequantize_triton(packed, scale, bits, group, cols)
            expected, actual = reference(), fused()
            assert torch.equal(actual, expected), (bits, rows, cols)
            x = torch.randn(1, 17, cols, device='cuda', dtype=torch.bfloat16)
            assert torch.equal(torch.nn.functional.linear(x, expected), torch.nn.functional.linear(x, actual))
            del expected, actual, x
            row = dict(bits=bits, shape=[rows, cols], weight_exact=True, gemm_exact=True,
                       torch=measure(reference), triton=measure(fused))
            results.append(row)
            print(json.dumps(row), flush=True)
            del packed, scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'device': torch.cuda.get_device_name(), 'rows': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
