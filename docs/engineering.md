# Engineering profile — 2026-10-18

There is useful headroom, but model loading and reuse currently offer more
end-to-end opportunity than optimizing one unpack operator. Measurements below
use the same RTX 4090D 24GiB environment as the release, INT6/g64/offload20,
and unchanged BF16 GEMMs. [Machine-readable evidence](engineering-metrics.json).

## What was implemented

1. An opt-in Triton kernel combines packed integer extraction, FP32 scaling and
   BF16 output conversion. It avoids whole-matrix unpacked UINT8 and FP32
   intermediates. This is fused **dequantization**, not fused INT6 GEMM.
2. Stage timing separates encoder load/forward, DiT load/conversion and
   denoising/decoding. These appear in each run's `encode.json` and `generate.json`.
3. With `--packed`, Hub resolution requests only encoder configuration and
   tokenizer files instead of also fetching original BF16 text-encoder shards.
   Generation still needs the original DiT, scheduler, VAE and pipeline code.

The default remains `--encoder-dequant torch`. The optional kernel is explicit:

```bash
moss-mini generate --packed packed-int6/encoder --encoder-dequant triton \
  --prompt 'a photo of a yellow skateboard and an orange computer mouse' \
  --output outputs/skateboard-triton
```

Triton 3.5.1 was tested (provided by the tested Linux PyTorch installation).
There is no silent fallback if this explicitly selected backend fails.

## Measurements

| Measurement | PyTorch reference | Triton |
|---|---:|---:|
| INT6 unpack, 12288×4096 matrix | 2.087 ms | 0.225 ms |
| Incremental allocator peak for that unpack | 301,989,888 B | 100,663,296 B |
| 256-token encoder forward, warm median | 1.090 s | 0.860 s |
| Whole single-image command | 57.140 s | 57.054 s |
| Whole-command task GPU peak | 6.637 GB | 6.637 GB |
| Whole-command tree RSS peak | 8.438 GB | 8.473 GB |

The warm 256-token forward took about **21% less time**, but this did **not**
establish a whole-command latency or peak-memory improvement. Weight storage is
unchanged. Single-operator allocator savings do not imply a lower application
peak: loading, other operations and the CUDA allocator can dominate that peak.
The new path is retained as an experimental option, not promoted to default.

Protocol: nine bit-width/shape combinations (INT4/6/8 × 3×256, 512×4096,
12288×4096), five warmup calls and 30 measured calls per operator. CUDA events
measure the repeated operator loop, including possible host launch starvation;
they are not isolated kernel instruction latency. Memory is incremental PyTorch
allocated peak, not NVML process usage. Compilation is outside those micro timings.
Large-matrix results show about 9.3× faster INT6 unpack only.

The context test uses a synthetic prompt truncated to exactly 256 tokens,
one first forward and five subsequent forwards; the table takes the median of
the five. First forwards were 1.675 s and 1.882 s respectively. These are process
first calls, not controlled empty compiler-cache timings. Full commands were
run once per backend (torch then Triton); context tests ran in reverse order.
Disk cache, JIT cache, allocator history and shared-host activity can affect
results. This is a bounded engineering witness, not a broad benchmark.

Every tested reconstructed weight and GEMM output matched exactly. The short
prompt's conditions and PNG pixels matched exactly across backends. Conditions
for the synthetic 256-token prompt also matched exactly; comparison was CPU-only.
Both full commands and both context tests exited 0 with no sampling errors.
No physical 8GB GPU or 16GB host was tested.

Validation also includes three CUDA tests for INT4/6/8, a CPU-only run with
10 passed and 3 CUDA tests skipped, and a wheel build containing both new kernel
modules. An independent read-only check matched the aggregate to original
receipts and repeated the CPU conditions/image comparisons successfully.

## Where to optimize next

The reference full command breaks down as follows:

| Stage | Seconds |
|---|---:|
| Load prepacked encoder | 10.72 |
| Encode positive/negative prompts and save | 2.95 |
| Load DiT nonblock state, scheduler and VAE | 3.46 |
| Read and convert DiT blocks to fused INT8 | 7.47 |
| Prepare generator | 0.05 |
| Denoise, decode and save image | 25.21 |
| Outside these recorded stages, including startup/imports | 7.27 |

Recommended priorities:

1. **Save and cold-load the fused INT8 DiT.** The 7.47 s conversion is repeated
   on every command. A versioned packed generator could avoid the conversion,
   although replacement loading time means this is not 7.47 s of guaranteed
   savings. Bind backend layout/version/source identity and verify packed state
   and PNG equivalence before using it as a default.
2. **Reuse conditions and generate multiple seeds per model load.** An explicit
   condition cache bound to prompt, negative prompt, tokenizer/source identity,
   precision and max length can avoid the encoder stage for repeated prompts.
   A sequential batch worker can amortize DiT loading and Python startup without
   keeping encoder and generator weights resident together. Avoid an unbounded
   cache or simultaneous model residency under the memory target.
3. **Profile the encoder's linear-attention kernels.** Current runtime warnings
   show PyTorch fallbacks for `causal_conv1d` and `chunk_gated_delta_rule` because
   `causal_conv1d` / `flash-linear-attention` are absent. Isolated optional installs
   and controlled equivalence checks are warranted; their speed and numerical
   behavior have not been measured here.
4. **Investigate actual peak allocation and placement.** Fused unpack alone did
   not lower the full process peak. Use a short allocator/NVML trace before
   moving more layers to GPU or keeping BF16 reconstructions resident. Those
   changes can exchange speed for VRAM and should not be assumed to fit 8GB.

## Reproduce the operator benchmark

From an installed checkout, on the selected CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_dequant.py --output outputs/dequant.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

`benchmarks/benchmark_context.py` accepts `--model`, `--source-version`,
`--packed`, `--backend` and a fresh `--output` directory for the exact 256-token
stress. It saves conditions for an independent elementwise comparison. The
external NVML/RSS sampler used for the reported experiment is distinct from
these benchmark scripts; their own outputs do not certify process-memory limits.
