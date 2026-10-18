# Second optimization pass: measured candidates without a demonstrated speedup

This pass tested the remaining small optimizations identified by the source
scan. None demonstrated a useful latency improvement in the measured workload,
so **the published runtime and defaults remain unchanged**. Candidate code and
execution artifacts were retained in the research workspace; these prototypes
are not exposed as supported CLI options.

The reference is commit `811313a`: original 9B INT6 encoder, prepacked fused
INT8 DiT, CPU vocabulary lookup with zero offloaded encoder layers, and exact
request caching. Tests used RTX 4090D 24GiB, four CPU threads, batch one,
1024×1024, 50 steps, seed 42 and the same skateboard/mouse development prompt.
Measurements are individual warmed runs on a shared host, not a held-out quality
benchmark or a statistically powered speed comparison. GB is decimal.

[Raw selected receipts and equality checks](optimization-round2-metrics.json)
retain the positive and negative results.

## Encoder loading

The candidate grouped packed and retained tensor reads by shard, kept at most
one safetensors handle open, and validated scale/bias values on CPU before GPU
placement. CPU-retained tensors were detached from mapped storage. The original
source identity, inventory, shape, dtype and finite-value checks were preserved.

| Complete encode-and-generate command | Seconds | Task GPU GB | Tree RSS GB |
|---|---:|---:|---:|
| Published reference | 48.200 | 6.598 | 7.360 |
| Loader candidate | 48.330 | 6.581 | 7.368 |

Encoder loading itself took 8.386 seconds in the reference and 8.453 seconds in
the candidate. Conditions, initial noise and decoded PNG pixels matched exactly.
The small sampled-memory difference is not evidence of a reliable memory saving.
This experiment does not establish an improvement for cold disk reads.

## Generator preprocessing and buffers

These runs loaded the same packed DiT and saved conditions without loading the
encoder. Their complete-process times must not be compared directly with the
encode-and-generate commands above. Both reference runs used exact request
caching. Execution order was reference A, unused-allocation removal, dense
prologue, CFG step cache, then reference B.

| Generator case | Denoise + decode seconds | Complete process seconds |
|---|---:|---:|
| Reference A | 24.802 | 35.623 |
| Remove unused INT8 allocation | 24.878 | 35.649 |
| Also omit discarded norm computation | 24.950 | 35.830 |
| Reuse video/time/modulation within a CFG step | 24.831 | 35.766 |
| Reference B | 24.922 | 35.946 |

All five final latent tensors and decoded PNGs matched reference A exactly.
Sampled task GPU peak was 6.407GB in each case, and tree RSS was approximately
7.35GB. These observations do not support a speedup for any candidate.

The CFG candidate achieved **50 hits and 50 misses in each of four stages**:
video projection, time MLP, shared modulation and head modulation. It retained
at most 25,221,120 counted output bytes, with zero at context exit. Replacing half
of these computations did not materially reduce whole-image time. The original
sinusoid still ran at each transformer call.

The dense-prologue candidate avoided a logically accumulated 104.84GB of unused
INT8 buffer allocations over 8,300 calls, plus discarded norm allocations and
reductions. This is allocator-request accounting, **not memory traffic or peak
memory saved**. Original residual clones were retained, preserving Cache-DiT's
input ownership. No fused norm-to-INT8 quantization kernel or clone-free block
implementation was tested in this pass.

## Correctness evidence and scope

The [prologue GPU witness](optimization-round2-prologue-witness.json) compared
BF16 outputs, subsequent INT8 codes/scales and input immutability at width 3072,
including 4117/4352-token inputs, zero modulation and outliers; all compared
outputs were exact. Microbenchmark improvements were inconsistent across shapes
and did not translate to complete-generation speedups.

The candidate CPU suites exercised malformed packed artifacts, mmap lifetime,
cache invalidation, parameter references, capacity, hooks and cleanup. A separate
tiny instance of the actual pinned pipeline exercised CFG/SLG and composition
with Cache-DiT and Accelerate on CPU. The step-cache prototype is not a general
API guarantee: review found that changing the FP16 accumulation setting between
calls was not included in its cache key. Production BF16 measurements here were
unaffected; that issue would need correction before a wider release.

All seven measured command/process runs completed successfully and stayed below
8GB sampled task GPU memory and 16GB sampled process-tree RSS. Sampling can miss
short peaks, and RSS sums include shared pages. **A physical 8GB GPU / 16GB host
has still not been tested.**

## Where the original block spends time

A [targeted eager profile](optimization-round2-block-profile.json) captured the
actual first joint block at the first positive-conditioning step, with 4117
tokens and width 3072. After eight warmups, ten uninstrumented repeats had a
median of **4.348ms**. Ten separate profiled repeats gave this breakdown of
observed device kernel time:

| Observed work | Mean device time per block | Share of device kernel time |
|---|---:|---:|
| INT8 GEMMs, four launches | 2.270ms | 55.20% |
| PyTorch FlashAttention | 1.490ms | 36.24% |
| Norm, RoPE and activation quantization | 0.309ms | 7.51% |
| Device-to-device copy | 0.043ms | 1.05% |

The INT8 kernel observed was OrbitQuant's Triton `_int8_scaled_gemm_body`, not a
cuBLASLt GEMM. The dependency on CUDA libraries elsewhere does not establish
which implementation executes this block. The original output and inputs stayed
unchanged across repeated eager calls.

The denominator excludes the enclosing profiler user annotation, which overlaps
the kernels it contains. Profiling introduces overhead; these shares describe
this captured block, **not the complete image-generation pipeline**, other
blocks, negative CFG or later steps. Model loading, offload and VAE decoding
were outside the profiled interval.

This evidence prioritizes the existing INT8 GEMM dispatch/tiling and attention
implementation for a later pass. Comparing another GEMM engine would also need
to preserve or explicitly measure changes in Orbit's fused scale/activation/
residual arithmetic. Residual-copy removal has a much smaller observed budget
here, and the unused prologue work already failed to improve full-image latency.
