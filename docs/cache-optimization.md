# Prepacked DiT and saved conditions

The optional `--packed-generator` loader removes repeated BF16-to-INT8 DiT
conversion. `--reuse-conditions` additionally skips the encoder for a previously
encoded prompt. They preserve the original 9B INT6/g64/offload20 recipe and fused
W8A8 DiT. [Commands](../README.md#pack-once-reuse), [raw numeric summary](cache-metrics.json).

## Whole-command measurements

All runs: RTX 4090D 24GiB, four CPU threads, 1024×1024, 50 steps, seed 42,
PyTorch encoder dequantization, the skateboard/mouse prompt shown in the README.
Reference runs already use the prepacked INT6 encoder. Seconds include startup,
imports, sequential subprocesses and output writing. GB below means decimal GB.

| Run, in execution order | Seconds | Task GPU peak GB | Tree RSS peak GB |
|---|---:|---:|---:|
| Reference 1, first-use cache effects | 81.425 | 6.637 | 8.511 |
| Packed DiT 1 | 53.062 | 6.637 | 8.476 |
| Packed DiT 2 | 54.049 | 6.637 | 8.389 |
| Reference 2 | 56.582 | 6.637 | 8.382 |
| Reference 3 | 55.719 | 6.637 | 8.364 |
| Packed DiT 3 | 52.426 | 6.637 | 8.499 |
| Packed DiT + reused conditions 1 | 38.467 | 6.407 | 7.360 |
| Packed DiT + reused conditions 2 | 38.231 | 6.407 | 7.362 |

Using the last two reference and packed runs, mean command time decreases
**56.150 → 53.237 seconds (5.2%)**. With saved conditions it is **38.349 seconds
(31.7% below reference)**. Reference 1 is retained but excluded from these means:
its load and first-use kernels were substantially slower. These are warm-cache
observations on a shared host, two runs per reported mean, one prompt and seed;
they are not a broad benchmark or a speed guarantee.

The last reference spends about 11.32 seconds loading/converting the DiT; packed
loading takes 6.51 seconds plus 0.81 seconds of preparation. Reuse additionally
avoids about 11.5 seconds of encoder load/forward and part of process startup.
There is still a small CPU subprocess to validate/copy saved conditions.

All eight conditions, initial noises and PNG pixel arrays matched exactly.
All commands exited successfully and the resource sampler reported no errors.
This establishes replay equivalence for this case, not improved image quality.
Cold restoration also matched every one of the **547 saved DiT tensors**, without
initializing CUDA. Packed tensor state is **5,413,609,536 bytes**. Packing took
about 25 seconds for the full command (16.78 seconds inside the worker).

Task-process GPU peaks and sampled tree RSS were below decimal 8GB/16GB here.
The NVML/RSS sampler requests 25ms intervals; sampling can miss short peaks and
RSS includes shared-page accounting. Whole-device GPU usage reached 8.77GB
(8.54GB with conditions reused), including baseline driver/other activity.
**No physical 8GB-card or 16GB-host test has been performed.** Four CPU threads
are a parallelism setting, not a utilization cap.

## Cache identity and portability

The generator manifest binds source identity, transformer config, backend
versions, fused layout, recipe and tensor inventory. Loading rejects missing or
extra tensors, incompatible layouts, nonfinite values and invalid scales. It
clones one CPU shard at a time; the pipeline then uses the existing model CPU
offload hooks. Packing and loading must use the pinned backend versions.

Condition reuse requires completed encoding records with matching model
identity, prompt, negative prompt, bit width, max token length and encoder
recipe. Embedding/mask shape, dtype and values are checked before generation.
Keep cache files together and use a fresh output directory. A local
`--source-version` is a caller-maintained identity, not content verification.
Changing source weights, tokenizer or pipeline requires a new version.

Packed artifacts remain local user-generated caches. Pipeline code, config,
scheduler and VAE are still needed. Original encoder/DiT weight downloads are
skipped for the corresponding supplied cache. The default path still performs
the original streamed conversion; both optimizations are explicit options.

## Validate a packed generator

From a configured checkout, this CPU-only check restores the DiT and compares
all tensors against the artifact. Replace the paths with your local source and
packed output; preserve the source identity used when packing.

```bash
CUDA_VISIBLE_DEVICES= python benchmarks/verify_generator_cache.py \
  --model /path/to/MOSS-Image-SFT --source-version sft-797fa8e678b4 \
  --packed packed-dit/generator --output outputs/dit-cold-parity.json
CUDA_VISIBLE_DEVICES= python -m pytest -q
```

CPU validation: 25 passed, 3 CUDA tests skipped. The wheel build includes both
cache modules. A fresh same-family reviewer independently repeated both CPU
commands, checked all eight rows against the original receipts, repeated
condition/noise/pixel comparisons, and checked imports/help from a newly unpacked
wheel. This is provisional independent execution, not cross-family review.
Earlier CUDA dequantization tests remain documented in
[engineering.md](engineering.md). The next opportunity is accelerating repeated
DiT execution; this cache change does not retain a loaded model across commands.
