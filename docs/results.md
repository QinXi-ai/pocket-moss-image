# Measured results

## Protocol

12 reused development prompts, one seed (42), 1024×1024, 50 steps, CFG 3,
shift 5, unconditional skip layer 22. Identical original FP32 initial noise and
fused INT8 DiT across all four columns. Only the text encoder precision and
placement change. Original tokenizer, prompt template, 32 text layers and
4096-wide final conditions are preserved. No learned projection or calibration.

INT8 uses symmetric per-output-row absmax. INT6/INT4 use symmetric absmax in
groups of 64 input columns, genuinely packed to 6/4 bits, with FP32 scales.
248 standard Linear modules are quantized; other state remains BF16. Encoder
GEMMs reconstruct weights in FP32 and execute in BF16.

## Resource measurements

Decimal GB (10^9 bytes). Measurements use task-process NVML usage and sampled
process-tree RSS on a **24GiB RTX 4090D**. The full workflow runs encoding and
generation as sequential subprocesses. Runtime covers all 12 development images
and is descriptive, not a controlled throughput comparison.

| Encoder | CPU layers | Encoder state GB | Full workflow task GPU peak GB | Tree RSS peak GB | 12-image runtime s |
|---|---:|---:|---:|---:|---:|
| INT8 row | 16 | 8.961 | 6.413 | 9.588 | 371.166 |
| **INT6 g64, prepacked** | **20** | **7.658** | **6.637** | **9.604** | **321.949** |
| INT4 g64, prepacked | 12 | 5.928 | 6.413 | 9.510 | 331.059 |

Original text-state size: 15,873,369,088 bytes. This excludes the generator.
All listed workflows completed with exit code 0 and no resource sampling errors.
These numbers describe the research runner; the extracted public CLI has its own
[release smoke test](validation.md).

### 256-token encoder-only stress

Two forwards on one synthetic prompt explicitly truncated to 256 tokens:

| Encoder placement | Task GPU peak GB | Tree RSS peak GB |
|---|---:|---:|
| INT6, 20 CPU layers | 6.627 | 8.273 |
| INT4, 12 CPU layers | 6.977 | 6.328 |

This tests resource usage, finite outputs and condition shapes, not image quality.
No physical 8GB GPU or 16GB host was tested. Whole-card usage includes memory
outside the task and exceeded decimal 8GB in some runs. RSS is not whole-host RAM,
and sampled peaks can miss short transients. Four CPU threads are a parallelism
setting, not a measured CPU utilization cap.

## Negative results retained

- INT6 with only 12 leading layers offloaded reached **8,044,675,072 bytes** of
  task GPU memory during the first export/full development run. Exit 0 did not
  mean it satisfied a decimal 8GB target.
- INT4 with no leading layers offloaded fit the short development prompts, but
  reached **8,449,425,408 bytes** on the 256-token encoder stress. It is not the
  recommended default.
- An initial generator loader attempted `.to()` on meta weights and failed.
  The release uses the direct shard loader and matching backend together.
- Early research records mislabeled some low-bit encoder metadata as INT8.
  Recipes and actual bits were checked; release records use explicit bit values.

## Equivalence and quality

Across all 12 development prompts, streamed INT8 matched the prior INT8 loader
exactly in positive/negative conditions, masks and PNG pixels. Prepacked INT4 and
INT6 likewise matched their respective on-the-fly recipes exactly despite
placement changes. This is **within-recipe loading equivalence**, not equivalence
to the BF16 encoder and not a quality improvement claim.

INT8 generally stays closer to the original compositions. INT6 preserves many
objects and relations with some compositional changes. INT4 changes composition
more often and sometimes confuses objects or geometry. The original BF16 column
also makes mistakes, so resemblance alone is not correctness. INT6 is recommended
based on this project's visual preference and resource tradeoff. No held-out
win rate, FID, or broad quality superiority is established. The reserved 48×2
final evaluation was not run for this release.

## Gallery

The [HTML gallery](comparison/index.html) includes every development prompt and
all four outputs. Open the downloaded HTML locally. Each sheet contains four
rows, ordered BF16 / INT8 / INT6 / INT4:

![Rows 1–4](comparison/comparison-1.jpg)
![Rows 5–8](comparison/comparison-2.jpg)
![Rows 9–12](comparison/comparison-3.jpg)
