# Third optimization pass: GEMM and attention backends

This pass measured the two largest kernel categories identified in the
[original-block profile](optimization-round2.md#where-the-original-block-spends-time).
**None of the tested replacements demonstrated a useful generation speedup.**
The published runtime remains `811313a`; this update adds evidence only.

The reference uses the original 9B INT6/g64 encoder, prepacked fused INT8 DiT,
CPU vocabulary lookup with zero offloaded encoder layers, and exact request
caching. Measurements used an RTX 4090D 24GiB, four CPU threads, batch one,
1024×1024, 50 steps, CFG 3, shift 5, seed 42, and the same skateboard/mouse
development prompt. These are bounded measurements on a shared host, not a
held-out quality benchmark. GB is decimal.

[Selected numeric receipts](optimization-round3-metrics.json) include operator
timings, unsuccessful candidates, tensor comparisons and sampled resource peaks.
Private source snapshots and complete receipts remain in the research workspace.

## Original Triton INT8 GEMM tiles

The actual implementation is OrbitQuant's Triton INT8 GEMM. We varied bounded
launch configurations while reusing its original computation body, INT32
accumulation, BF16 rounding, scales, SiTU and residual-gate epilogues. SiTU
retained its paired gate/up layout with a 256-column tile.

Four real first-step first-block projections were captured separately for
positive conditioning (4117 tokens) and negative conditioning (4106 tokens).
All **74 tile/projection candidates** matched their captured outputs exactly.
Input-preservation checks and independent complete-block equality also passed.

| Independent resident first-block test | Original ms | Selected tiles ms | Time change |
|---|---:|---:|---:|
| Positive conditioning | 4.155 | 4.226 | +1.69% |
| Negative conditioning | 4.183 | 4.163 | −0.48% |

Each comparison used ten warmed A-B-B-A rounds. The candidate was faster in
1/10 positive rounds and 9/10 negative rounds. These opposite, small effects
do not support adopting the combined dispatch; it was not advanced to complete
generation.

The search selected each candidate's lowest absolute median across a sequential
search. Baseline drift makes that selection optimistic: for example, negative
QKV reference medians ranged from 0.488 to 0.558ms, and two selected negative
tiles were slower than their own paired references. The independent block check
was therefore essential. These results do not establish that every possible
tile or architecture-specific implementation is exhausted.

## Vendor INT8 multiply plus a separate epilogue

A second prototype used `torch._int_mm(out=...)`, followed by a Triton epilogue
written to reproduce the original arithmetic. Timed calls included INT32
materialization, the epilogue and any required padding copies. Persistent
workspace allocation and zero-tail initialization were outside warmed timing;
this makes the comparison more favorable than allocating on every call.

| Projection | Positive original / candidate ms | Negative original / candidate ms | Output |
|---|---:|---:|---|
| QKV | 0.584 / 0.789 | 0.641 / 0.829 | Exact |
| SiTU MLP | 1.206 / 1.291 | 1.204 / 1.288 | Exact |
| Attention output + residual | Not timed | Not timed | Different |
| MLP output + residual | Not timed | Not timed | Different |

The exact routes lost in every one of their twenty A-B-B-A rounds. Both
residual-gate projections failed equality with direct and row-padded routes;
maximum absolute differences were 8 and 64 respectively. The cause was not
isolated, so source-level similarity is not claimed as numerical equivalence.
No route was both exact and faster, and this prototype was not adopted.

## BF16 attention backends

Actual Q/K/V had 24 heads, head dimension 128, and 4117/4106 tokens. The original
runtime, default fused SDPA and forced Flash all executed the same observed
PyTorch Flash kernel and produced identical outputs. EFFICIENT was slower;
copying Q/K/V into contiguous BHSD buffers also made the tested backends slower.

Native-stride cuDNN was the only promising operator candidate: attention was
about 2.8% faster than forced Flash, and the resident first block was 2.7–3.0%
faster. Its output differed, however: first-block relative L2 differences were
0.00182 positive and 0.00107 negative. These measurements excluded loading,
offload, other blocks, later steps and VAE decoding.

We therefore tested cuDNN through the complete generator using the same saved
conditions and initial noise. Both cases used exact request caching; only the
existing runtime attention selector changed. All values below are measured
in chronological A-B-B-A order.

| Generator run | Denoise + decode s | Complete generator process s | Task GPU GB | Tree RSS GB |
|---|---:|---:|---:|---:|
| Original Flash A | 24.711 | 35.251 | 6.407 | 7.345 |
| cuDNN A | 25.371 | 36.477 | 6.409 | 7.388 |
| cuDNN B | 25.476 | 35.994 | 6.409 | 7.385 |
| Original Flash B | 24.949 | 36.100 | 6.407 | 7.344 |

cuDNN averaged about **2.4% more denoise/decode time** across these two runs per
backend. The small resident-block improvement did not survive complete
generation. These generator processes load the packed DiT and saved conditions;
they do **not** load or run the encoder. Complete-process timing comes from the
outer resource wrapper and includes imports, output writes and process teardown.

Final Flash latents and PNG pixels matched across repeats. cuDNN changed the
final latent (relative L2 0.415) and image (pixel PSNR 19.73dB against Flash).
The broad scene remained similar, but the skateboard illustration and other
details changed. PSNR measures pixel difference, not quality. Neither an image
quality gain nor quality equivalence is established by this one prompt.
The attention selector was restored and request-cache accounting was zero on
exit. cuDNN is not exposed as a new CLI option.

## Resource and release scope

A fresh complete encode-and-generate CLI reference took **48.057s**, with
6.598GB sampled task GPU memory and 7.365GB process-tree RSS. Its PNG matched
the generator-only reference. The original 9B conditioning was preserved.

All measured commands completed successfully and stayed below decimal 8GB
sampled task GPU memory and 16GB sampled tree RSS. Sampling at 25ms can miss
short peaks; RSS sums include shared pages. Whole-device usage includes driver
and other activity. **A physical 8GB GPU / 16GB host remains untested.**

No runtime, dependency or default changed. The existing exact request cache and
opt-in Cache-DiT remain available. No new Cache-DiT/backend composition claim is
made because none of these candidates was adopted. Further work would need a
different implementation or a measured pipeline bottleneck; changing backend
names or choosing an isolated fastest operator is insufficient evidence.
