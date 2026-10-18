# Release validation — 2026-10-18

The extracted package was tested in the existing Linux/Python 3.12 environment
on a 24GiB RTX 4090D. This is warm-environment validation, not a clean-machine
installation or physical 8GB-card certification.

## Completed checks

- CPU tests cover all three bit widths against independently reconstructed
  quantization, a known-byte INT6 packing example, zero rows, malformed scale
  rejection, manifest identity/precision checks, local source identity, CLI
  defaults and context validation. The final suite passed **9 tests**, including
  a subprocess dispatch test added with the help-only polish.
- A fresh documentation-check agent independently executed `python -m moss_mini
  --help` and `python -m pytest -q` before the final help-only polish: 8 tests
  passed. No missing external imports or conflicting declared dependencies were
  found by inspection of package code and the pinned upstream pipeline.
- The final package built successfully as a wheel, installed into a separate
  target directory, and its `moss-mini --help` console entrypoint ran outside
  the source checkout.
- Full `pack`, prepacked `generate`, and on-the-fly `generate` commands completed
  with exit code 0, with CPU threads limited to four.
- Saved positive/negative conditions, masks and initial noise match exactly
  between the two generation paths. PNG pixels also match exactly, including
  the corresponding image from the original research runner.
- The comparison itself was CPU-only and did not initialize CUDA.

Prompt: `a photo of a yellow skateboard and an orange computer mouse`.
INT6/g64, 20 CPU-offloaded layers, 1024×1024, seed 42, 50 steps, CFG 3,
shift 5, unconditional skip layer 22. No GPU retry or alternate backend was
needed. JSON evidence: [release-smoke.json](release-smoke.json).

| Public command | Task GPU peak GB | Tree RSS peak GB | Elapsed s |
|---|---:|---:|---:|
| Pack INT6 | 2.538 | 6.762 | 83.612 |
| Generate from packed INT6 | 6.637 | 8.499 | 57.502 |
| Generate with on-the-fly INT6 | 6.637 | 7.484 | 90.679 |

Values are decimal GB and sampled process measurements. Whole-device peak during
generation was 8.486 GB including other device memory. These single-prompt
timings include load/conversion and process startup; they are not a throughput
benchmark. Different cache/order effects can change them.

The full 12-prompt comparisons predate the packaging change; see
[results.md](results.md). One exact image replay verifies this entrypoint's
integration, not a broad quality or hardware compatibility claim.

## Environment

Python 3.12.3; torch 2.9.1+cu128; diffusers 0.38.0; transformers 5.17.0;
accelerate 1.15.0; safetensors 0.8.0; huggingface-hub 1.32.0;
OrbitQuant 0.14.0; Comfy Kitchen 0.2.37; nvidia-cublas 13.0.0.19.
Backend packages were reused from a separately installed dependency overlay.
The README describes a normal pip installation; that clean installation path
was inspected but not rerun by downloading a second complete environment.
