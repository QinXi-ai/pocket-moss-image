# MOSS Mini

**Keep the original 9B text encoder. Shrink its weight storage.**

MOSS Mini 是一个独立的社区推理项目：保留 MOSS-Image 原始 9B 编码器、tokenizer、提示模板和条件表示，用 **INT6 编码器 + fused INT8 DiT** 降低推理占用。默认推荐 INT6；INT4、INT8 可切换。无需训练小编码器，也没有增加投影网络。

在 RTX 4090D 上，12 条开发提示的 INT6 完整流程测得 **6.64 GB 任务进程显存峰值、9.60 GB 进程树内存峰值**。这是 24GiB 显卡上的进程测量，**尚未在物理 8GB 显卡 / 16GB 主机上验证**；整卡占用还包含驱动和其他进程。

This is a community inference project, not an official OpenMOSS release. The recommended INT6 recipe reflects visual preference on 12 reused development prompts, not a measured universal quality win. [Evidence and limits](docs/results.md).

![BF16 / INT8 / INT6 / INT4, left to right](docs/comparison/comparison-1.jpg)

Columns: original BF16 encoder → INT8 → **INT6** → INT4. All use the same fused INT8 DiT. [All 12 comparisons](docs/comparison/index.html) (download/open the HTML locally), [sheet 2](docs/comparison/comparison-2.jpg), [sheet 3](docs/comparison/comparison-3.jpg).

## Install

Tested runtime: **Linux, Python 3.12, NVIDIA CUDA, PyTorch 2.9.1+cu128**. The fused backend is pinned to OrbitQuant 0.14.0 and Comfy Kitchen 0.2.37. Windows and other GPU architectures are not validated. Use a driver compatible with the required CUDA libraries.

```bash
git clone https://github.com/QinXi-ai/moss-mini.git
cd moss-mini
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m moss_mini --help
python -m pytest -q
```

The CLI locates the installed `nvidia-cublas` CUDA 13 library directory and exposes it to its subprocesses. This is needed by Comfy Kitchen's cuBLASLt INT8 backend; it does not replace PyTorch's CUDA 12.8 runtime. No ComfyUI installation is required.

## Generate with INT6

```bash
CUDA_VISIBLE_DEVICES=0 moss-mini generate \
  --prompt 'a photo of a yellow skateboard and an orange computer mouse' \
  --output outputs/skateboard
```

Output: `outputs/skateboard/image.png`, saved conditions/noise, and JSON completion records. Each run needs a new output directory. The default model is [`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), pinned to revision `797fa8e678b4f0a766271bf5103e6cd7a2faa999`. The first invocation downloads the original weights and pipeline code from Hugging Face; allow tens of GB of disk space. Pipeline code is executed from this pinned model snapshot.

Default settings: 1024×1024, 50 steps, seed 42, CFG 3, shift 5, unconditional skip layer 22, at most 256 text tokens, batch size 1, SDPA attention, four CPU threads. Long prompts are truncated by the original tokenizer. Thread limits reduce CPU parallelism; they are not a CPU-utilization guarantee.

The encoder process exits before the generator starts. Linear weights are packed, while embedding/norm/convolution state stays BF16. Each Linear temporarily reconstructs a BF16 matrix for GEMM. **INT6 refers to weight storage, not a fused INT6 compute kernel.** The DiT uses fused W8A8 computation and model CPU offload; source DiT tensors are loaded one block at a time.

An experimental `--encoder-dequant triton` option fuses unpacking and scaling. It reduced warm 256-token encoder forward time by about 21% in a bounded test, with exact conditions and image replay; it did not measurably improve whole-command latency or VRAM peak. The default remains PyTorch. See the [engineering profile and next priorities](docs/engineering.md).

## Pack once, reuse

Avoid repeating CPU quantization on every run:

```bash
CUDA_VISIBLE_DEVICES=0 moss-mini pack --bits 6 --output packed-int6
CUDA_VISIBLE_DEVICES=0 moss-mini generate \
  --packed packed-int6/encoder \
  --prompt 'a photo of a yellow skateboard and an orange computer mouse' \
  --output outputs/skateboard-packed
```

Packing also needs a CUDA GPU in this release. The packed INT6 encoder contains 7.66 GB of tensor state; leave additional disk space for metadata and the original model. The manifest is relocatable and binds model identity, configuration and quantization recipe. Do not reuse an artifact with different weights or tokenizer files.

For an existing local snapshot:

```bash
moss-mini generate --model /path/to/MOSS-Image-SFT \
  --source-version sft-797fa8e678b4 \
  --prompt 'a photo of three tennis rackets' --output outputs/rackets
```

Use the same `--model` and `--source-version` for packing and generation. A local identity is caller-declared: change the version whenever weights, tokenizer or pipeline code change. It does not cryptographically verify content. Original snapshots remain necessary for tokenizer/configuration and generator weights; packed artifacts are not standalone model distributions. With `--packed`, Hub downloads exclude the unused original text-encoder weight shards.

## Other precisions

| Option | Default CPU-offloaded leading layers | Intended use |
|---|---:|---|
| `--bits 6` | 20 | Recommended visual/storage compromise |
| `--bits 4` | 12 | Smaller encoder state; larger visual changes on some prompts |
| `--bits 8` | 16 | Higher precision reference |

The embedding is always offloaded to CPU. Override with `--offload-layers N` only if you can measure your own resource envelope. More GPU placement can reduce RAM but increases VRAM. The default INT6 and INT4 placements survived a synthetic 256-token encoder stress test. We do not claim longer contexts, larger batches or other resolutions stay inside the same budget.

## Results and reproducibility

See [measured results, failed configurations and protocol](docs/results.md), [portable metrics](docs/metrics.json) and [release validation](docs/validation.md). No model weights, development caches or private server configuration are included.

The comparison images are JPEG previews of actual generated images. They illustrate differences, not a blinded ranking or a held-out benchmark. Original BF16 outputs also contain mistakes. The earlier 4B replacement did not recover the desired visual behavior in our experiments, which motivated preserving the original encoder and quantizing its weights.

## Credits and license

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Built on [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), [OrbitQuant](https://pypi.org/project/orbitquant/), [Comfy Kitchen](https://pypi.org/project/comfy-kitchen/), Diffusers, Transformers and Accelerate. Model weights and downloaded pipeline code retain their upstream licenses. Generated previews are provided for comparison; the code license does not grant rights to third-party marks or model weights.
