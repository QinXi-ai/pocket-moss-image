"""Encode and generate in separate processes to release all encoder memory."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .source import MODEL, REVISION, resolve_source

OFFLOAD = {4: 12, 6: 20, 8: 16}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description="MOSS Mini: original 9B encoder, packed weights, fused INT8 DiT")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("pack", "generate"):
        q = sub.add_parser(name)
        q.add_argument("--model", default=MODEL)
        q.add_argument("--revision", default=REVISION)
        q.add_argument("--source-version", help="Required portable identity for a local model directory")
        q.add_argument("--offline", action="store_true")
        q.add_argument("--bits", type=int, choices=(4, 6, 8), default=6)
        q.add_argument("--offload-layers", type=int, help="CPU-offloaded leading layers; default 20/12/16 for INT6/4/8")
        q.add_argument("--cpu-threads", type=int, default=4)
        q.add_argument("--output", type=Path, required=True, help="New directory; existing directories are never overwritten")
        if name == "generate":
            q.add_argument("--encoder-dequant", choices=("torch", "triton"), default="torch",
                           help="Experimental Triton fused unpack/scale, or reference torch (default)")
            q.add_argument("--packed", type=Path, help="Prepacked encoder directory from moss-mini pack")
            q.add_argument("--prompt", required=True)
            q.add_argument("--negative-prompt", default="")
            q.add_argument("--seed", type=int, default=42)
            q.add_argument("--width", type=int, default=1024)
            q.add_argument("--height", type=int, default=1024)
            q.add_argument("--steps", type=int, default=50)
            q.add_argument("--guidance-scale", type=float, default=3.0)
            q.add_argument("--shift", type=float, default=5.0)
            q.add_argument("--max-sequence-length", type=int, default=256)
    return p


def worker(phase, plan_path):
    import torch
    from safetensors.torch import load_file, save_file
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    torch.set_num_threads(plan["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("This release requires Linux and a CUDA GPU")
    out = plan_path.parent
    started = time.perf_counter()
    if phase in ("pack", "encode"):
        from .encoder import load_streamed_pipeline, load_prepacked_pipeline, set_dequant_backend
        kwargs = dict(bits=plan["bits"], group_size=64, offload_layers=plan["offload_layers"],
                      source_identity=plan["source_identity"], device="cuda:0")
        if phase == "pack":
            kwargs["export_packed"] = str(out / "encoder")
        if plan.get("packed"):
            pipe, conversion = load_prepacked_pipeline(plan["source"], packed_dir=plan["packed"], **kwargs)
        else:
            pipe, conversion = load_streamed_pipeline(plan["source"], **kwargs)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        if phase == "encode":
            set_dequant_backend(pipe, plan.get("encoder_dequant", "torch"))
            conversion["dequant_backend"] = plan.get("encoder_dequant", "torch")
            tensors = {}
            with torch.no_grad():
                for prefix, prompt in (("", plan["prompt"]), ("negative_", plan["negative_prompt"])):
                    embeds, mask = pipe.encode_prompt(prompt, plan["max_sequence_length"])
                    if not torch.isfinite(embeds).all():
                        raise ValueError("Encoder produced nonfinite conditions")
                    tensors[prefix + "prompt_embeds"] = embeds.cpu().contiguous()
                    tensors[prefix + "prompt_attention_mask"] = mask.cpu().contiguous()
                    del embeds, mask
            save_file(tensors, str(out / "conditions.safetensors"))
        write_json(out / f"{phase}.json", {"status": "complete", "seconds": time.perf_counter()-started,
                   "timings": {"load_seconds": loaded-started, "encode_and_save_seconds": time.perf_counter()-loaded},
                   "conversion": conversion})
        return
    from .generator_loading import load_direct_stage
    from .generator import streamed_fused_int8
    pipe = load_direct_stage(plan["source"], "generate")
    staged = time.perf_counter()
    conversion = streamed_fused_int8(pipe, {"attention": "sdpa", "conversion_device": "cuda:0"})
    converted = time.perf_counter()
    pipe.enable_model_cpu_offload(device="cuda:0")
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=plan["shift"])
    tensors = load_file(str(out / "conditions.safetensors"))
    shape = (1, pipe.transformer.config.latent_channels, plan["height"]//16, plan["width"]//16)
    noise = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(plan["seed"]), dtype=torch.float32)
    save_file({"initial_noise": noise}, str(out / "initial_noise.safetensors"))
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    with torch.no_grad():
        image = pipe(**tensors, latents=noise, width=plan["width"], height=plan["height"],
                     num_inference_steps=plan["steps"], guidance_scale=plan["guidance_scale"],
                     uncond_skip_layers=[22], num_images_per_prompt=1, output_type="pil").images[0]
    image.save(out / "image.png")
    write_json(out / "generate.json", {"status": "complete", "seconds": time.perf_counter()-started,
               "timings": {"load_seconds": staged-started, "convert_seconds": converted-staged,
                           "prepare_seconds": prepared-converted, "denoise_decode_save_seconds": time.perf_counter()-prepared},
               "conversion": conversion})


def child_environment(threads):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    env["MALLOC_ARENA_MAX"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        dist = importlib.metadata.distribution("nvidia-cublas")
        lib = Path(dist.locate_file("nvidia/cu13/lib"))
        if lib.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    except importlib.metadata.PackageNotFoundError:
        pass
    return env


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        internal = argparse.ArgumentParser()
        internal.add_argument("phase", choices=("pack", "encode", "generate"))
        internal.add_argument("plan", type=Path)
        args = internal.parse_args(argv[1:])
        worker(args.phase, args.plan)
        return
    args = parser().parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    offload = OFFLOAD[args.bits] if args.offload_layers is None else args.offload_layers
    if not 0 <= offload <= 32:
        raise ValueError("--offload-layers must be between 0 and 32")
    if args.command == "generate":
        if any(n < 16 or n % 16 for n in (args.width, args.height)):
            raise ValueError("Image dimensions must be positive multiples of 16")
        if args.steps < 1 or not 1 <= args.max_sequence_length <= 256 or not 0 <= args.seed < 2**63:
            raise ValueError("Require positive steps, 1..256 tokens and a nonnegative 63-bit seed")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "phases": []}
    started = time.perf_counter()
    try:
        source, identity = resolve_source(args.model, args.revision, args.source_version, "encode", args.offline,
                                          encoder_weights=not bool(getattr(args, "packed", None)))
        if args.command == "generate":
            generated_source, gen_identity = resolve_source(args.model, args.revision, args.source_version, "generate", args.offline)
            if generated_source != source or gen_identity != identity:
                raise ValueError("Encoder and generator must use the same pinned source")
        plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        if plan.get("packed"):
            plan["packed"] = str(Path(plan["packed"]).expanduser().resolve())
        plan.update(source=str(source), source_identity=identity, offload_layers=offload)
        write_json(out / "plan.json", plan)
        phases = ("pack",) if args.command == "pack" else ("encode", "generate")
        for phase in phases:
            print(f"[{phase}] INT{args.bits}, {offload} encoder layers offloaded", flush=True)
            row = {"phase": phase, "status": "running"}
            record["phases"].append(row)
            write_json(out / "run.json", record)
            result = subprocess.run([sys.executable, "-m", "moss_mini", "_worker", phase, str(out/"plan.json")],
                                    env=child_environment(args.cpu_threads), stdin=subprocess.DEVNULL)
            row.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed")
            if result.returncode:
                raise RuntimeError(f"{phase} failed (exit {result.returncode}); output retained at {out}")
        record["status"] = "complete"
        print(f"Saved: {out / ('encoder' if args.command == 'pack' else 'image.png')}", flush=True)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["seconds"] = time.perf_counter()-started
        write_json(out / "run.json", record)
