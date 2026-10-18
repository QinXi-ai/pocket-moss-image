"""Read original joint blocks from meta directly to the reference CUDA device.

The complete original BF16 DiT is never materialized on CPU. OrbitQuant group
construction, row order, prepare/attach and runtime installation are unchanged.
This is a research loading path, separate from the production default.
"""
from __future__ import annotations

import importlib.metadata
import time
import types

import torch
from torch import nn

REVISION = "20261018-direct-shard-generator-loading-v1"


def _bytes(module):
    return sum(value.numel() * value.element_size() for value in module.state_dict().values())


@torch.inference_mode(False)
@torch.no_grad()
def build_streamed_fused_int8(pipe, config, *, stage_callback=None, materializer=None):
    """Pure INT8 groups, from fresh meta joint blocks and CPU nonblock modules.

    CUDA numerical equivalence is intended by construction, not claimed from
    CPU testing alone. A real-model GPU witness must compare every q/scale/bias and
    retained tensor with the reference CUDA conversion before adoption.
    """
    from . import reference
    from orbitquant.config import OrbitQuantConfig
    from orbitquant.fused.families import adapter_for_class
    from orbitquant.fused.groups import GROUP_TYPES, _resolve
    from orbitquant.fused.install import LAYOUT_VERSION, _attach, _group_kind, _install_model
    from orbitquant.fused.runtime import FusedRuntime

    if not isinstance(config, dict) or set(config) - {"attention", "conversion_device"}:
        raise ValueError("Streamed loading supports only attention and conversion_device; no mixed/calibration recipe")
    if importlib.metadata.version("orbitquant") != "0.14.0":
        raise RuntimeError("This loading experiment is pinned to OrbitQuant0.14.0 private fuse helpers")
    attention = config.get("attention", "sdpa")
    if attention not in {"sdpa", "flash", "cudnn"}:
        raise ValueError("Retain a reference BF16 attention backend")
    conversion_device = torch.device(config.get("conversion_device", "cuda:0"))
    if conversion_device.type not in {"cpu", "cuda"}:
        raise ValueError("Conversion device must be CPU or the reference CUDA device")
    if stage_callback is not None and not callable(stage_callback):
        raise TypeError("stage_callback must be callable or None")
    model = pipe.transformer
    if type(model).__name__ != "MOSSImageTransformer" or hasattr(model, "oq_runtime"):
        raise ValueError("Requires a fresh original MOSSImageTransformer")
    if any(value.device.type not in ("cpu", "meta") for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Direct loading starts with meta joint blocks and CPU nonblocks")
    if any(hasattr(module, "_hf_hook") for module in model.modules()):
        raise ValueError("Convert before installing Accelerate/offload hooks")
    adapter = adapter_for_class(type(model).__name__)
    paths = list(adapter.blocks(model))
    if adapter.FAMILY != "moss_image" or paths != [(f"layers.{index}", "block") for index in range(len(model.layers))]:
        raise ValueError("Unexpected pinned MOSS fused block layout")
    original_device = next(model.parameters()).device
    before_bytes = _bytes(model)
    started = time.perf_counter()
    qconfig = OrbitQuantConfig(weight_bits=8, activation_bits=8, target_policy="universal")
    runtime = FusedRuntime.from_config(qconfig, attention=attention)
    block_receipts = []
    for index, (path, kind) in enumerate(paths):
        if stage_callback is not None:
            stage_callback(f"convert_joint_block_{index:03d}")
        block = _resolve(model, path)
        source_bytes = _bytes(block)
        if materializer is None:
            raise ValueError('Direct loading requires an explicit pinned-source materializer')
        materializer(index, block, conversion_device)
        block_started = time.perf_counter()
        cuda = conversion_device.type == "cuda"
        allocated_before = torch.cuda.memory_allocated(conversion_device) if cuda else None
        try:
            block.to(conversion_device)
            modules = nn.ModuleDict()
            for name, spec in adapter.groups(kind, block).items():
                group_cls = GROUP_TYPES[_group_kind(block, spec, "int8")]
                modules[name] = group_cls.from_sources(block, list(spec.sources), swiglu=spec.swiglu, activation_bits=None)
            prepare = getattr(adapter, "prepare", None)
            if prepare is not None:
                prepare(block, kind)
            _attach(model, block, adapter, kind, modules, runtime)
            if cuda:
                torch.cuda.synchronize(conversion_device)
            row = {"path": path, "source_bf16_state_bytes": source_bytes,
                                   "fused_state_bytes": _bytes(block), "conversion_device": str(conversion_device),
                                   "returned_device": "cpu", "allocated_before_bytes": allocated_before,
                                   "allocated_peak_bytes": torch.cuda.max_memory_allocated(conversion_device) if cuda else None,
                                   "reserved_peak_bytes": torch.cuda.max_memory_reserved(conversion_device) if cuda else None,
                                   "cuda_peak_scope": "since caller stage reset; pass Stages.transition for per-block windows; helper never resets caller counters"}
        finally:
            # No previous/next block and no root DiT is moved onto CUDA here.
            block.to(original_device)
        if cuda:
            torch.cuda.synchronize(conversion_device)
        row.update(seconds=time.perf_counter()-block_started,
                   allocated_after_cpu_restore_bytes=torch.cuda.memory_allocated(conversion_device) if cuda else None,
                   allocated_peak_bytes=torch.cuda.max_memory_allocated(conversion_device) if cuda else None,
                   reserved_peak_bytes=torch.cuda.max_memory_reserved(conversion_device) if cuda else None)
        block_receipts.append(row)
        del block, modules
    layout = {"family": adapter.FAMILY, "version": LAYOUT_VERSION, "dense": "int8"}
    model.oq_fused_layout = layout
    model.__dict__["oq_runtime"] = runtime
    _install_model(model, adapter)
    model.__dict__["_moss_opt_original_apply"] = model._apply
    model._apply = types.MethodType(reference._release_apply, model)
    model.to(original_device)
    if any(torch.is_inference(value) for value in model.state_dict().values()):
        raise RuntimeError("Persistent inference-mode tensors are incompatible with normal pipeline no_grad")
    precisions, storage = {}, {}
    for index, block in enumerate(model.layers):
        for name, group in block.oq_fused.items():
            key = f"layers.{index}.{name}"
            precisions[key], storage[key] = group.kind, _bytes(group)
    return {"adapter": "moss-fused-int8-streamed-loading", "implementation_revision": REVISION,
            "reference_implementation_revision": reference.REVISION, "orbitquant": "0.14.0", "orbitquant_commit": reference.ORBIT_COMMIT,
            "recipe": reference.RECIPE, "weight_storage": "int8",
            "activation_quantization": "dynamic row-absmax INT8 on INT8 groups; BF16 on BF16 groups",
            "rotation": "none; distinct from Comfy ConvRot INT8", "accumulator": "INT32 on INT8 groups",
            "attention": attention, "fused_layout": layout, "group_precisions": precisions, "bf16_groups": [],
            "group_storage_bytes": storage, "model_state_bytes_before": before_bytes, "model_state_bytes_after": _bytes(model),
            "conversion_device": str(conversion_device), "conversion_seconds": time.perf_counter()-started, "calibration": None,
            "loading_strategy": "Meta original DiT; read one source joint block directly to conversion GPU; return fused block to CPU",
            "blocks": block_receipts, "maximum_source_block_state_bytes": max(row["source_bf16_state_bytes"] for row in block_receipts),
            "whole_bf16_transformer_moved_to_cuda": False,
            "cuda_numerical_equivalence_verified": False,
            "equivalence_scope": "same Orbit group conversion operators/order/shapes and same conversion device; CPU exact-state tests are not a CUDA witness",
            "offload": "same reference root _apply release hook; caller installs placement after conversion"}


def streamed_fused_int8(pipe, config):
    """Explicit direct module:callable backend; separate from production defaults."""
    from .generator_loading import materializer_for
    materializer = materializer_for(pipe.transformer)
    return build_streamed_fused_int8(pipe, config, materializer=materializer)
