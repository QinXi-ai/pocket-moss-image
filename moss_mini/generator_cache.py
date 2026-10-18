"""Version-bound, streamed safetensors storage for the fused INT8 DiT."""
import importlib.metadata
import json
from pathlib import Path
import time
import types

import torch

from . import reference

LAYOUT = {"family": "moss_image", "version": 1, "dense": "int8"}
BACKEND = {"orbitquant": "0.14.0", "comfy-kitchen": "0.2.37"}


def require_backend():
    for name, version in BACKEND.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Packed DiT requires {name}=={version}")


def validate_manifest(manifest, source_identity, config):
    if not source_identity or manifest.get("source_identity") != source_identity:
        raise ValueError("Packed DiT source identity differs from request")
    expected = {"format_version": 1, "status": "complete", "recipe": reference.RECIPE,
                "layout": LAYOUT, "backend": BACKEND, "transformer_config": config}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Packed DiT {key} mismatch")
    if not isinstance(manifest.get("shards"), dict) or not manifest["shards"]:
        raise ValueError("Packed DiT has no shards")


def safe_shard_path(directory, filename):
    if not isinstance(filename, str) or not filename.endswith('.safetensors'):
        raise ValueError("Invalid packed DiT shard filename")
    path = (Path(directory)/filename).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("Packed DiT shard escapes its directory")
    return path


def state_prefixes(model):
    keys = model.state_dict().keys()
    return sorted({'.'.join(k.split('.')[:2]) if k.startswith('layers.') else k.split('.')[0] for k in keys})


@torch.inference_mode(False)
@torch.no_grad()
def export_generator(model, directory, source, source_identity):
    from safetensors.torch import save_file
    require_backend()
    if not source_identity or getattr(model, 'oq_fused_layout', None) != LAYOUT:
        raise ValueError("Export requires a source-bound pure fused INT8 DiT")
    if any(t.device.type != 'cpu' for t in model.state_dict().values()):
        raise ValueError("Export the fused CPU model before installing GPU placement")
    if any(hasattr(m, '_hf_hook') for m in model.modules()):
        raise ValueError("Export before offload hooks are installed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"format_version": 1, "status": "writing", "recipe": reference.RECIPE,
                "source_identity": source_identity, "layout": LAYOUT, "backend": BACKEND,
                "transformer_config": json.loads((Path(source)/'transformer/config.json').read_text()),
                "shards": {}, "state_bytes": 0, "tensor_count": 0}
    def write():
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    write()
    for prefix in state_prefixes(model):
        module = model.get_submodule(prefix)
        state = {k: v.detach().contiguous() for k, v in module.state_dict().items()}
        filename = prefix.replace('.', '-')+'.safetensors'
        save_file(state, str(directory/filename))
        manifest['shards'][prefix] = {'file': filename, 'tensors': {
            k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in state.items()}}
        manifest['state_bytes'] += sum(v.numel()*v.element_size() for v in state.values())
        manifest['tensor_count'] += len(state)
        del state
    if manifest['tensor_count'] != len(model.state_dict()):
        raise ValueError("Packed DiT export inventory is incomplete")
    manifest['status'] = 'complete'
    write()
    return {k: manifest[k] for k in ('recipe', 'state_bytes', 'tensor_count', 'backend', 'layout')}


@torch.inference_mode(False)
@torch.no_grad()
def load_prepacked_generator(source, directory, source_identity):
    from safetensors import safe_open
    from orbitquant.config import OrbitQuantConfig
    from orbitquant.fused.install import prepare_skeleton, finalize
    from .generator_loading import empty_transformer, assemble_pipeline
    require_backend()
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((Path(source)/'transformer/config.json').read_text())
    validate_manifest(manifest, source_identity, config)
    model = empty_transformer(source)
    prepare_skeleton(model, LAYOUT, attention='sdpa',
                     config=OrbitQuantConfig(weight_bits=8, activation_bits=8, target_policy='universal'))
    if set(manifest['shards']) != set(state_prefixes(model)):
        raise ValueError("Packed DiT module inventory mismatch")
    loaded_count, loaded_bytes = 0, 0
    filenames = [row['file'] for row in manifest['shards'].values()]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Packed DiT duplicate shard filenames")
    for prefix in state_prefixes(model):
        module = model.get_submodule(prefix)
        expected = module.state_dict()
        row = manifest['shards'][prefix]
        if set(row['tensors']) != set(expected):
            raise ValueError("Packed DiT tensor inventory mismatch: "+prefix)
        state = {}
        with safe_open(str(safe_shard_path(directory, row['file'])), framework='pt', device='cpu') as handle:
            if set(handle.keys()) != set(expected):
                raise ValueError("Packed DiT shard tensor inventory mismatch: "+prefix)
            for name, target in expected.items():
                tensor = handle.get_tensor(name)
                declared = {'shape': list(target.shape), 'dtype': str(target.dtype)}
                if row['tensors'][name] != declared or tensor.shape != target.shape or tensor.dtype != target.dtype:
                    raise ValueError("Packed DiT tensor layout mismatch: "+prefix+'.'+name)
                # Detach mmap ownership one shard at a time. No full second model.
                value = tensor.clone()
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError("Packed DiT contains nonfinite state")
                if name.endswith('.scales') and not (value > 0).all():
                    raise ValueError("Packed DiT scales must be positive")
                state[name] = value
                loaded_count += 1
                loaded_bytes += value.numel()*value.element_size()
            del tensor, value
        module.load_state_dict(state, strict=True, assign=True)
        del state, expected
    if loaded_count != manifest['tensor_count'] or loaded_bytes != manifest['state_bytes']:
        raise ValueError("Packed DiT size/count mismatch")
    if any(t.is_meta for t in model.state_dict().values()):
        raise ValueError("Packed DiT left meta tensors")
    finalize(model)
    model.__dict__['_moss_opt_original_apply'] = model._apply
    model._apply = types.MethodType(reference._release_apply, model)
    pipe = assemble_pipeline(source, model.eval())
    return pipe, {'recipe': reference.RECIPE, 'loading': 'prepacked fused INT8; one CPU shard at a time',
                  'state_bytes': loaded_bytes, 'tensor_count': loaded_count, 'backend': BACKEND,
                  'layout': LAYOUT, 'seconds': time.perf_counter()-started, 'runtime_quantization': False,
                  'attention': 'sdpa'}
