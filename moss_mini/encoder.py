"""Stream original 9B safetensors into packed weights without a full BF16 model."""
import ctypes
import gc
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pack_codes(q, bits):
    q = q.to(torch.uint8).contiguous()
    if bits == 8:
        return q
    if bits == 4:
        return q[:, 0::2] | (q[:, 1::2] << 4)
    if bits == 6:
        a, b, c, d = (q[:, i::4] for i in range(4))
        return torch.stack((a | ((b & 3) << 6), (b >> 2) | ((c & 15) << 4),
                            (c >> 4) | (d << 2)), dim=-1).flatten(1)
    raise ValueError(bits)


def unpack_codes(q, bits):
    if bits == 8:
        return q
    if bits == 4:
        return torch.stack((q & 15, q >> 4), dim=-1).flatten(1)
    if bits == 6:
        a, b, c = (q[:, i::3] for i in range(3))
        return torch.stack((a & 63, (a >> 6) | ((b & 15) << 2),
                            (b >> 4) | ((c & 3) << 4), c >> 2), dim=-1).flatten(1)
    raise ValueError(bits)


class PackedLinear(nn.Module):
    def __init__(self, qweight, scale, bits, group_size, in_features, bias=None):
        super().__init__()
        if bits not in (4, 6, 8) or group_size < 1 or in_features < 1 or in_features % group_size or in_features % 4:
            raise ValueError('Invalid packed precision/group layout')
        if qweight.dtype != torch.uint8 or qweight.ndim != 2 or qweight.shape[0] < 1 or qweight.shape[1] != in_features*bits//8:
            raise ValueError('Packed weight shape/dtype mismatch')
        if scale.dtype != torch.float32 or scale.shape != (qweight.shape[0], in_features//group_size):
            raise ValueError('Packed scale shape/dtype mismatch')
        if scale.device != qweight.device or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('Packed scales must be finite, positive and colocated')
        if bias is not None and (bias.dtype != torch.bfloat16 or bias.shape != (qweight.shape[0],)
                                 or bias.device != qweight.device or not torch.isfinite(bias).all()):
            raise ValueError('Packed bias shape/dtype/device mismatch')
        self.bits, self.group_size, self.in_features = bits, group_size, in_features
        self.out_features = qweight.shape[0]
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.register_parameter('bias', None if bias is None else nn.Parameter(bias, requires_grad=False))
        self.eval()

    @classmethod
    def from_weight(cls, weight, bits, group_size, device, bias=None):
        if bits not in (4, 6, 8) or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError('Expected BF16 matrix and 4/6/8 bits')
        rows, cols = weight.shape
        group_size = cols if bits == 8 else group_size
        if cols % group_size or cols % 4:
            raise ValueError('Group/packing alignment mismatch')
        maximum = 2 ** (bits-1)-1
        packed = torch.empty(rows, cols*bits//8, dtype=torch.uint8, device=device)
        scales = torch.empty(rows, cols//group_size, dtype=torch.float32, device=device)
        for start in range(0, rows, 128):
            w = weight[start:start+128].float().reshape(-1, cols//group_size, group_size)
            if not torch.isfinite(w).all():
                raise ValueError('Nonfinite source weight')
            scale = w.abs().amax(-1).div_(maximum)
            scale.masked_fill_(scale == 0, 1.0)
            q = w.div_(scale[..., None]).round_().clamp_(-maximum, maximum).reshape(-1, cols)
            codes = (q + maximum).to(torch.uint8)
            packed[start:start+128].copy_(pack_codes(codes, bits))
            scales[start:start+128].copy_(scale)
        return cls(packed, scales, bits, group_size, cols, bias)

    def forward(self, inputs):
        if inputs.dtype != torch.bfloat16 or inputs.requires_grad or self.training:
            raise ValueError('Inference-only BF16 input required')
        if inputs.device != self.qweight.device or self.scale.dtype != torch.float32:
            raise ValueError('Placement or scale dtype changed')
        maximum = 2 ** (self.bits-1)-1
        with torch.no_grad():
            w = unpack_codes(self.qweight, self.bits).float().sub_(maximum)
            w = w.reshape(self.out_features, -1, self.group_size)
            w.mul_(self.scale[..., None])
            return F.linear(inputs, w.reshape(self.out_features, self.in_features).to(torch.bfloat16), self.bias)


def recipe(bits, group_size):
    return f'moss-original-streamed-packed-int{bits}-' + ('row' if bits == 8 else f'g{group_size}') + '-bf16-gemm-v1'


def validate_manifest(manifest, config, source_identity, bits, group_size):
    if (manifest.get('format_version') != 1 or manifest.get('status') != 'complete'
            or not source_identity or manifest.get('source_identity') != source_identity
            or manifest.get('text_config') != config):
        raise ValueError('Packed artifact does not match source identity/configuration')
    if (manifest.get('bits') != bits or manifest.get('group_size') != group_size
            or manifest.get('recipe') != recipe(bits, group_size)):
        raise ValueError('Packed artifact precision differs from request')


def load_streamed_pipeline(source, *, bits=8, group_size=64, offload_layers=16,
                           device='cuda:0', stage_callback=None, export_packed=None, source_identity=None):
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3_5TextModel, Qwen3_5TextConfig, AutoTokenizer
    from .source import pipeline_class as _pipeline_class
    source = Path(source)
    if export_packed and not source_identity:
        raise ValueError('Export requires a portable source identity')
    if export_packed and Path(export_packed).exists():
        raise FileExistsError(export_packed)
    folder = source/'text_encoder'
    config = json.loads((folder/'config.json').read_text())['text_config']
    if config['hidden_size'] != 4096 or not 0 <= offload_layers <= config['num_hidden_layers']:
        raise ValueError('Unexpected source/configuration')
    transition = stage_callback or (lambda name: None)
    transition('construct_meta_text_model')
    with init_empty_weights():
        text = Qwen3_5TextModel._from_config(Qwen3_5TextConfig(**config), torch_dtype=torch.bfloat16)
    text.eval().requires_grad_(False)
    index = json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    expected = list(text.state_dict())
    if set('language_model.'+name for name in expected) - set(index):
        raise ValueError('Source is missing original text state')
    placement = {name: device for name, _ in text.named_children() if name != 'layers'}
    placement['embed_tokens'] = 'cpu'
    placement.update({f'layers.{i}': 'cpu' if i < offload_layers else device for i in range(len(text.layers))})
    def target(name):
        prefix = '.'.join(name.split('.')[:2]) if name.startswith('layers.') else name.split('.')[0]
        return placement[prefix]
    linear_names = {name for name, module in text.named_modules() if type(module) is nn.Linear}
    handled, modules = set(), []
    original_bytes = 0
    transition('stream_quantize_and_place')
    for name in expected:
        if name in handled:
            continue
        parent_name, _, leaf = name.rpartition('.')
        key = 'language_model.'+name
        with safe_open(str(folder/index[key]), framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(key)
            original_bytes += weight.numel()*weight.element_size()
            destination = target(name)
            if parent_name in linear_names and leaf == 'weight':
                original = text.get_submodule(parent_name)
                bias = None
                if original.bias is not None:
                    bias_name = parent_name+'.bias'; bias_key = 'language_model.'+bias_name
                    with safe_open(str(folder/index[bias_key]), framework='pt', device='cpu') as bhandle:
                        bias = bhandle.get_tensor(bias_key).to(device=destination, dtype=torch.bfloat16, copy=True)
                    handled.add(bias_name); original_bytes += bias.numel()*bias.element_size()
                replacement = PackedLinear.from_weight(weight, bits, group_size, destination, bias)
                owner, _, child = parent_name.rpartition('.')
                setattr(text.get_submodule(owner) if owner else text, child, replacement)
                modules.append({'name': parent_name, 'bits': bits, 'group_size': replacement.group_size,
                                'device': destination, 'shape': list(weight.shape)})
                del replacement, original, bias
            else:
                set_module_tensor_to_device(text, name, destination,
                    value=weight.to(device=destination, dtype=torch.bfloat16, copy=True), dtype=torch.bfloat16)
            del weight
        handled.add(name)
    if set(expected) != handled or any(p.is_meta for p in text.parameters()):
        raise ValueError('Incomplete streamed load')
    state_bytes = sum(t.numel()*t.element_size() for t in text.state_dict().values())
    if export_packed:
        from safetensors.torch import save_file
        export = Path(export_packed)
        export.mkdir(parents=True, exist_ok=False)
        manifest = {'format_version': 1, 'status': 'writing', 'source_identity': source_identity, 'text_config': config,
                    'bits': bits, 'group_size': group_size, 'modules': modules, 'files': {},
                    'model_state_bytes_before': original_bytes, 'model_state_bytes_after': state_bytes,
                    'recipe': recipe(bits, group_size)}
        (export/'manifest.json').write_text(json.dumps(manifest, indent=2))
        for prefix in placement:
            state = text.get_submodule(prefix).state_dict()
            if not state:
                continue
            file = prefix.replace('.', '-')+'.safetensors'
            local = {k: v.detach().cpu().contiguous() for k, v in state.items()}
            save_file(local, str(export/file))
            for key in local:
                manifest['files'][prefix+'.'+key] = file
            del state, local
        manifest['status'] = 'complete'
        (export/'manifest.json').write_text(json.dumps(manifest, indent=2))
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass
    transition('install_hybrid_offload')
    text = dispatch_model(text, device_map=placement, main_device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(source/'tokenizer'), local_files_only=True)
    pipe = _pipeline_class(source)(transformer=None, scheduler=None, text_encoder=text, tokenizer=tokenizer, vae=None)
    conversion = {'recipe': recipe(bits, group_size), 'bits': bits, 'group_size': group_size,
        'linear_modules': len(modules), 'modules': modules, 'placement': placement,
        'model_state_bytes_before': original_bytes, 'model_state_bytes_after': state_bytes,
        'retained_bf16': 'embedding, norm, conv and all non-Linear state',
        'loading': 'meta text model; one source tensor at a time; CPU quantization in 128-row chunks; direct placement',
        'source_path': str(source), 'learned_projection': False, 'fused_low_bit_encoder': False,
        'linear_execution': 'packed integer storage; FP32 dequantization; temporary BF16 F.linear',
        'activation_quantization': 'none', 'calibration': None, 'offload_layers': offload_layers}
    return pipe, conversion


def load_prepacked_pipeline(source, *, packed_dir, bits, group_size, offload_layers=16, device='cuda:0',
                            stage_callback=None, source_identity=None):
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3_5TextModel, Qwen3_5TextConfig, AutoTokenizer
    from .source import pipeline_class as _pipeline_class
    source, packed = Path(source), Path(packed_dir)
    manifest = json.loads((packed/'manifest.json').read_text())
    config = json.loads((source/'text_encoder/config.json').read_text())['text_config']
    validate_manifest(manifest, config, source_identity, bits, group_size)
    if not 0 <= offload_layers <= config['num_hidden_layers']:
        raise ValueError('Invalid offload layer count')
    transition = stage_callback or (lambda name: None)
    transition('construct_meta_and_load_prepacked')
    with init_empty_weights():
        text = Qwen3_5TextModel._from_config(Qwen3_5TextConfig(**config), torch_dtype=torch.bfloat16)
    text.eval().requires_grad_(False)
    placement = {name: device for name, _ in text.named_children() if name != 'layers'}
    placement['embed_tokens'] = 'cpu'
    placement.update({f'layers.{i}': 'cpu' if i < offload_layers else device for i in range(len(text.layers))})
    def prefix_for(name):
        return '.'.join(name.split('.')[:2]) if name.startswith('layers.') else name.split('.')[0]
    def get(name):
        prefix = prefix_for(name)
        path = (packed/manifest['files'][name]).resolve()
        path.relative_to(packed.resolve())
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            value = handle.get_tensor(name[len(prefix)+1:])
            return value.to(device=placement[prefix], copy=True)
    names = [row['name'] for row in manifest['modules']]
    if len(names) != len(set(names)) or set(names) != {name for name, m in text.named_modules() if type(m) is nn.Linear}:
        raise ValueError('Packed Linear inventory mismatch')
    for row in manifest['modules']:
        name = row['name']
        original = text.get_submodule(name)
        if type(original) is not nn.Linear or list(original.weight.shape) != row['shape']:
            raise ValueError('Packed Linear layout mismatch')
        if row['bits'] != bits or row['group_size'] != (row['shape'][1] if bits == 8 else group_size):
            raise ValueError('Packed module precision differs from request')
        bias = get(name+'.bias') if original.bias is not None else None
        replacement = PackedLinear(get(name+'.qweight'), get(name+'.scale'), row['bits'],
                                    row['group_size'], row['shape'][1], bias)
        owner, _, child = name.rpartition('.')
        setattr(text.get_submodule(owner) if owner else text, child, replacement)
    expected = set(text.state_dict())
    if expected != set(manifest['files']):
        raise ValueError('Packed artifact state inventory mismatch')
    for name, tensor in list(text.state_dict().items()):
        if tensor.is_meta:
            value = get(name)
            if value.shape != tensor.shape or value.dtype != torch.bfloat16:
                raise ValueError('Packed state shape/dtype mismatch: '+name)
            set_module_tensor_to_device(text, name, placement[prefix_for(name)], value=value, dtype=torch.bfloat16)
            del value
    if any(p.is_meta for p in text.parameters()):
        raise ValueError('Packed state left meta parameters')
    if sum(t.numel()*t.element_size() for t in text.state_dict().values()) != manifest['model_state_bytes_after']:
        raise ValueError('Packed state size mismatch')
    del tensor
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass
    text = dispatch_model(text, device_map=placement, main_device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(source/'tokenizer'), local_files_only=True)
    pipe = _pipeline_class(source)(transformer=None, scheduler=None, text_encoder=text, tokenizer=tokenizer, vae=None)
    conversion = {k: manifest[k] for k in ('bits','group_size','recipe','modules','model_state_bytes_before','model_state_bytes_after')}
    conversion['modules'] = [{**row, 'export_device': row['device'],
                              'device': placement[prefix_for(row['name'])]} for row in manifest['modules']]
    conversion.update(linear_modules=len(manifest['modules']), offload_layers=offload_layers, placement=placement,
        loading='prepacked per-layer safetensors; no runtime CPU quantization', packed_dir=str(packed),
        source_path=str(source), learned_projection=False, fused_low_bit_encoder=False)
    return pipe, conversion
