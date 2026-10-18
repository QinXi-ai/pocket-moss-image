"""Read each original DiT tensor once, without retaining a full BF16 CPU DiT."""
import json
from pathlib import Path


def load_direct_stage(source, phase, offload='none', device='cpu'):
    if phase != 'generate' or offload != 'none' or device != 'cpu':
        raise ValueError('Direct loader is only the initial generation CPU/meta stage')
    import torch
    from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
    from safetensors import safe_open
    source = Path(source)
    cls = get_class_from_dynamic_module(str(source), module_file='pipeline_moss_image.py',
        class_name='MOSSImageTransformer', local_files_only=True, trust_remote_code=True)
    with torch.device('meta'):
        model = cls.from_config(str(source/'transformer/config.json'))
    model.to(dtype=torch.bfloat16)
    index = json.loads((source/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    expected = model.state_dict()
    if set(index) != set(expected):
        raise ValueError('Original shard index does not exactly cover meta model')
    for shard in set(index.values()):
        path = (source/'transformer'/shard).resolve()
        if not path.is_relative_to((source/'transformer').resolve()) or not path.is_file():
            raise ValueError('Invalid source shard')
    # Small non-joint state remains on CPU. Joint weights are still meta.
    for name in sorted({name.split('.')[0] for name in expected if not name.startswith('layers.')}):
        module = model.get_submodule(name)
        state = {}
        for key, shape in module.state_dict().items():
            full = name+'.'+key
            with safe_open(str(source/'transformer'/index[full]), framework='pt', device='cpu') as stream:
                tensor = stream.get_tensor(full)
                if tensor.shape != shape.shape:
                    raise ValueError('Nonblock source shape mismatch')
                state[key] = tensor.to(dtype=shape.dtype, copy=True)
        module.load_state_dict(state, strict=True, assign=True)
        del state
    model.__dict__['_direct_source'] = str(source/'transformer')
    model.__dict__['_direct_index'] = index
    pipeline_class = get_class_from_dynamic_module(str(source), module_file='pipeline_moss_image.py',
        class_name='MOSSImagePipeline', local_files_only=True, trust_remote_code=True)
    from diffusers import AutoencoderKLFlux2
    scheduler_class = get_class_from_dynamic_module(str(source), module_file='pipeline_moss_image.py',
        class_name='MOSSImageScheduler', local_files_only=True, trust_remote_code=True)
    scheduler = scheduler_class.from_pretrained(str(source/'scheduler'), local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(str(source/'vae'), torch_dtype=torch.float32,
                                           local_files_only=True).eval()
    return pipeline_class(transformer=model.eval(), scheduler=scheduler,
                          text_encoder=None, tokenizer=None, vae=vae)


def materializer_for(model):
    import torch
    from safetensors import safe_open
    source = Path(model.__dict__['_direct_source']); index = model.__dict__['_direct_index']
    used = set()

    def load_block(i, block, device):
        if i in used or any(v.device.type != 'meta' for v in block.state_dict().values()):
            raise ValueError('Block must be fresh meta before source materialization')
        state = {}
        for name, target in block.state_dict().items():
            full = f'layers.{i}.{name}'
            with safe_open(str(source/index[full]), framework='pt', device='cpu') as stream:
                tensor = stream.get_tensor(full)
                if tensor.shape != target.shape:
                    raise ValueError('Block source shape mismatch')
                state[name] = tensor.to(device=device, dtype=target.dtype, copy=True)
        block.load_state_dict(state, strict=True, assign=True)
        used.add(i)
    return load_block
