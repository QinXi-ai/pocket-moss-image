"""Explicit reuse of completed, source-bound prompt encodings."""
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from .encoder import recipe


def reuse_conditions(directory, output, plan):
    started = time.perf_counter()
    directory, output = Path(directory), Path(output)
    if any((output/name).exists() for name in ('conditions.safetensors', 'encode.json')):
        raise FileExistsError('Condition cache output already exists')
    source_plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    source_record = json.loads((directory/'encode.json').read_text(encoding='utf-8'))
    if source_record.get('status') != 'complete':
        raise ValueError('Condition cache encoding is incomplete')
    for key in ('source_identity', 'prompt', 'negative_prompt', 'bits', 'max_sequence_length'):
        if key not in source_plan or key not in plan or source_plan[key] != plan[key]:
            raise ValueError('Condition cache request mismatch: '+key)
    expected_recipe = recipe(plan['bits'], 64)
    conversion = source_record.get('conversion', {})
    if conversion.get('recipe') != expected_recipe:
        raise ValueError('Condition cache encoder recipe mismatch')
    tensors = load_file(str(directory/'conditions.safetensors'), device='cpu')
    expected_names = {prefix+name for prefix in ('', 'negative_') for name in ('prompt_embeds', 'prompt_attention_mask')}
    if set(tensors) != expected_names:
        raise ValueError('Condition cache must contain both embeddings and masks')
    for prefix in ('', 'negative_'):
        embeds, mask = tensors[prefix+'prompt_embeds'], tensors[prefix+'prompt_attention_mask']
        if (embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != 4096
                or not 1 <= embeds.shape[1] <= plan['max_sequence_length'] or embeds.dtype != torch.bfloat16
                or not torch.isfinite(embeds).all()):
            raise ValueError('Invalid condition cache embeddings')
        if (mask.shape != embeds.shape[:2] or mask.dtype not in (torch.bool, torch.int32, torch.int64)
                or not ((mask == 0) | (mask == 1)).all() or not mask.any()):
            raise ValueError('Invalid condition cache attention mask')
    save_file(tensors, str(output/'conditions.safetensors'))
    result = {'status': 'complete', 'seconds': time.perf_counter()-started,
              'conversion': {'recipe': expected_recipe, 'loading': 'validated saved conditions; encoder not loaded'},
              'reused_from': str(directory.resolve()), 'source_identity': plan['source_identity']}
    (output/'encode.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result
