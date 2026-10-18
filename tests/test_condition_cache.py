import json
import pytest
import torch
from safetensors.torch import load_file, save_file
from moss_mini.condition_cache import reuse_conditions
from moss_mini.encoder import recipe


def cache(tmp_path):
    src, dst = tmp_path/'src', tmp_path/'dst'
    src.mkdir(); dst.mkdir()
    plan = dict(source_identity={'kind':'local','version':'test-v1'}, prompt='cat', negative_prompt='',
                bits=6, max_sequence_length=256, seed=42)
    (src/'plan.json').write_text(json.dumps(plan))
    (src/'encode.json').write_text(json.dumps({'status':'complete','conversion':{'recipe':recipe(6,64)}}))
    tensors = {prefix+name: (torch.ones(1,2,4096,dtype=torch.bfloat16) if name=='prompt_embeds'
                            else torch.ones(1,2,dtype=torch.int64))
               for prefix in ('','negative_') for name in ('prompt_embeds','prompt_attention_mask')}
    save_file(tensors, str(src/'conditions.safetensors'))
    return src, dst, plan, tensors


def test_conditions_reuse_changes_seed_without_reencoding(tmp_path):
    src,dst,plan,tensors = cache(tmp_path)
    plan['seed'] = 123
    result = reuse_conditions(src,dst,plan)
    assert result['status']=='complete'
    restored = load_file(str(dst/'conditions.safetensors'))
    assert all(torch.equal(tensors[k],restored[k]) for k in tensors)
    # A reused output can itself serve as a cache once its plan is written by the CLI.
    (dst/'plan.json').write_text(json.dumps(plan))
    chained = tmp_path/'chained'; chained.mkdir()
    reuse_conditions(dst,chained,plan)


@pytest.mark.parametrize('key,value', [('prompt','dog'), ('negative_prompt','blur'), ('bits',4),
    ('max_sequence_length',128), ('source_identity',{'kind':'local','version':'different'})])
def test_reuse_rejects_changed_condition_identity(tmp_path, key, value):
    src,dst,plan,_ = cache(tmp_path)
    plan[key] = value
    with pytest.raises(ValueError, match='mismatch'):
        reuse_conditions(src,dst,plan)
    assert not (dst/'conditions.safetensors').exists()


def test_reuse_rejects_bad_mask(tmp_path):
    src,dst,plan,tensors = cache(tmp_path)
    tensors['negative_prompt_attention_mask'].fill_(2)
    save_file(tensors,str(src/'conditions.safetensors'))
    with pytest.raises(ValueError, match='mask'):
        reuse_conditions(src,dst,plan)


@pytest.mark.parametrize('kind', ['incomplete', 'recipe', 'missing', 'nonfinite', 'existing'])
def test_reuse_rejects_invalid_cache_without_overwrite(tmp_path, kind):
    src,dst,plan,tensors = cache(tmp_path)
    if kind in ('incomplete', 'recipe'):
        record = {'status':'failed' if kind=='incomplete' else 'complete',
                  'conversion':{'recipe':'other' if kind=='recipe' else recipe(6,64)}}
        (src/'encode.json').write_text(json.dumps(record))
    elif kind=='missing':
        del tensors['negative_prompt_embeds']
        save_file(tensors,str(src/'conditions.safetensors'))
    elif kind=='nonfinite':
        tensors['prompt_embeds'][0,0,0] = float('nan')
        save_file(tensors,str(src/'conditions.safetensors'))
    else:
        (dst/'conditions.safetensors').write_bytes(b'preserve')
    with pytest.raises((ValueError, FileExistsError)):
        reuse_conditions(src,dst,plan)
    if kind=='existing':
        assert (dst/'conditions.safetensors').read_bytes()==b'preserve'
    else:
        assert not (dst/'conditions.safetensors').exists()
