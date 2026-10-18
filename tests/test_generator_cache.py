import copy
import pytest
from moss_mini.generator_cache import validate_manifest, safe_shard_path, LAYOUT, BACKEND
from moss_mini.reference import RECIPE


def manifest():
    return dict(format_version=1, status='complete', source_identity={'kind':'local','version':'test-v1'},
                recipe=RECIPE, layout=LAYOUT, backend=BACKEND, transformer_config={'num_layers':42},
                shards={'layers.0':{'file':'layers-0.safetensors'}})


def test_manifest_identity_recipe_layout_and_completion():
    m = manifest()
    validate_manifest(m, m['source_identity'], m['transformer_config'])
    changes = {'format_version':2, 'status':'writing', 'source_identity':{'version':'other'},
               'recipe':'other', 'layout':{}, 'backend':{}, 'transformer_config':{}, 'shards':{}}
    for key, bad in changes.items():
        modified = copy.deepcopy(m)
        modified[key] = bad
        with pytest.raises(ValueError):
            validate_manifest(modified, m['source_identity'], m['transformer_config'])


def test_shard_cannot_escape_directory(tmp_path):
    assert safe_shard_path(tmp_path, 'layers-0.safetensors') == tmp_path/'layers-0.safetensors'
    for name in ('../escape.safetensors', str(tmp_path.parent/'outside.safetensors'), 'file.pt'):
        with pytest.raises(ValueError):
            safe_shard_path(tmp_path, name)
