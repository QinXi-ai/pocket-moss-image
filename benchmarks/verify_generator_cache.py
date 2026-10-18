"""Cold-load a packed DiT and compare every restored tensor to its safetensor."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from moss_mini.generator_cache import load_prepacked_generator, safe_shard_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source-version', required=True)
    p.add_argument('--packed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pipe, info = load_prepacked_generator(a.model, a.packed, {'kind':'local','version':a.source_version})
    manifest = json.loads((a.packed/'manifest.json').read_text())
    count = 0
    for prefix, row in manifest['shards'].items():
        state = pipe.transformer.get_submodule(prefix).state_dict()
        with safe_open(str(safe_shard_path(a.packed, row['file'])), framework='pt', device='cpu') as f:
            for key, value in state.items():
                assert torch.equal(value, f.get_tensor(key)), prefix+'.'+key
                assert not value.is_meta and not torch.is_inference(value)
                count += 1
    assert count == manifest['tensor_count']
    a.output.write_text(json.dumps({'status':'complete','tensor_count':count,'all_tensors_exact':True,
        'cuda_initialized':torch.cuda.is_initialized(),'state_bytes':info['state_bytes']},indent=2)+'\n')


if __name__ == '__main__':
    main()
