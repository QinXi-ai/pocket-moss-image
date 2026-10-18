"""Original encoder, synthetic 256-token stress; no image-quality claims."""
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors.torch import save_file
from moss_mini.encoder import load_prepacked_pipeline, set_dequant_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source-version', required=True)
    p.add_argument('--packed', required=True)
    p.add_argument('--backend', choices=('torch', 'triton'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    pipe, conversion = load_prepacked_pipeline(a.model, packed_dir=a.packed, bits=6,
        group_size=64, offload_layers=20, source_identity={'kind': 'local', 'version': a.source_version})
    set_dequant_backend(pipe, a.backend)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    prompt = 'A detailed photograph of a red chair beside a wooden table in a sunlit room. '*100
    rows = []
    reference = None
    with torch.no_grad():
        for repetition in range(6):
            torch.cuda.synchronize()
            started = time.perf_counter()
            embeds, mask = pipe.encode_prompt(prompt, 256)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            assert embeds.shape == (1, 256, 4096) and mask.shape == (1, 256)
            assert mask.all() and torch.isfinite(embeds).all()
            cpu = embeds.cpu().contiguous()
            if reference is None:
                reference = cpu
            else:
                assert torch.equal(cpu, reference)
            rows.append({'repetition': repetition, 'seconds': elapsed})
            del cpu, embeds, mask
    save_file({'conditions': reference}, str(a.output/'conditions.safetensors'))
    (a.output/'run.json').write_text(json.dumps({'status': 'complete', 'backend': a.backend,
        'shape': [1,256,4096], 'load_seconds': load_seconds, 'forwards': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
