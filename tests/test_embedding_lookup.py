import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from moss_mini.embedding_lookup import install_cpu_embedding_lookup


class TinyText(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16, **embedding_kwargs):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(1024, 32, dtype=dtype, **embedding_kwargs)

    def get_input_embeddings(self):
        return self.embed_tokens


def dispatched_cpu_model(dtype=torch.bfloat16, **embedding_kwargs):
    model = TinyText(dtype=dtype, **embedding_kwargs).eval().requires_grad_(False)
    # Real Accelerate offload lifecycle; no CUDA initialization is required.
    add_hook_to_module(model.embed_tokens, AlignDevicesHook(execution_device='cpu', offload=True))
    assert model.embed_tokens.weight.device.type == 'meta'
    return model


def test_cpu_lookup_restores_exact_backing_weight_and_lookup_outputs():
    model = dispatched_cpu_model(padding_idx=0)
    embedding = model.get_input_embeddings()
    backing = embedding._hf_hook.weights_map['weight']
    original = backing.clone()
    record = install_cpu_embedding_lookup(model)
    body = embedding._old_forward
    observed = []
    def observe_body(input):
        output = body(input)
        observed.append((str(embedding.weight.device), str(input.device), str(output.device)))
        return output
    embedding._old_forward = observe_body
    for ids in (torch.tensor([[0, 1023, 31, 31]]), torch.arange(256).reshape(1, 256)):
        expected = torch.nn.functional.embedding(ids, original, padding_idx=0)
        assert torch.equal(embedding(ids), expected)
    assert observed == [('cpu', 'cpu', 'cpu')] * 2
    assert list(model.state_dict()) == ['embed_tokens.weight']
    assert embedding.weight.data_ptr() == backing.data_ptr()
    assert torch.equal(embedding.weight, original)
    assert embedding._hf_hook.offload is False
    assert embedding._hf_hook.io_same_device is True
    assert record['weight_bytes'] == 1024*32*2
    assert record['shape'] == [1024, 32]
    assert record['state_key_changed'] is False


@pytest.mark.parametrize('invalid', ['training', 'trainable', 'float32', 'max_norm', 'wrong_origin'])
def test_unsupported_embeddings_are_rejected_without_detaching_hook(invalid):
    model = dispatched_cpu_model(dtype=torch.float32 if invalid == 'float32' else torch.bfloat16,
                                 **({'max_norm': 1.0} if invalid == 'max_norm' else {}))
    embedding = model.get_input_embeddings()
    if invalid == 'training':
        embedding.train()
    elif invalid == 'trainable':
        embedding.weight.requires_grad_(True)
    elif invalid == 'wrong_origin':
        embedding._hf_hook.original_devices['weight'] = torch.device('meta')
    hook, weight = embedding._hf_hook, embedding.weight
    with pytest.raises(ValueError):
        install_cpu_embedding_lookup(model)
    assert embedding._hf_hook is hook
    assert embedding.weight is weight
    assert embedding.weight.device.type == 'meta'


def test_requires_offload_hook_and_does_not_reinstall():
    fresh = TinyText().eval().requires_grad_(False)
    with pytest.raises(ValueError, match='dispatch_model'):
        install_cpu_embedding_lookup(fresh)
    model = dispatched_cpu_model()
    install_cpu_embedding_lookup(model)
    hook, weight = model.embed_tokens._hf_hook, model.embed_tokens.weight
    with pytest.raises(ValueError, match='dispatch_model'):
        install_cpu_embedding_lookup(model)
    assert model.embed_tokens._hf_hook is hook
    assert model.embed_tokens.weight is weight
