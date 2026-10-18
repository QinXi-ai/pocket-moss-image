"""Optional real Cache-DiT integration tests; no model weights or GPU needed."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('cache_dit')

from moss_mini.cache_dit_adapter import CacheOptions, MossCacheHandle, moss_cache


class ToyBlock(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index, self.calls = index, 0

    def forward(self, hidden_states, rope, modulation):
        self.calls += 1
        # Text tokens affect image tokens, so CFG cross-contamination is visible.
        return (hidden_states + torch.sin(hidden_states*.01+self.index)*.02
                + hidden_states.mean(dim=1, keepdim=True)*.003 + modulation*.001)


class ToyMoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyBlock(i) for i in range(42)])

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skips = set(skip_layers or ())
        if any(i >= len(self.layers) for i in skips):
            raise ValueError('Invalid original skip index')
        context = encoder_hidden_states
        if encoder_attention_mask is not None:
            context = context[:, encoder_attention_mask[0].bool()]
        x = torch.cat((hidden_states, context), dim=1)
        for i, block in enumerate(self.layers):
            if i not in skips:
                x = block(x, None, timestep)
        if timestep == -999:
            raise RuntimeError('injected failure')
        sample = x[:, :hidden_states.shape[1]]
        return SimpleNamespace(sample=sample) if return_dict else (sample,)


@pytest.fixture
def data():
    torch.manual_seed(123)
    model = ToyMoss().eval()
    with torch.no_grad():
        yield model, torch.randn(1, 5, 4), torch.randn(1, 7, 4), torch.randn(1, 7, 4)
    handle = getattr(model, '_moss_cache_handle', None)
    if handle is not None:
        handle.close()


@pytest.mark.parametrize('bad', [
    {'threshold': float('nan')}, {'threshold': float('inf')}, {'threshold': -.1},
    {'threshold': 1.1}, {'threshold': True}, {'threshold': '0.03'},
    {'first_blocks': 0}, {'first_blocks': 1.5}, {'first_blocks': True},
    {'last_blocks': -1}, {'warmup_steps': -1}, {'warmup_steps': 51},
    {'max_continuous_cached_steps': -1}, {'num_inference_steps': 0},
    {'max_contexts': 0}, {'max_contexts': 5},
])
def test_invalid_options(bad):
    with pytest.raises(ValueError):
        CacheOptions(**bad)


def test_exact_zero_threshold_cfg_and_skips(data):
    model, x, positive, negative = data
    original, layers = model.forward, model.layers
    forwards = [b.forward for b in layers]
    with moss_cache(model, CacheOptions(threshold=0, warmup_steps=1)) as handle:
        for step in range(8):
            for condition, skips in [(positive, ()), (negative, (22,)), (negative, ())]:
                expected = original(x+step*.001, step, condition, skip_layers=skips).sample
                actual = model(x+step*.001, step, condition, skip_layers=skips).sample
                assert torch.equal(expected, actual)
                assert model.layers is layers
        assert len(handle.scopes) == 3
        assert all(s['hits'] == 0 for s in handle.stats())
    assert model.forward == original and model.layers is layers
    assert [b.forward for b in layers] == forwards
    assert handle.scopes == {} and handle.closed
    assert handle.close() == handle.records


def test_real_hits_and_same_shape_cfg_isolation(data):
    model, x, positive, negative = data
    original = model.forward
    branches = [(positive, ()), (negative, (22,)), (negative, ())]
    expected = [original(x, 1, c, skip_layers=s).sample for c, s in branches]
    assert not torch.equal(expected[0], expected[2])
    before = model.layers[20].calls
    with moss_cache(model, CacheOptions(threshold=.99, warmup_steps=1)) as handle:
        for _ in range(8):
            for index, (condition, skips) in enumerate(branches):
                actual = model(x, 1, condition, skip_layers=skips).sample
                assert torch.allclose(expected[index], actual, atol=1e-6, rtol=1e-6)
        assert model.layers[20].calls-before < 24
        stats = handle.stats()
        assert [s['calls'] for s in stats] == [8, 8, 8]
        assert all(s['hits'] > 0 and s['hits']+s['misses'] == 8 for s in stats)


def test_exception_restores_registered_tree(data):
    model, x, positive, _ = data
    layers, original = model.layers, model.forward
    with pytest.raises(RuntimeError, match='injected'):
        with moss_cache(model):
            model(x, -999, positive)
    assert model.layers is layers and model.forward == original
    assert not hasattr(model, '_moss_cache_handle')


def test_context_capacity_and_nested_handle(data):
    model, x, positive, negative = data
    with moss_cache(model, CacheOptions(max_contexts=1)):
        model(x, 1, positive)
        with pytest.raises(ValueError, match='context limit'):
            model(x, 1, negative)
        with pytest.raises(ValueError, match='already has'):
            MossCacheHandle(model)


def test_changed_condition_version_has_new_scope(data):
    model, x, positive, _ = data
    original = model.forward
    with moss_cache(model, CacheOptions(threshold=.99, warmup_steps=1)) as handle:
        for _ in range(3):
            model(x, 1, positive)
        positive.add_(2)
        expected = original(x, 1, positive).sample
        actual = model(x, 1, positive).sample
        assert torch.equal(expected, actual)
        assert len(handle.scopes) == 2


@pytest.mark.parametrize('unversioned', ['condition', 'mask'])
def test_unversioned_inputs_bypass_cache(data, unversioned):
    model, x, positive, _ = data
    original = model.forward
    mask = torch.ones(1, 7, dtype=torch.bool)
    with torch.inference_mode():
        if unversioned == 'condition':
            positive = positive.clone()
        else:
            mask = mask.clone()
        with moss_cache(model, CacheOptions(threshold=.99)) as handle:
            for _ in range(3):
                positive.add_(1)
                expected = original(x, 1, positive, mask).sample
                actual = model(x, 1, positive, mask).sample
                assert torch.equal(expected, actual)
            assert handle.bypassed_calls == 3 and not handle.scopes


def test_eval_no_grad_and_batch_restrictions(data):
    model, x, positive, _ = data
    model.train()
    with pytest.raises(ValueError, match='eval'):
        MossCacheHandle(model)
    model.eval()
    with moss_cache(model):
        with torch.enable_grad(), pytest.raises(RuntimeError, match='no_grad'):
            model(x, 1, positive)
        with pytest.raises(ValueError, match='batch=1'):
            model(x.expand(2, -1, -1), 1, positive)
        model.train()
        with pytest.raises(ValueError, match='eval'):
            model(x, 1, positive)
        model.eval()


def test_skip_validation_and_request_length(data):
    model, x, positive, _ = data
    with moss_cache(model, CacheOptions(num_inference_steps=2, warmup_steps=1)):
        for invalid in [(42,), (-1,), (True,), (1.5,)]:
            with pytest.raises(ValueError, match='Invalid skip'):
                model(x, 1, positive, skip_layers=invalid)
        with pytest.raises(ValueError, match='middle block'):
            model(x, 1, positive, skip_layers=range(32))
        model(x, 1, positive)
        model(x, 1, positive)
        with pytest.raises(ValueError, match='step limit'):
            model(x, 1, positive)


def test_accelerate_outer_hook_is_preserved(data):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module
    model, x, positive, _ = data
    original = model.forward
    expected = original(x, 1, positive, skip_layers=(22,)).sample
    handle = MossCacheHandle(model, CacheOptions(threshold=0))
    add_hook_to_module(model, AlignDevicesHook(execution_device='cpu'))
    assert torch.equal(expected, model(x, 1, positive, skip_layers=(22,)).sample)
    handle.close()
    assert model._old_forward == original
    assert torch.equal(expected, model(x, 1, positive, skip_layers=(22,)).sample)
    remove_hook_from_module(model)


def test_strict_backend_version(data, monkeypatch):
    from moss_mini import cache_dit_adapter
    model, *_ = data
    monkeypatch.setattr(cache_dit_adapter.importlib.metadata, 'version', lambda _: '1.5.4')
    with pytest.raises(RuntimeError, match='cache-dit==1.5.3'):
        MossCacheHandle(model)
    assert not hasattr(model, '_moss_cache_handle')


def test_cleanup_even_if_stats_fail(data, monkeypatch):
    model, x, positive, _ = data
    original = model.forward
    handle = MossCacheHandle(model)
    model(x, 1, positive)
    managers = [scope['stack']._context_manager for scope in handle.scopes.values()]
    def fail():
        raise RuntimeError('stats failure')
    monkeypatch.setattr(handle, 'stats', fail)
    with pytest.raises(RuntimeError, match='stats failure'):
        handle.close()
    assert handle.closed and not handle.scopes
    assert model.forward == original and not hasattr(model, '_moss_cache_handle')
    assert all(not manager._cached_context_manager for manager in managers)
