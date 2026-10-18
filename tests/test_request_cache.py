"""CPU contract tests; no model downloads, private sources, or CUDA required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from moss_mini.request_cache import exact_request_cache


class Rope(nn.Module):
    def __init__(self, base=10000.):
        super().__init__()
        self.base = base
        self.register_buffer('bands', torch.arange(1,5,dtype=torch.float32))

    def forward(self, height, width, length, device):
        positions = torch.arange(height*width+length,device=device,dtype=torch.float32)
        return (positions[:,None]*self.bands/self.base).sin()[None]


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6,4)

    def forward(self, context, rope):
        return self.proj(context) + rope


class ToyMoss(nn.Module):
    """Original source patterns and order, tiny arithmetic elsewhere."""
    def __init__(self):
        super().__init__()
        self.text_embedder = Projection()
        self.video_embedder = nn.Linear(3,4)
        self.rope = Rope()
        self.eval()

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        batch, _, height, width = hidden_states.shape
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = hidden_states[index:index+1].flatten(2).transpose(1,2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype))
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            outputs.append(text.sum() + video.sum() * timestep)
        return torch.stack(outputs)


@pytest.fixture
def sample():
    torch.manual_seed(42)
    return ToyMoss(), torch.randn(1,3,2,2), torch.randn(1,4,6), torch.randn(1,4,6), torch.tensor([[1,1,1,0]],dtype=torch.bool)


@torch.no_grad()
def test_exact_cfg_hits_and_request_cleanup(sample):
    model,x,positive,negative,mask = sample
    original = model.forward
    baseline = [model(x,t,e,mask) for t in (1.,.5,.1) for e in (positive,negative)]
    with exact_request_cache(model) as cache:
        actual = [model(x,t,e,mask) for t in (1.,.5,.1) for e in (positive,negative)]
        assert all(torch.equal(a,b) for a,b in zip(actual,baseline))
        stats = cache.as_dict()
        assert (stats['text_hits'],stats['text_misses']) == (4,2)
        assert (stats['context_hits'],stats['context_misses']) == (4,2)
        assert (stats['mask_hits'],stats['mask_misses']) == (5,1)
        assert (stats['rope_hits'],stats['rope_misses']) == (5,1)
    assert model.forward == original
    assert cache.as_dict()['current_entries'] == cache.as_dict()['current_bytes'] == 0
    with exact_request_cache(model) as fresh:
        model(x,.5,positive,mask)
    assert fresh.as_dict()['text_misses'] == 1


@torch.no_grad()
def test_input_mask_parameter_and_rope_invalidation(sample):
    model,x,positive,_,mask = sample
    with exact_request_cache(model) as cache:
        model(x,.5,positive,mask)
        positive.add_(1)
        model(x,.5,positive,mask)
        mask[0,0],mask[0,3] = False,True
        model(x,.5,positive,mask)
        model.text_embedder.proj.weight.add_(.2)
        model(x,.5,positive,mask)
        model.text_embedder.proj.weight = nn.Parameter(model.text_embedder.proj.weight.clone())
        model(x,.5,positive,mask)
        model.rope.bands.add_(1)
        model(x,.5,positive,mask)
        model.rope = Rope(base=100.)
        actual = model(x,.5,positive,mask)
        assert cache.as_dict()['text_misses'] == 7
    assert torch.equal(actual,model(x,.5,positive,mask))


@torch.no_grad()
def test_mask_validation_is_repeated_after_inplace_change(sample):
    model,x,positive,_,mask = sample
    integer_mask = mask.int()
    with exact_request_cache(model):
        model(x,.5,positive,integer_mask)
        integer_mask[0,0] = 2
        with pytest.raises(ValueError,match='binary'):
            model(x,.5,positive,integer_mask)
        with pytest.raises(ValueError,match='binary'):
            model(x,.5,positive,torch.ones(1,9))
        with pytest.raises(ValueError,match='at least one'):
            model(x,.5,positive,torch.zeros_like(mask))
        assert torch.isfinite(model(x,.5,positive,mask)).all()


@torch.no_grad()
def test_offload_clears_cache_and_capacity_is_bounded(sample):
    model,x,positive,negative,mask = sample
    with exact_request_cache(model,max_entries=1) as cache:
        model(x,.5,positive,mask)
        model(x,.5,negative,mask)
        assert cache.as_dict()['evictions'] > 0
        model.to('cpu')
        assert cache.as_dict()['current_bytes'] == 0
        model(x,.5,negative,mask)
        assert cache.as_dict()['text_misses'] == 3
    with exact_request_cache(model,max_bytes=1) as tiny:
        model(x,.5,positive,mask)
        model(x,.5,positive,mask)
        assert tiny.as_dict()['current_bytes'] <= 1
        assert tiny.as_dict()['text_hits'] == 0


@torch.no_grad()
def test_batch_rows_shape_dtype_and_mask_none(sample):
    model,x,positive,negative,mask = sample
    batch_x = x.repeat(2,1,1,1)
    embeds = torch.cat((positive,negative))
    masks = mask.repeat(2,1)
    baseline = model(batch_x,.5,embeds,masks)
    with exact_request_cache(model) as cache:
        assert torch.equal(model(batch_x,.5,embeds,masks),baseline)
        assert torch.equal(model(batch_x,.5,embeds,masks),baseline)
        assert cache.as_dict()['text_hits'] == 2
        model(torch.randn(1,3,3,2),.5,positive,None)
        assert cache.as_dict()['rope_misses'] == 2
        model.double()
        result = model(x.double(),.5,positive.double(),mask)
        assert result.dtype == torch.float64


@torch.no_grad()
def test_exception_and_nested_context_cleanup(sample):
    model,x,positive,_,mask = sample
    original = model.forward
    with pytest.raises(RuntimeError,match='deliberate'):
        with exact_request_cache(model) as cache:
            model(x,.5,positive,mask)
            with pytest.raises(ValueError,match='nest'):
                with exact_request_cache(model):
                    pass
            raise RuntimeError('deliberate')
    assert model.forward == original
    assert cache.as_dict()['current_bytes'] == 0
    assert '_exact_text_projection_cache_active' not in model.__dict__


def test_grad_and_unversioned_inputs_bypass(sample):
    model,x,positive,_,mask = sample
    with exact_request_cache(model) as grad:
        model(x,.5,positive,mask)
        model(x,.5,positive,mask)
    assert grad.as_dict()['text_hits'] == 0
    with torch.inference_mode():
        unversioned = positive.clone()
        with exact_request_cache(model) as inferred:
            model(x,.5,unversioned,mask)
            model(x,.5,unversioned,mask)
    assert inferred.as_dict()['text_hits'] == 0


@pytest.mark.parametrize('kwargs',[{'max_entries':0},{'max_entries':True},{'max_entries':1.5},{'max_bytes':-1},{'max_bytes':False}])
def test_invalid_capacity(sample,kwargs):
    with pytest.raises(ValueError,match='positive integers'):
        with exact_request_cache(sample[0],**kwargs):
            pass


def test_unsupported_source_and_existing_hooks_rejected(sample):
    model,*_ = sample
    original = model.forward
    model.forward = lambda *args,**kwargs: original(*args,**kwargs)
    with pytest.raises(ValueError,match='pinned original MOSS forward'):
        with exact_request_cache(model):
            pass
    assert '_exact_text_projection_cache_active' not in model.__dict__
    model.forward = original
    model._hf_hook = SimpleNamespace()
    with pytest.raises(ValueError,match='before Accelerate'):
        with exact_request_cache(model):
            pass


class DifferentSource(ToyMoss):
    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None):
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        return encoder_hidden_states


def test_missing_ast_pattern_is_explicit_and_leaves_model_untouched():
    model = DifferentSource()
    original = model.forward
    with pytest.raises(ValueError,match='context/RoPE source patterns differ'):
        with exact_request_cache(model):
            pass
    assert model.forward == original
    assert '_exact_text_projection_cache_active' not in model.__dict__


@torch.no_grad()
def test_restores_payload_if_offload_wrapper_installed_inside_context(sample):
    model,x,positive,_,mask = sample
    original = model.forward
    with exact_request_cache(model) as cache:
        # Mirrors the outer-forward/payload convention used by Accelerate.
        model._old_forward = model.forward
        model._hf_hook = SimpleNamespace()
        def outer(*args,**kwargs):
            return model._old_forward(*args,**kwargs)
        model.forward = outer
        model(x,.5,positive,mask)
        model(x,.5,positive,mask)
        assert cache.as_dict()['text_hits'] == 1
    assert model.forward is outer
    assert model._old_forward == original
    assert torch.isfinite(model(x,.5,positive,mask)).all()
    assert cache.as_dict()['current_bytes'] == 0
