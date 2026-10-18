"""Optional, approximate Cache-DiT routing for the MOSS joint transformer.

Use one handle per generation request. Install after loading/compiling individual
blocks and before Accelerate model CPU offload; close after inference. Batch size
one, eval mode and no_grad/inference_mode are required. Model weights, module
structure and tensor aliases must not be mutated during a request. In particular,
mutations through .data or external storage evade PyTorch version counters.

Cache-DiT 1.5.3 is pinned because statistics use its private live context API.
Root/group compilation, group offload, concurrent requests and training are not
supported. A nonzero threshold changes model predictions; it is not lossless.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import math
from numbers import Real
import threading
from unittest.mock import patch

import torch


CACHE_DIT_VERSION = '1.5.3'


@dataclass(frozen=True)
class CacheOptions:
    threshold: float = 0.03
    first_blocks: int = 8
    last_blocks: int = 2
    warmup_steps: int = 8
    max_continuous_cached_steps: int = 1
    num_inference_steps: int = 50
    max_contexts: int = 4

    def __post_init__(self):
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError('threshold must be finite and between 0 and 1')
        for name, minimum in (('first_blocks', 1), ('last_blocks', 0),
                              ('warmup_steps', 0), ('max_continuous_cached_steps', 0),
                              ('num_inference_steps', 1), ('max_contexts', 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.warmup_steps > self.num_inference_steps:
            raise ValueError('warmup_steps cannot exceed num_inference_steps')
        if self.max_contexts > 4:
            raise ValueError('At most four independent request cache contexts are supported')


class _Stack(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = torch.nn.ModuleList(blocks)

    def forward(self, hidden_states, *args, **kwargs):
        for block in self.layers:
            hidden_states = block(hidden_states, *args, **kwargs)
        return hidden_states


def _version(tensor):
    if tensor is None:
        return None
    try:
        return tensor._version
    except RuntimeError:
        return None  # inference tensors are handled through uncached execution


class MossCacheHandle:
    """Keep positive/negative/skip-layer residual caches independent.

    Cache-DiT wraps private stacks of references to the original blocks. Shared
    block forward methods are never replaced. The original MOSS forward retains
    its embedding, attention and output path. Condition identity/version and skip
    set select independent contexts; no alternating-call CFG assumption is used.
    Unversioned conditions/masks run the original forward without caching.
    """
    def __init__(self, model, options=CacheOptions()):
        if not isinstance(options, CacheOptions):
            raise TypeError('options must be CacheOptions')
        version = importlib.metadata.version('cache-dit')
        if version != CACHE_DIT_VERSION:
            raise RuntimeError(f'MOSS cache requires cache-dit=={CACHE_DIT_VERSION}; found {version}')
        import cache_dit
        if hasattr(model, '_moss_cache_handle'):
            raise ValueError('Model already has an active MOSS cache handle')
        if hasattr(model, '_hf_hook'):
            raise ValueError('Install MOSS cache before model-offload hooks')
        if not isinstance(getattr(model, 'layers', None), torch.nn.ModuleList):
            raise ValueError('Expected original MOSS layers ModuleList')
        if any(module.training for module in model.modules()):
            raise ValueError('MOSS cache requires model.eval()')
        if options.first_blocks + options.last_blocks >= len(model.layers):
            raise ValueError('Cache must leave at least one middle block')
        self.model, self.options, self.backend = model, options, cache_dit
        self.layers, self.original = model.layers, model.forward
        self.scopes, self.records = {}, []
        self.closed, self.bypassed_calls = False, 0
        self._lock = threading.Lock()
        self.block_forwards = [block.forward for block in self.layers]
        model.__dict__['_moss_cache_handle'] = self
        model.forward = self.forward

    def _scope(self, hidden, condition, mask, skips):
        # Strong references prevent Python identity reuse during this request.
        key = (id(condition), _version(condition), id(mask), _version(mask), skips,
               tuple(hidden.shape), hidden.dtype, hidden.device)
        if key not in self.scopes:
            if len(self.scopes) >= self.options.max_contexts:
                raise ValueError('Request cache context limit exceeded; close between requests')
            active = [b for i, b in enumerate(self.layers) if i not in skips]
            if self.options.first_blocks + self.options.last_blocks >= len(active):
                raise ValueError('Cache must leave at least one middle block after skips')
            stack = _Stack(active).eval()
            adapter = self.backend.BlockAdapter(
                transformer=stack, blocks=stack.layers, blocks_name='layers',
                forward_pattern=self.backend.ForwardPattern.Pattern_3,
                has_separate_cfg=False, check_forward_pattern=False)
            config = self.backend.DBCacheConfig(
                Fn_compute_blocks=self.options.first_blocks,
                Bn_compute_blocks=self.options.last_blocks,
                residual_diff_threshold=float(self.options.threshold),
                max_warmup_steps=self.options.warmup_steps,
                max_continuous_cached_steps=self.options.max_continuous_cached_steps,
                num_inference_steps=self.options.num_inference_steps,
                enable_separate_cfg=False)
            self.backend.enable_cache(adapter, cache_config=config)
            self.scopes[key] = {'stack': stack, 'condition': condition, 'mask': mask,
                                'skips': skips, 'calls': 0}
        scope = self.scopes[key]
        if scope['calls'] >= self.options.num_inference_steps:
            raise ValueError('Request step limit exceeded; create a new handle for each generation')
        scope['calls'] += 1
        return scope['stack']

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent/reentrant MOSS cache invocation is unsupported')
        try:
            if self.closed:
                raise RuntimeError('MOSS cache handle is closed')
            if torch.is_grad_enabled():
                raise RuntimeError('MOSS cache requires torch.no_grad() or inference_mode()')
            if any(module.training for module in self.model.modules()):
                raise ValueError('MOSS cache requires model.eval() throughout the request')
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 1 or hidden_states.shape[0] != 1:
                raise ValueError('MOSS cache supports batch=1 only')
            if not isinstance(encoder_hidden_states, torch.Tensor):
                raise TypeError('encoder_hidden_states must be a tensor')
            if encoder_attention_mask is not None and not isinstance(encoder_attention_mask, torch.Tensor):
                raise TypeError('encoder_attention_mask must be a tensor or None')
            raw_skips = tuple(() if skip_layers is None else skip_layers)
            if any(type(i) is not int or not 0 <= i < len(self.layers) for i in raw_skips):
                raise ValueError('Invalid skip_layers')
            skips = tuple(sorted(set(raw_skips)))
            if (_version(encoder_hidden_states) is None or
                    (encoder_attention_mask is not None and _version(encoder_attention_mask) is None)):
                self.bypassed_calls += 1
                return self.original(hidden_states, timestep, encoder_hidden_states,
                                     encoder_attention_mask, return_dict=return_dict,
                                     skip_layers=skips)
            stack = self._scope(hidden_states, encoder_hidden_states, encoder_attention_mask, skips)
            # Original active blocks are already filtered. Passing skip=None
            # preserves index validation while the original sees one stack.
            with patch.object(self.model, 'layers', torch.nn.ModuleList([stack])):
                return self.original(hidden_states, timestep, encoder_hidden_states,
                                     encoder_attention_mask, return_dict=return_dict,
                                     skip_layers=None)
        finally:
            self._lock.release()

    def stats(self):
        if self.closed:
            return self.records
        records = []
        for index, scope in enumerate(self.scopes.values()):
            stack = scope['stack']
            context = stack._context_manager.get_context(stack._context_names[0])
            hits = int(context.accumulated_cached_steps)
            calls = int(context.accumulated_transformer_executed_steps)
            records.append({'scope_index': index, 'skip_layers': list(scope['skips']),
                            'condition_shape': list(scope['condition'].shape),
                            'hits': hits, 'calls': calls, 'misses': calls-hits,
                            'cached_steps': list(context.cached_steps),
                            'residual_diffs': dict(context.residual_diffs)})
        return records

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close MOSS cache during inference')
        try:
            if self.closed:
                return self.records
            # Accelerate installed after us wraps our forward as _old_forward.
            if hasattr(self.model, '_hf_hook') and hasattr(self.model, '_old_forward'):
                self.model._old_forward = self.original
            else:
                self.model.forward = self.original
            try:
                self.records = self.stats()
            finally:
                first_error = None
                for scope in self.scopes.values():
                    stack = scope['stack']
                    manager = getattr(stack, '_context_manager', None)
                    try:
                        self.backend.disable_cache(stack)
                    except Exception as exc:
                        first_error = first_error or exc
                    finally:
                        if manager is not None:
                            manager.clear_contexts()
                self.scopes.clear()
                self.model.__dict__.pop('_moss_cache_handle', None)
                self.closed = True
                if first_error:
                    raise first_error
            if self.model.layers is not self.layers:
                raise RuntimeError('MOSS registered tree was not restored')
            if any(b.forward != f for b, f in zip(self.layers, self.block_forwards)):
                raise RuntimeError('Shared MOSS block forwards were unexpectedly modified')
            return self.records
        finally:
            self._lock.release()


@contextmanager
def moss_cache(model, options=CacheOptions()):
    """Use exactly once around one pipeline generation; no persistent cache."""
    handle = MossCacheHandle(model, options)
    try:
        yield handle
    finally:
        handle.close()
