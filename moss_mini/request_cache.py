"""Opt-in exact request cache for pinned MOSS preprocessing and text projection.

Install BEFORE Cache-DiT and Accelerate. The original forward is AST-patched at
three exact source patterns; all diffusion mathematics and validations remain.
Never use concurrent requests or untracked .data/storage writes on this model.
"""
import ast
from collections import OrderedDict
from contextlib import contextmanager
import inspect
import textwrap
import threading
from types import MethodType

import torch

def _tensor_token(value):
    if value is None:
        return None
    # Version and layout metadata live on the host; this does not read CUDA data.
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.device, value.dtype)


class _Tracker:
    """Cache registration traversal, still check every tensor's host version."""
    def __init__(self, module):
        self.module = module
        self.rebuild()

    def rebuild(self):
        self.maps = []
        for module in self.module.modules():
            for mapping in (module._modules, module._parameters, module._buffers):
                self.maps.append((mapping, tuple(mapping), tuple(mapping.values())))
        self.tensors = tuple(self.module.parameters()) + tuple(self.module.buffers())

    def token(self):
        if any(tuple(mapping) != keys or len(mapping) != len(values) or
               any(a is not b for a,b in zip(mapping.values(), values))
               for mapping, keys, values in self.maps):
            self.rebuild()
        return tuple((id(t), t._version, tuple(t.shape), t.dtype) for t in self.tensors)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(t for item in value for t in _tensors(item)) if isinstance(value, (tuple,list)) else ()


class _RequestCache:
    def __init__(self, model, max_entries, max_bytes):
        self.model = model
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.owner = threading.get_ident()
        self.caches = {name: OrderedDict() for name in ('mask','context','rope','text','orbit')}
        self.counts = {f'{name}_{kind}': 0 for name in self.caches for kind in ('hits','misses','bypass')}
        self.counts.update(current_bytes=0, peak_bytes=0, evictions=0, clear_events=0, root_calls=0, orbit_restores=0)
        self.projection_tracker = _Tracker(model.text_embedder)
        self.rope_tracker = _Tracker(model.rope)
        self.context_key = None
        self.context_refs = ()
        self.active = False
        self.runtime = model.__dict__.get('oq_runtime')
        self.original_projection = model.text_embedder.forward

    def stats(self):
        return dict(self.counts, current_entries=sum(len(cache) for cache in self.caches.values()))

    as_dict = stats

    def clear(self):
        for cache in self.caches.values():
            cache.clear()
        self.context_key, self.context_refs = None, ()
        self.counts['current_bytes'] = 0
        self.counts['clear_events'] += 1

    def enabled(self):
        return not torch.is_grad_enabled() and not self.model.training and not self.model.text_embedder.training

    def lookup(self, kind, key, build, refs=()):
        if key is None or not self.enabled():
            self.counts[kind+'_bypass'] += 1
            return build()
        cache = self.caches[kind]
        found = cache.get(key)
        if found is not None:
            value, versions, saved_refs, size = found
            if tuple(t._version for t in _tensors(value)) == versions:
                cache.move_to_end(key)
                self.counts[kind+'_hits'] += 1
                return value
            self.counts['current_bytes'] -= size
            del cache[key]
        self.counts[kind+'_misses'] += 1
        value = build()
        tensors = _tensors(value)
        size = sum(t.numel()*t.element_size() for t in tensors)
        try:
            versions = tuple(t._version for t in tensors)
        except RuntimeError:
            self.counts[kind+'_bypass'] += 1
            return value
        while cache and (len(cache) >= self.max_entries or self.counts['current_bytes'] + size > self.max_bytes):
            _, (_,_,_,old_size) = cache.popitem(last=False)
            self.counts['current_bytes'] -= old_size
            self.counts['evictions'] += 1
        if self.counts['current_bytes'] + size <= self.max_bytes:
            cache[key] = (value, versions, refs, size)
            self.counts['current_bytes'] += size
            self.counts['peak_bytes'] = max(self.counts['peak_bytes'],self.counts['current_bytes'])
        else:
            self.counts[kind+'_bypass'] += 1
        return value

    def mask(self, embeds, mask, hidden):
        def build():
            result = mask
            if result is None:
                result = torch.ones(embeds.shape[:2], dtype=torch.bool, device=hidden.device)
            if result.shape != embeds.shape[:2] or not torch.all((result == 0) | (result == 1)):
                raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
            return result
        try:
            key = (_tensor_token(mask), tuple(embeds.shape[:2]), hidden.device)
        except RuntimeError:
            key = None
        return self.lookup('mask',key,build,(mask,))

    def context(self, embeds, mask, index):
        try:
            key = (_tensor_token(embeds), _tensor_token(mask), index)
        except RuntimeError:
            key = None
        self.context_key = key
        self.context_refs = (embeds,mask)
        return self.lookup('context',key,lambda: embeds[index:index+1,mask[index].bool()],(embeds,mask))

    def rope(self, height, width, length, device, dtype):
        if self.rope_tracker.module is not self.model.rope:
            self.rope_tracker = _Tracker(self.model.rope)
        # Save the preceding CFG branch's Orbit expansion before selecting this
        # branch. The existing per-layer Orbit function remains untouched.
        previous = None if self.runtime is None else self.runtime.__dict__.get('moss_image_rope')
        if previous is not None:
            try:
                previous_key = _tensor_token(previous[0])
            except RuntimeError:
                previous_key = None
            self.lookup('orbit', previous_key, lambda: previous, (previous[0],))
        try:
            key = (height,width,length,device,dtype,self.rope_tracker.token(),getattr(self.model.rope,'base',None))
        except RuntimeError:
            key = None
        result = self.lookup('rope',key,lambda: self.model.rope(height,width,length,device).to(dtype),self.rope_tracker.tensors)
        self.current_rope_key = key
        if self.runtime is not None:
            try:
                orbit_key = _tensor_token(result)
            except RuntimeError:
                orbit_key = None
            saved = self.caches['orbit'].get(orbit_key)
            if saved is not None and tuple(t._version for t in _tensors(saved[0])) == saved[1]:
                self.runtime.__dict__['moss_image_rope'] = saved[0]
                self.counts['orbit_restores'] += 1
        return result

    def projection(self, context, rope):
        try:
            key = None if self.context_key is None else (
                self.context_key, self.current_rope_key, tuple(context.shape),context.dtype,context.device,
                tuple(rope.shape),rope.dtype,rope.device,self.rope_tracker.token(),
                getattr(self.model.rope,'base',None),self.projection_tracker.token(),
                torch.is_autocast_enabled(context.device.type),torch.get_autocast_dtype(context.device.type))
        except RuntimeError:
            key = None
        if not self.active:
            key = None
        refs = self.context_refs + self.projection_tracker.tensors + self.rope_tracker.tensors
        return self.lookup('text',key,lambda: self.original_projection(context,rope),refs)


def _patch(original, cache):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise ValueError('Cannot inspect the pinned MOSS forward for exact request caching') from exc
    target = tree.body[0]
    if not isinstance(target,ast.FunctionDef) or target.name != 'forward':
        raise ValueError('Expected pinned original MOSS forward; install exact request caching before other wrappers')
    block = ast.parse("""mask = encoder_attention_mask
if mask is None:
    mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
    raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
""").body
    dump = lambda value: ast.dump(value,include_attributes=False)
    matches = [i for i in range(len(target.body)-2) if all(dump(a)==dump(b) for a,b in zip(target.body[i:i+3],block))]
    if len(matches) != 1:
        raise ValueError('Pinned MOSS mask validation pattern differs')
    start = matches[0]
    target.body[start:start+3] = ast.parse('mask = _exact_request_cache.mask(encoder_hidden_states, encoder_attention_mask, hidden_states)').body
    replacements = {
        dump(ast.parse('context = encoder_hidden_states[index:index + 1, mask[index].bool()]').body[0]):
            'context = _exact_request_cache.context(encoder_hidden_states, mask, index)',
        dump(ast.parse('rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)').body[0]):
            'rope = _exact_request_cache.rope(height, width, context.shape[1], hidden_states.device, video.dtype)',
    }
    counts = {key:0 for key in replacements}
    class Replace(ast.NodeTransformer):
        def visit_Assign(self,node):
            key = dump(node)
            if key in replacements:
                counts[key] += 1
                return ast.copy_location(ast.parse(replacements[key]).body[0],node)
            return self.generic_visit(node)
    tree = Replace().visit(tree)
    if set(counts.values()) != {1}:
        raise ValueError('Pinned MOSS context/RoPE source patterns differ')
    namespace = dict(original.__func__.__globals__,_exact_request_cache=cache)
    exec(compile(ast.fix_missing_locations(tree),'<moss-exact-request-cache>','exec'),namespace)
    return MethodType(namespace['forward'],original.__self__)


@contextmanager
def exact_request_cache(model, *, max_entries=2, max_bytes=32*1024*1024):
    """Cache unchanged prompt work during exactly one pipeline request.

    Install this context before Cache-DiT and Accelerate offload hooks; close in
    reverse order. The pinned source patterns must match exactly. Cache capacity
    counts tensors conservatively, including shared references between stages.
    Each stage holds at most ``max_entries`` entries, subject to ``max_bytes``
    across all stages. Tensor/parameter changes detected by PyTorch version
    counters invalidate entries. Untracked ``.data`` writes are unsupported.
    Grad-enabled execution and tensors without version counters bypass caching.
    Device transfers and context exit (including exceptional exit) clear cached tensors.
    """
    if type(max_entries) is not int or type(max_bytes) is not int or max_entries < 1 or max_bytes < 1:
        raise ValueError('Cache capacities must be positive integers')
    if hasattr(model,'_hf_hook') or model.__dict__.get('_exact_text_projection_cache_active'):
        raise ValueError('Install exact request caching before Accelerate and do not nest request caches')
    original = model.forward
    original_apply = model._apply
    had_forward, had_apply = 'forward' in model.__dict__, '_apply' in model.__dict__
    had_projection = 'forward' in model.text_embedder.__dict__
    cache = _RequestCache(model,max_entries,max_bytes)
    patched = _patch(original,cache)
    def forward(*args,**kwargs):
        if cache.active or threading.get_ident()!=cache.owner:
            raise RuntimeError('Concurrent/reentrant cached transformer call')
        cache.active = True
        cache.counts['root_calls'] += 1
        try:
            return patched(*args,**kwargs)
        finally:
            cache.active = False
            cache.context_key,cache.context_refs = None,()
    def apply(*args,**kwargs):
        cache.clear()
        return original_apply(*args,**kwargs)
    model.__dict__['_exact_text_projection_cache_active'] = True
    model.forward, model._apply, model.text_embedder.forward = forward,apply,cache.projection
    try:
        yield cache
    finally:
        cache.clear()
        # Accelerate keeps its outer hook; restore the wrapped payload.
        if hasattr(model,'_hf_hook') and hasattr(model,'_old_forward'):
            model._old_forward = original
        elif had_forward:
            model.forward = original
        else:
            model.__dict__.pop('forward',None)
        if had_apply:
            model._apply = original_apply
        else:
            model.__dict__.pop('_apply',None)
        if had_projection:
            model.text_embedder.forward = cache.original_projection
        else:
            model.text_embedder.__dict__.pop('forward',None)
        model.__dict__.pop('_exact_text_projection_cache_active',None)
