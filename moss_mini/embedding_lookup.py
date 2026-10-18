"""Optional CPU embedding lookup after Accelerate dispatch.

Keeping the vocabulary on CPU reduces transfers and GPU memory use, but can
increase encoding latency. The caller chooses layer offload separately.
"""


def install_cpu_embedding_lookup(text_model):
    """Replace CPU weight offload with CPU lookup and return placement metadata.

    Install once after dispatch_model, before inference. Accelerate normally
    copies an entire offloaded vocabulary onto its execution device. This hook
    instead copies the small token-ID tensor to CPU and returns only the selected
    embedding rows to the incoming tensor's device. No weights or state keys
    change. The remaining model's placement is untouched.
    """
    import torch
    from torch import nn
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

    embedding = text_model.get_input_embeddings()
    if type(embedding) is not nn.Embedding or embedding.training or embedding.weight.requires_grad:
        raise ValueError('Require a frozen eval ordinary Embedding')
    if embedding.max_norm is not None:
        raise ValueError('CPU lookup requires an embedding without in-place max_norm updates')
    hook = getattr(embedding, '_hf_hook', None)
    if not isinstance(hook, AlignDevicesHook) or not hook.offload:
        raise ValueError('Install after dispatch_model has attached embedding CPU offload')
    if hook.original_devices.get('weight') != torch.device('cpu'):
        raise ValueError('The original embedding weight must live on CPU')
    backing = hook.weights_map['weight']
    if backing.device.type != 'cpu' or backing.dtype != torch.bfloat16:
        raise ValueError('Require the original BF16 CPU backing weight')
    if tuple(backing.shape) != (embedding.num_embeddings, embedding.embedding_dim):
        raise ValueError('CPU embedding backing shape mismatch')

    # detach_hook restores the original CPU tensor from its offload weights_map.
    remove_hook_from_module(embedding)
    if (embedding.weight.device.type != 'cpu' or embedding.weight.dtype != backing.dtype
            or embedding.weight.shape != backing.shape or embedding.weight.data_ptr() != backing.data_ptr()):
        raise RuntimeError('CPU embedding restore failed')
    add_hook_to_module(embedding, AlignDevicesHook(
        execution_device=torch.device('cpu'), offload=False, io_same_device=True))
    return {'strategy': 'cpu_embedding_lookup',
            'weight_bytes': embedding.weight.numel()*embedding.weight.element_size(),
            'shape': list(embedding.weight.shape), 'output_device': 'same as incoming input_ids',
            'state_key_changed': False}
