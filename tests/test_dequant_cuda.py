import pytest
import torch
from moss_mini.encoder import PackedLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA witness; CPU CI skips")
@pytest.mark.parametrize('bits', [4, 6, 8])
def test_fused_unpack_forward_exact(bits):
    pytest.importorskip('triton')
    torch.manual_seed(518)
    weight = torch.randn(37, 256, device='cuda', dtype=torch.bfloat16)
    weight[0].zero_()
    bias = torch.randn(37, device='cuda', dtype=torch.bfloat16)
    layer = PackedLinear.from_weight(weight, bits, 64, 'cuda', bias)
    x = torch.randn(1, 23, 256, device='cuda', dtype=torch.bfloat16)
    expected = layer(x)
    layer.dequant_backend = 'triton'
    assert torch.equal(layer(x), expected)
