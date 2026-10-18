import copy
import pytest
import torch
from moss_mini.encoder import PackedLinear, pack_codes, unpack_codes, recipe, validate_manifest


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_packing_and_independent_quantization(bits):
    torch.manual_seed(518)
    codes = torch.arange(2**bits, dtype=torch.uint8).repeat(8).reshape(8, -1)
    assert torch.equal(unpack_codes(pack_codes(codes, bits), bits), codes)
    weight = torch.randn(96, 256, dtype=torch.bfloat16)
    weight[0].zero_()
    bias = torch.randn(96, dtype=torch.bfloat16)
    packed = PackedLinear.from_weight(weight, bits, 64, "cpu", bias)
    group = 256 if bits == 8 else 64
    values = weight.float().reshape(96, -1, group)
    scale = values.abs().amax(-1) / (2**(bits-1)-1)
    scale[scale == 0] = 1
    q = (values / scale[..., None]).round().clamp(-(2**(bits-1)-1), 2**(bits-1)-1)
    reference = (q * scale[..., None]).reshape_as(weight).to(torch.bfloat16)
    inputs = torch.randn(2, 7, 256, dtype=torch.bfloat16)
    assert torch.equal(packed(inputs), torch.nn.functional.linear(inputs, reference, bias))
    assert torch.equal(packed(inputs)[..., 0], bias[0].expand(2, 7))
    with pytest.raises(ValueError, match="scale"):
        PackedLinear(packed.qweight, packed.scale[:1], bits, group, 256)


def test_int6_layout_known_bytes():
    codes = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    assert pack_codes(codes, 6).tolist() == [[64, 32, 12]]


def test_relocatable_manifest_requires_identity_and_recipe():
    config = {"hidden_size": 4096}
    identity = {"kind": "local", "version": "example-v1"}
    manifest = dict(format_version=1, status="complete", source_identity=identity,
                    text_config=config, bits=6, group_size=64, recipe=recipe(6, 64))
    # No absolute path participates: a moved artifact with the same identity works.
    validate_manifest(manifest, config, identity, 6, 64)
    with pytest.raises(ValueError):
        validate_manifest(manifest, config, {"kind": "local", "version": "example-v2"}, 6, 64)
    with pytest.raises(ValueError):
        validate_manifest(manifest, config, identity, 4, 64)
    incomplete = copy.deepcopy(manifest)
    incomplete["status"] = "writing"
    with pytest.raises(ValueError):
        validate_manifest(incomplete, config, identity, 6, 64)
