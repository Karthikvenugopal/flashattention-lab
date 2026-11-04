"""Independent GPU tests. Skips explicitly when no compatible NVIDIA hardware."""
import pytest
import torch
from attention_lab.triton_attention import triton_attention
from attention_lab.reference import float64_reference, TOLERANCES


def test_triton_rejects_cpu_without_importing_triton():
    x = torch.ones(1, 1, 3, 5)
    with pytest.raises(ValueError, match='NVIDIA'):
        triton_attention(x, x, x)


cuda = torch.cuda.is_available() and torch.version.hip is None
compatible = cuda and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(not compatible, reason='Requires NVIDIA SM80+ and Triton')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('shape,qb,kb', [((1, 1, 1, 1), 16, 16), ((2, 3, 37, 19), 16, 32),
                                       ((1, 2, 129, 64), 32, 16), ((1, 1, 137, 128), 64, 64)])
def test_fused_reference(shape, qb, kb, causal, dtype):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(872)
    inputs = [torch.randn(shape, generator=g).to(device='cuda', dtype=dtype) for _ in range(3)]
    actual = triton_attention(*inputs, causal=causal, query_block_size=qb, kv_block_size=kb)
    torch.cuda.synchronize()
    atol, rtol = TOLERANCES[dtype]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu().double(), float64_reference(*inputs, causal=causal), atol=atol, rtol=rtol)
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(*inputs, is_causal=causal)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    if causal and shape[2] > 1:
        # Future K/V perturbations must not affect earlier query outputs.
        k, v = inputs[1].clone(), inputs[2].clone()
        split = shape[2] // 2
        k[:, :, split:] += 3
        v[:, :, split:] -= 7
        changed = triton_attention(inputs[0], k, v, causal=True, query_block_size=qb, kv_block_size=kb)
        torch.testing.assert_close(actual[:, :, :split], changed[:, :, :split], atol=atol, rtol=rtol)
        torch.testing.assert_close(actual[:, :, 0], inputs[2][:, :, 0], atol=atol, rtol=rtol)
