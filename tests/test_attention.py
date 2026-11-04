import math

import pytest
import torch

from attention_lab import blockwise_attention, standard_attention
from attention_lab.reference import TOLERANCES, errors, float64_reference, math_reference
from attention_lab.tiny import tiny_inputs


IMPLEMENTATIONS = [standard_attention, blockwise_attention]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", list(TOLERANCES))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1, 2), (2, 3, 7, 5), (1, 2, 17, 8)])
def test_against_independent_references(device, dtype, causal, shape):
    if device == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable in this execution session")
        if dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        try:
            # Probe library capabilities, not the implementations being tested.
            probe = torch.ones((1, 1, 3, 2), device=device, dtype=dtype)
            math_reference(probe, probe, probe)
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            pytest.skip(f"MPS dtype/reference unavailable: {exc}")
    rng = torch.Generator().manual_seed(2205)
    inputs = tuple(torch.randn(shape, generator=rng).to(device=device, dtype=dtype) for _ in range(3))
    high_precision = float64_reference(*inputs, causal=causal)
    same_device = math_reference(*inputs, causal=causal)
    atol, rtol = TOLERANCES[dtype]
    ordinary = standard_attention(*inputs, causal=causal)
    torch.testing.assert_close(ordinary.cpu().double(), high_precision, atol=atol, rtol=rtol)
    for qb, kb in [(1, 1), (3, 4), (4, 3), (32, 32), (2, 5)]:
        blocked = blockwise_attention(*inputs, query_block_size=qb, kv_block_size=kb, causal=causal)
        assert blocked.shape == inputs[0].shape
        assert blocked.dtype == dtype
        assert blocked.device == inputs[0].device
        assert torch.isfinite(blocked).all()
        torch.testing.assert_close(blocked.cpu().double(), high_precision, atol=atol, rtol=rtol)
        torch.testing.assert_close(blocked, same_device, atol=atol, rtol=rtol)
        torch.testing.assert_close(blocked, ordinary, atol=atol, rtol=rtol)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_future_keys_and_values_cannot_change_the_past(implementation):
    rng = torch.Generator().manual_seed(8)
    q, k, v = (torch.randn((2, 3, 9, 4), generator=rng) for _ in range(3))
    options = {"query_block_size": 3, "kv_block_size": 4} if implementation is blockwise_attention else {}
    before = implementation(q, k, v, causal=True, **options)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 5:] = 1000
    changed_v[:, :, 5:] = -500
    after = implementation(q, changed_k, changed_v, causal=True, **options)
    torch.testing.assert_close(before[:, :, :5], after[:, :, :5], atol=0, rtol=0)
    torch.testing.assert_close(before[:, :, 0], v[:, :, 0], atol=0, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_later_larger_score_rescales_both_accumulators(causal):
    q, k, v = tiny_inputs()
    events = []
    result = blockwise_attention(q, k, v, query_block_size=2, kv_block_size=2,
                                causal=causal, trace_callback=events.append)
    last = [e for e in events if e.query_start == 2][-1]
    assert last.old_max.item() == pytest.approx(1 / math.sqrt(2))
    assert last.new_max.item() == pytest.approx(4 / math.sqrt(2))
    assert last.rescale.item() == pytest.approx(math.exp(-3 / math.sqrt(2)))
    torch.testing.assert_close(last.new_sum, last.rescale * last.old_sum + last.exponentials.sum(-1, keepdim=True))
    torch.testing.assert_close(last.new_accumulator, last.rescale * last.old_accumulator + last.value_contributions.sum(-2))
    torch.testing.assert_close(result, float64_reference(q, k, v, causal=causal), atol=1e-10, rtol=1e-10)


def test_partial_and_wholly_masked_tiles_have_zero_contributions():
    q, k, v = tiny_inputs()
    events = []
    output = blockwise_attention(q, k, v, causal=True, query_block_size=2,
                                kv_block_size=1, trace_callback=events.append)
    assert not any(e.query_start == 0 and e.kv_start == 2 for e in events)
    partial = next(e for e in events if e.query_start == 0 and e.kv_start == 1)
    assert partial.exponentials[0, 0, 0, 0] == 0
    assert torch.isfinite(partial.new_accumulator).all()
    torch.testing.assert_close(output, float64_reference(q, k, v, causal=True), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", list(TOLERANCES))
@pytest.mark.parametrize("causal", [False, True])
def test_large_logits_and_low_precision_dot_product_promotion(dtype, causal):
    # fp16 Q@K would overflow before softmax without promotion. The logits
    # remain finite in float32; direct exp(logit) would also overflow.
    q = torch.tensor([[[[300., 0.], [300., 0.], [300., 0.]]]], dtype=dtype)
    k = torch.tensor([[[[300., 0.], [299., 0.], [301., 0.]]]], dtype=dtype)
    v = torch.tensor([[[[1., 2.], [2., -1.], [-3., 4.]]]], dtype=dtype)
    atol, rtol = TOLERANCES[dtype]
    for output in (standard_attention(q, k, v, causal=causal),
                   blockwise_attention(q, k, v, causal=causal, query_block_size=2, kv_block_size=2)):
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), float64_reference(q, k, v, causal=causal), atol=atol, rtol=rtol)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("causal", [False, True])
def test_equal_logits_are_uniform_over_allowed_values(implementation, causal):
    q = torch.zeros(1, 1, 5, 2)
    k = torch.ones_like(q)
    v = torch.arange(10, dtype=torch.float32).reshape_as(q)
    expected = torch.stack([v[0, 0, :i + 1].mean(0) for i in range(5)]) if causal else v[0, 0].mean(0).expand(5, 2)
    torch.testing.assert_close(implementation(q, k, v, causal=causal)[0, 0], expected)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_noncontiguous_inputs_and_forward_only(implementation):
    rng = torch.Generator().manual_seed(19)
    inputs = tuple(torch.randn(2, 3, 4, 7, generator=rng).transpose(-1, -2).requires_grad_() for _ in range(3))
    assert not inputs[0].is_contiguous()
    output = implementation(*inputs)
    assert not output.requires_grad
    torch.testing.assert_close(output.double(), float64_reference(*inputs), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("bad", ["rank", "shape", "empty", "integer", "dtype"])
def test_invalid_inputs(implementation, bad):
    q, k, v = tiny_inputs()
    if bad == "rank": q = q[0]
    if bad == "shape": k = k[:, :, :2]
    if bad == "empty": q, k, v = (x[:, :, :0] for x in (q, k, v))
    if bad == "integer": q, k, v = (x.long() for x in (q, k, v))
    if bad == "dtype": v = v.float()
    with pytest.raises(ValueError):
        implementation(q, k, v)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
@pytest.mark.parametrize("argument", ["query_block_size", "kv_block_size"])
def test_invalid_block_sizes(size, argument):
    with pytest.raises(ValueError, match="positive integers"):
        blockwise_attention(*tiny_inputs(), **{argument: size})


def test_zero_reference_error_is_defined():
    zero = torch.zeros(1, 1, 3, 2)
    assert errors(zero, zero) == {"max_abs_error": 0.0, "relative_l2_error": 0.0}
