import json
import pytest
import torch
from attention_lab.sdpa import sdpa_attention, identify_backend
from attention_lab.reference import float64_reference, TOLERANCES
from attention_lab.benchmark import Config, run_benchmark, configurations


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_sdpa_independent(device, causal, dtype):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    g = torch.Generator().manual_seed(121)
    inputs = [torch.randn(2, 3, 37, 19, generator=g).to(device=device, dtype=dtype) for _ in range(3)]
    actual = sdpa_attention(*inputs, causal=causal)
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.cpu().double(), float64_reference(*inputs, causal=causal), atol=atol, rtol=rtol)


def test_backend_unknown_on_profiler_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('no profiler')
    monkeypatch.setattr(torch.profiler, 'profile', fail)
    assert identify_backend(lambda: None)['backend'] == 'unknown'


def test_sweeps_cover_required_axes():
    cases = configurations()
    for field in ('length', 'dimension', 'batch', 'heads', 'query_block_size', 'kv_block_size'):
        assert len({getattr(c, field) for c in cases}) >= 3


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_benchmark_artifacts(tmp_path, device, record_testsuite_property):
    # Same harness as CLI, preserving process-specific MPS visibility in JUnit.
    report = run_benchmark(device=device, output=tmp_path, warmups=3 if device == 'mps' else 1, repeats=10 if device == 'mps' else 2,
                           configs=None if device == 'mps' else [Config(length=17, dimension=8, causal=True)])
    assert json.loads((tmp_path / 'benchmark.json').read_text()) == report
    if report['status'] == 'unavailable':
        pytest.skip(report['error'])
    assert report['status'] == 'passed'
    record_testsuite_property(f'benchmark_{device}', json.dumps(report))
    for case in report['cases']:
        assert len(case['samples_ms']) == report['repeats']
        assert case['first_call_ms'] > 0
        assert case['memory']['kind'] != 'peak' if device == 'mps' else True
    assert len((tmp_path / 'timings.csv').read_text().splitlines()) == 1 + len(report['cases']) * report['repeats']
