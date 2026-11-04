"""Verify released measurements and source identity without running any GPU code."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(root=ROOT):
    root = Path(root)
    provenance = json.loads((root / 'results/provenance/import.json').read_text())
    source_hashes = json.loads((root / 'results/provenance/nvidia-source-sha256.json').read_text())
    source_matches = {}
    with zipfile.ZipFile(root / 'results/provenance/nvidia-source.zip') as source:
        for name, expected in source_hashes.items():
            assert digest(source.read(name)) == expected, f'Snapshot hash mismatch: {name}'
        # Plotting is the documented, non-computational change after the run.
        for name in source.namelist():
            if name.startswith('src/') and name.endswith('.py') and not name.endswith('/plots.py'):
                source_matches[name] = digest((root / name).read_bytes()) == source_hashes[name]
        name = 'tests/test_triton.py'
        source_matches[name] = digest((root / name).read_bytes()) == source_hashes[name]
    assert all(source_matches.values()), 'Published computation/test source changed: recorded GPU results require revalidation'
    mac_hashes = json.loads((root / 'results/provenance/mac-source-sha256.json').read_text())
    for name in ('attention.py', 'benchmark.py', 'reference.py', 'sdpa.py'):
        path = 'src/attention_lab/' + name
        assert digest((root / path).read_bytes()) == mac_hashes[path], 'Mac computation changed: results require revalidation'
    results = {}
    reports = {}
    for record in provenance['datasets']:
        device = record['dataset']
        directory = root / 'results' / device
        report = json.loads((directory / 'benchmark.json').read_text())
        reports[device] = report
        cases = report['cases']
        assert report['status'] == 'passed' and cases
        assert len({c['case_id'] for c in cases}) == len(cases)
        assert digest(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode()) == record['cases_sha256'], device
        assert digest((directory / 'benchmark.json').read_bytes()) == record['released_benchmark_sha256'], device
        timings = list(csv.DictReader((directory / 'timings.csv').open()))
        summaries = list(csv.DictReader((directory / 'summary.csv').open()))
        assert len(summaries) == len(cases)
        assert len(timings) == len(cases) * report['repeats']
        for case in cases:
            assert case['status'] == 'passed', (device, case['case_id'])
            raw = [row for row in timings if int(row['case_id']) == case['case_id']]
            assert [int(row['repeat']) for row in raw] == list(range(report['repeats']))
            samples = [float(row['latency_ms']) for row in raw]
            assert samples == case['samples_ms']
            assert statistics.median(samples) == case['median_ms']
            assert min(samples) == case['min_ms'] and max(samples) == case['max_ms']
            assert statistics.stdev(samples) == case['stdev_ms']
            row = next(s for s in summaries if int(s['case_id']) == case['case_id'])
            for field in ['median_ms', 'min_ms', 'max_ms', 'stdev_ms', 'first_call_ms', 'max_abs_error', 'relative_l2_error']:
                assert float(row[field]) == case[field], (device, field)
            for field, value in case['config'].items():
                assert row[field].lower() == str(value).lower(), (device, field)
            for field, value in case['memory'].items():
                assert row['memory_' + field] == ('' if value is None else str(value))
        results[device] = {'cases': len(cases), 'samples': len(timings), 'status': report['status'],
                           'max_abs_error': max(c['max_abs_error'] for c in cases)}
    gpu = reports['nvidia-l4']
    hardware = json.loads((root / 'results/nvidia-l4/hardware.json').read_text())
    assert hardware['gpu_name'] == gpu['environment']['gpu'] == 'NVIDIA L4'
    assert hardware['compute_capability'] == '8.9'
    # NVML (nvidia-smi) and CUDA can expose different memory totals. Keep both.
    assert 0 < gpu['environment']['gpu_total_memory'] <= int(hardware['memory_mib']) * 2**20
    assert gpu['dtype'] == 'float16' and gpu['warmups'] == 3 and gpu['repeats'] == 10
    junit = ET.parse(root / 'results/nvidia-l4/correctness.xml')
    tests = junit.findall('.//testcase')
    assert len(tests) == 17
    assert not any(case.find(tag) is not None for case in tests for tag in ['failure', 'error', 'skipped'])
    assert len([t for t in tests if t.get('name', '').startswith('test_fused_reference[')]) == 16
    groups = {}
    for case in gpu['cases']:
        key = json.dumps(case['config'], sort_keys=True)
        groups.setdefault(key, {})[case['implementation']] = case
    assert all(set(g) == {'standard', 'blockwise', 'sdpa', 'triton'} for g in groups.values())
    target = next(g for g in groups.values() if g['triton']['config']['sweep'] == 'length'
                  and g['triton']['config']['length'] == 2048)
    assert target['triton']['config'] == {
        'batch': 1, 'heads': 4, 'length': 2048, 'dimension': 64,
        'query_block_size': 64, 'kv_block_size': 32, 'causal': False, 'sweep': 'length'}
    standard, triton, sdpa = (target[m]['median_ms'] for m in ['standard', 'triton', 'sdpa'])
    speedup = standard / triton
    slowdown = triton / sdpa
    assert round(speedup, 1) == 10.6 and round(triton, 3) == .188
    assert round((slowdown - 1) * 100, 1) == 27.5
    return {'status': 'passed', 'datasets': results, 'nvidia_correctness_tests': 17,
            'nvidia_source_matches': source_matches, 'computation_requires_revalidation': False,
            'plotting_change': 'Agg selected before Matplotlib import; computation unchanged',
            'highlight': {'hardware': hardware, 'config': target['triton']['config'],
                          'dtype': gpu['dtype'], 'statistic': 'median synchronized wall milliseconds of 10 repeats after 3 warmups',
                          'case_ids': {m: target[m]['case_id'] for m in target},
                          'median_ms': {m: target[m]['median_ms'] for m in target},
                          'triton_speedup_vs_standard': speedup,
                          'triton_latency_ratio_vs_sdpa': slowdown,
                          'sdpa_backend': target['sdpa']['backend_evidence']},
            'triton_faster_than_standard_configurations': sum(g['triton']['median_ms'] < g['standard']['median_ms'] for g in groups.values()),
            'triton_slower_than_sdpa_configurations': sum(g['triton']['median_ms'] > g['sdpa']['median_ms'] for g in groups.values()),
            'configurations': len(groups)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Save the deterministic audit report')
    args = parser.parse_args()
    report = audit()
    if args.write:
        (ROOT / 'results/audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
