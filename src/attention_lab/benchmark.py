"""Small deterministic OFAT sweeps; synchronized wall latency, no timed references."""
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import csv
import json
import platform
import statistics
import subprocess
import time
import torch
from .attention import standard_attention, blockwise_attention
from .sdpa import sdpa_attention, identify_backend
from .devices import environment, resolve_device
from .reference import float64_reference, errors, TOLERANCES


@dataclass(frozen=True)
class Config:
    batch: int = 1
    heads: int = 2
    length: int = 128
    dimension: int = 32
    query_block_size: int = 32
    kv_block_size: int = 32
    causal: bool = False
    sweep: str = 'anchor'


def configurations(profile='small'):
    if profile not in ('small', 'nvidia'):
        raise ValueError('Unknown profile')
    base = Config() if profile == 'small' else Config(length=512, dimension=64, heads=4, query_block_size=64)
    values = {'length': [32, 64, 128, 256, 512] if profile == 'small' else [128, 512, 1024, 2048],
              'dimension': [16, 32, 64, 128], 'batch': [1, 2, 4], 'heads': [1, 2, 4, 8],
              'query_block_size': [16, 32, 64], 'kv_block_size': [16, 32, 64]}
    cases = [base]
    for field, choices in values.items():
        cases += [replace(base, **{field: value}, sweep=field) for value in choices if value != getattr(base, field)]
    cases += [replace(base, causal=True, sweep='causal'), replace(base, length=137, causal=True, sweep='partial')]
    return cases


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def metadata():
    result = environment()
    result.update(cpu=platform.processor(), cpu_threads=torch.get_num_threads(), cuda_version=torch.version.cuda)
    for package in ('triton', 'matplotlib', 'modal'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    if platform.system() == 'Darwin':
        for key in ('hw.model', 'hw.memsize', 'machdep.cpu.brand_string'):
            try:
                result[key] = subprocess.check_output(['sysctl', '-n', key], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                result[key] = 'unknown (query restricted)'
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name()
        result['compute_capability'] = torch.cuda.get_device_capability()
        result['gpu_total_memory'] = torch.cuda.get_device_properties(0).total_memory
    return result


def memory_probe(fn, device):
    """Separate execution from latency; no CPU/MPS peak claims."""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = fn()
        synchronize(device)
        data = {'kind': 'cuda_peak_allocated_increment', 'bytes': torch.cuda.max_memory_allocated() - before,
                'baseline_bytes': before}
    elif device == 'mps':
        before = torch.mps.current_allocated_memory()
        out = fn()
        synchronize(device)
        data = {'kind': 'mps_current_after_call_not_peak', 'bytes': torch.mps.current_allocated_memory(),
                'baseline_bytes': before, 'driver_bytes': torch.mps.driver_allocated_memory()}
    else:
        out = fn()
        data = {'kind': 'unavailable', 'bytes': None}
    del out
    return data


def write_results(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows = []
    samples = []
    for case in report['cases']:
        row = {'device': report['device'], 'dtype': report['dtype'], 'seed': report['seed'],
               'warmups': report['warmups'], 'repeats': report['repeats'],
               **{k: v for k, v in case.items() if k not in ('config', 'samples_ms', 'backend_evidence', 'memory')},
               **case['config'], 'backend': case.get('backend_evidence', {}).get('backend', 'unknown'),
               **{'memory_' + k: v for k, v in case.get('memory', {}).items()}}
        rows.append(row)
        samples += [{'case_id': case['case_id'], 'repeat': i, 'latency_ms': t} for i, t in enumerate(case.get('samples_ms', []))]
    for name, data in [('summary.csv', rows), ('timings.csv', samples)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)


@torch.inference_mode()
def run_benchmark(*, device='cpu', dtype='float32', output='results/benchmark-cpu', profile='small',
                  warmups=3, repeats=10, seed=2205, threads=1, implementations=None, configs=None):
    if warmups < 1 or repeats < 2 or threads < 1:
        raise ValueError('Require warmups >= 1, repeats >= 2 and threads >= 1')
    torch.set_num_threads(threads)
    report = {'kind': 'forward benchmark', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'environment': metadata(), 'device': device, 'dtype': dtype, 'seed': seed,
              'warmups': warmups, 'repeats': repeats, 'profile': profile,
              'timing': 'synchronized perf_counter wall time per call; inputs and reference excluded',
              'first_call': 'first invocation of this configuration in this process; includes cold work/JIT if any, cache may be warm',
              'cases': [], 'status': 'running'}
    try:
        target = resolve_device(device)
    except RuntimeError as exc:
        report.update(status='unavailable', error=str(exc))
        write_results(report, output)
        return report
    dt = getattr(torch, dtype)
    methods = {'standard': standard_attention, 'blockwise': blockwise_attention, 'sdpa': sdpa_attention}
    if implementations and 'triton' in implementations:
        from .triton_attention import triton_attention
        methods['triton'] = triton_attention
    names = list(methods) if implementations is None else implementations
    for config in (configurations(profile) if configs is None else configs):
        generator = torch.Generator().manual_seed(seed)
        c = asdict(config)
        shape = (config.batch, config.heads, config.length, config.dimension)
        inputs = tuple(torch.randn(shape, generator=generator).to(device=target, dtype=dt) for _ in range(3))
        # A CPU double reference uses the actual rounded input values.
        reference = float64_reference(*inputs, causal=config.causal)
        for name in names:
            case = {'case_id': len(report['cases']), 'implementation': name, 'config': c, 'status': 'running',
                    'input_bytes': 3 * inputs[0].numel() * inputs[0].element_size(),
                    'analytical_two_dense_score_buffers_bytes': 2 * config.batch * config.heads * config.length**2 * (8 if dt == torch.float64 else 4)}
            kwargs = {'causal': config.causal}
            if name in ('blockwise', 'triton'):
                kwargs.update(query_block_size=config.query_block_size, kv_block_size=config.kv_block_size)
            fn = lambda: methods[name](*inputs, **kwargs)
            try:
                synchronize(device)
                start = time.perf_counter()
                actual = fn()
                synchronize(device)
                case['first_call_ms'] = (time.perf_counter() - start) * 1000
                atol, rtol = TOLERANCES[dt]
                case.update(errors(actual, reference), atol=atol, rtol=rtol)
                torch.testing.assert_close(actual.cpu().double(), reference, atol=atol, rtol=rtol)
                del actual
                for _ in range(warmups):
                    fn()
                synchronize(device)
                samples = []
                for _ in range(repeats):
                    synchronize(device)
                    start = time.perf_counter()
                    out = fn()
                    synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000)
                    del out
                case.update(samples_ms=samples, median_ms=statistics.median(samples),
                            min_ms=min(samples), max_ms=max(samples), stdev_ms=statistics.stdev(samples),
                            memory=memory_probe(fn, device), status='passed')
                if name == 'sdpa':
                    case['backend_evidence'] = identify_backend(fn)
                    synchronize(device)
            except (RuntimeError, AssertionError, ValueError, ImportError, NotImplementedError) as exc:
                case.update(status='failed', error=str(exc))
            report['cases'].append(case)
            write_results(report, output)
        del inputs, reference
    report['status'] = 'passed' if report['cases'] and all(c['status'] == 'passed' for c in report['cases']) else 'failed'
    write_results(report, output)
    return report
