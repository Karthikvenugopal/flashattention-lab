"""Run only inside the isolated Colab environment; no Modal dependency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    import torch
    import triton
    from attention_lab.benchmark import metadata, run_benchmark
    from attention_lab.cloud_checks import run_verified_benchmark
    from attention_lab.plots import plot_results
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError('NVIDIA CUDA is unavailable in the installed environment; no tests or benchmarks ran.')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('GPU changed: the existing Triton kernel requires SM80+. No benchmarks ran.')
    if torch.__version__.split('+')[0] != '2.8.0' or triton.__version__ != '3.4.0':
        raise RuntimeError('Installed versions differ from the pinned stack; no benchmarks ran.')
    torch.cuda.synchronize()
    (output / 'environment.json').write_text(json.dumps(metadata(), indent=2) + '\n')
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=30, check=True)
    (output / 'dependencies.txt').write_text(frozen.stdout)
    # Remove stale evidence before any repeated attempt.
    junit = output / 'triton-tests.xml'
    junit.unlink(missing_ok=True)
    command = [sys.executable, '-m', 'pytest', str(args.root / 'tests/test_triton.py'),
               '-q', '-rs', f'--junitxml={junit}']
    with (output / 'triton-tests.log').open('w') as log:
        completed = subprocess.run(command, cwd=args.root, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}, timeout=240)
    print((output / 'triton-tests.log').read_text(), flush=True)

    def benchmarks():
        # This callback is never invoked until the unchanged GPU tests all pass.
        print('All 16 Triton correctness cases passed. Starting benchmark sweep.', flush=True)
        return run_benchmark(device='cuda', dtype='float16', profile='nvidia', output=output / 'benchmark',
                             implementations=['standard', 'blockwise', 'sdpa', 'triton'], warmups=3, repeats=10)

    try:
        report = run_verified_benchmark(completed.returncode, junit.read_bytes() if junit.exists() else b'', benchmarks)
    except ValueError as exc:
        (output / 'correctness-gate.txt').write_text(str(exc) + '\n')
        raise RuntimeError('Correctness gate blocked benchmarks: ' + str(exc)) from exc
    plot_results(output / 'benchmark')
    if report['status'] != 'passed':
        raise RuntimeError('Benchmark has failed cases; raw failures and timing data are preserved.')
    print(f"PASSED: {len(report['cases'])} benchmark cases; raw CSV/JSON and plots saved.")


if __name__ == '__main__':
    main()
