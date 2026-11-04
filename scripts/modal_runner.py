"""Explicitly authorized execution only. Import/help never invokes remote work."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('attention-lab-bounded')
image = (modal.Image.debian_slim(python_version='3.13')
         .pip_install('torch==2.8.0', 'triton==3.4.0', 'pytest==8.3.4', 'matplotlib==3.10.3')
         .env({'PYTHONPATH': '/lab/src', 'MPLCONFIGDIR': '/tmp/matplotlib'})
         .add_local_dir(ROOT / 'src', remote_path='/lab/src')
         .add_local_dir(ROOT / 'tests', remote_path='/lab/tests'))


@app.function(image=image, gpu='L4', cpu=(2, 2), memory=(4096, 8192), timeout=600,
              startup_timeout=600, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, single_use_containers=True)
def run():
    import os
    import subprocess
    import time
    import torch
    from attention_lab.cloud_checks import run_verified_benchmark
    from attention_lab.benchmark import run_benchmark
    from attention_lab.plots import plot_results
    from attention_lab.devices import resolve_device
    resolve_device('cuda')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Requires SM80+')
    started = time.perf_counter()
    completed = subprocess.run(
        ['python', '-m', 'pytest', '/lab/tests/test_triton.py', '-q', '-rs', '--junitxml=/tmp/triton.xml'],
        capture_output=True, text=True, timeout=240,
        env={**os.environ, 'PYTHONPATH': '/lab/src'})
    artifacts = {'triton-tests.txt': (completed.stdout + completed.stderr).encode()}
    if Path('/tmp/triton.xml').exists():
        artifacts['triton-tests.xml'] = Path('/tmp/triton.xml').read_bytes()
    try:
        report = run_verified_benchmark(
            completed.returncode, artifacts.get('triton-tests.xml', b''),
            lambda: run_benchmark(device='cuda', dtype='float16', profile='nvidia', output='/tmp/bench',
                                  implementations=['standard', 'blockwise', 'sdpa', 'triton'], warmups=3, repeats=10))
    except ValueError as exc:
        artifacts['correctness-gate.txt'] = str(exc).encode()
        return {'status': 'correctness_failed', 'artifacts': artifacts}
    plot_results('/tmp/bench')
    artifacts.update({p.name: p.read_bytes() for p in Path('/tmp/bench').iterdir() if p.is_file()})
    artifacts['runtime.txt'] = str(time.perf_counter() - started).encode()
    return {'status': report['status'], 'artifacts': artifacts}


@app.local_entrypoint()
def main(output: str = 'results/modal-l4'):
    result = run.remote()  # Exactly one call; no spawn/map/retry/schedule/deploy.
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in result['artifacts'].items():
        (destination / name).write_bytes(data)
    print(f"{result['status']}: {destination}")
    if result['status'] != 'passed':
        raise RuntimeError('Remote validation/benchmark failed; inspect saved artifacts')
