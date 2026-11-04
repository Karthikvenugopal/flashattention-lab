"""Standard-library-only Colab setup. Hardware checks precede every installation."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import zipfile


def write_status(context, status, **extra):
    context['status'] = status
    path = Path(context['output']) / 'status.json'
    path.write_text(json.dumps({**context, **extra}, indent=2) + '\n')


def bundle(context):
    output = Path(context['output'])
    path = shutil.make_archive(str(output.parent / 'attention-lab-results'), 'zip', output)
    print(f'Downloadable results: {path}')
    return path


def initialize(payload, expected_sha256):
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError('Embedded source checksum mismatch; no installation attempted')
    parent = '/content' if Path('/content').is_dir() else None
    root = Path(tempfile.mkdtemp(prefix='attention-lab-', dir=parent))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if not (root / info.filename).resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe embedded archive path')
        archive.extractall(root)
    output = root / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    (output / 'source.zip').write_bytes(payload)
    (output / 'source-sha256.json').write_text(json.dumps({
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src', 'tests', 'scripts') for p in (root / folder).rglob('*.py')}, indent=2))
    context = {'root': str(root), 'output': str(output), 'source_sha256': expected_sha256,
               'preflight_passed': False, 'installed': False}
    write_status(context, 'initialized')
    return context


def preflight(context):
    context['preflight_passed'] = False
    context['installed'] = False
    output = Path(context['output'])
    info = {'python': sys.version, 'system': platform.system(), 'architecture': platform.machine()}
    try:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise RuntimeError('This notebook requires a Linux x86_64 NVIDIA GPU runtime.')
        if not (3, 10) <= sys.version_info[:2] <= (3, 13):
            raise RuntimeError('Pinned dependencies require Python 3.10–3.13. Select a compatible Colab runtime.')
        command = ['nvidia-smi', '--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version',
                   '--format=csv,noheader,nounits']
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        (output / 'nvidia-smi.txt').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError('NVIDIA GPU query failed. Select Runtime → Change runtime type → GPU.')
        rows = list(csv.reader(io.StringIO(completed.stdout)))
        if not rows:
            raise RuntimeError('No NVIDIA GPU was assigned. Select a GPU runtime and rerun from the first cell.')
        row = [v.strip() for v in rows[0]]
        if len(row) != 6:
            raise RuntimeError('Cannot identify GPU compute capability from nvidia-smi; refusing dependency installation.')
        index, uuid, name, capability, memory, driver = row
        info.update(gpu_index=index, gpu_uuid=uuid, gpu_name=name, compute_capability=capability,
                    memory_mib=memory, driver=driver)
        if not re.fullmatch(r'\d+\.\d+', capability):
            raise RuntimeError(f'Unknown compute capability {capability!r}; refusing dependency installation.')
        major, minor = map(int, capability.split('.'))
        if major < 8:
            raise RuntimeError(f'Incompatible GPU: {name} has SM{major}{minor}; this existing kernel requires SM80+. '
                               'T4 (SM75), P100 (SM60), and V100 (SM70) cannot run these tests. '
                               'Use an Ampere-or-newer assignment such as A100 or L4. No dependencies installed.')
        driver_parts = tuple(map(int, driver.split('.')))
        if driver_parts < (570, 26):
            raise RuntimeError(f'GPU {name} meets SM80+, but driver {driver} is below this notebook’s conservative '
                               '570.26 minimum for pinned CUDA 12.8. No dependencies installed. '
                               'Select a runtime with a newer NVIDIA driver.')
        context['gpu'] = info
        context['preflight_passed'] = True
        write_status(context, 'compatible')
        print(json.dumps(info, indent=2))
        return info
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        info['error'] = str(exc)
        write_status(context, 'incompatible_or_unavailable', reason=str(exc))
        raise RuntimeError(f'STOP: {exc}') from exc
    finally:
        (output / 'hardware.json').write_text(json.dumps(info, indent=2) + '\n')
        if not context['preflight_passed']:
            bundle(context)


def install_dependencies(context):
    # Recheck live hardware even if cells are rerun out of order.
    if not context.get('preflight_passed'):
        raise RuntimeError('STOP: GPU preflight must pass before dependency installation.')
    preflight(context)
    root = Path(context['root'])
    output = Path(context['output'])
    python = root / '.venv' / 'bin' / 'python'
    commands = [
        [sys.executable, '-m', 'venv', '--without-pip', str(root / '.venv')],
        [sys.executable, '-m', 'pip', '--python', str(python), 'install', '--disable-pip-version-check', 'pip==25.2'],
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url',
         'https://download.pytorch.org/whl/cu128', 'torch==2.8.0'],
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
         'triton==3.4.0', 'pytest==8.3.4', 'matplotlib==3.10.3'],
    ]
    try:
        write_status(context, 'installing')
        with (output / 'install.log').open('w') as log:
            for command in commands:
                print('Running:', ' '.join(command), flush=True)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        context.update(installed=True, python=str(python))
        write_status(context, 'installed')
    except Exception as exc:
        write_status(context, 'installation_failed', reason=str(exc))
        raise
    finally:
        bundle(context)


def run_experiment(context):
    if not context.get('installed') or not context.get('preflight_passed'):
        raise RuntimeError('STOP: GPU checks and dependency installation must pass before tests/benchmarks.')
    root = Path(context['root'])
    output = Path(context['output'])
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'CUDA_VISIBLE_DEVICES': context['gpu']['gpu_uuid'],
           'MPLCONFIGDIR': str(root / '.mpl'), 'MPLBACKEND': 'Agg', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
    attempt = output / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt.mkdir(parents=True)
    context['last_attempt'] = str(attempt)
    command = [context['python'], str(root / 'scripts/colab_gpu_worker.py'), '--root', str(root), '--output', str(attempt)]
    write_status(context, 'running_correctness_then_benchmarks')
    try:
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=600)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        print((attempt / 'worker.log').read_text()[-12000:])
        if code:
            raise RuntimeError(f'Tests/benchmarks stopped with exit {code}; inspect worker.log and correctness logs.')
        write_status(context, 'passed')
    except BaseException as exc:
        write_status(context, 'failed_or_interrupted', reason=str(exc))
        raise
    finally:
        bundle(context)
