"""Offline validation of the notebook's pre-install checks and source snapshot."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from unittest.mock import Mock
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = runpy.run_path(str(ROOT / 'scripts/colab_support.py'))


def notebook_payload():
    notebook = json.loads((ROOT / 'notebooks/attention_lab_colab.ipynb').read_text())
    source = ''.join(next(c for c in notebook['cells'] if c['cell_type'] == 'code')['source'])
    assignments = {node.targets[0].id: node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)}
    payload = base64.b64decode(ast.literal_eval(assignments['PAYLOAD'].args[0]))
    return notebook, payload, ast.literal_eval(assignments['SOURCE_SHA256'])


def test_notebook_compiles_and_embeds_exact_existing_tests_and_sources():
    notebook, payload, checksum = notebook_payload()
    assert hashlib.sha256(payload).hexdigest() == checksum
    assert notebook['nbformat'] == 4
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert not cell['outputs'] and cell['execution_count'] is None
            compile(''.join(cell['source']), '<notebook-cell>', 'exec')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for path in [*ROOT.joinpath('tests').glob('*.py'), *ROOT.joinpath('src').rglob('*.py')]:
            assert archive.read(str(path.relative_to(ROOT))) == path.read_bytes()
        assert 'scripts/modal_runner.py' not in archive.namelist()


@pytest.mark.parametrize('name,cap,driver,error', [
    ('Tesla T4', '7.5', '580.1', 'SM75'),
    ('Tesla P100', '6.0', '580.1', 'SM60'),
    ('Tesla V100', '7.0', '580.1', 'SM70'),
    ('NVIDIA A100', '8.0', '550.54', 'driver'),
    ('Unknown', 'N/A', '580.1', 'Unknown compute capability'),
])
def test_incompatible_hardware_stops_before_install_and_preserves_reason(tmp_path, monkeypatch, name, cap, driver, error):
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    probe = Mock(return_value=subprocess.CompletedProcess([], 0, f'0, GPU-test, {name}, {cap}, 16384, {driver}\n', ''))
    monkeypatch.setattr(subprocess, 'run', probe)
    context = {'root': str(tmp_path), 'output': str(tmp_path / 'results'), 'preflight_passed': False}
    Path(context['output']).mkdir()
    with pytest.raises(RuntimeError, match=error):
        SUPPORT['preflight'](context)
    with pytest.raises(RuntimeError, match='preflight must pass'):
        SUPPORT['install_dependencies'](context)
    with pytest.raises(RuntimeError, match='must pass'):
        SUPPORT['run_experiment'](context)
    assert probe.call_count == 1
    assert probe.call_args.args[0][0] == 'nvidia-smi'
    assert (tmp_path / 'attention-lab-results.zip').exists()
    assert json.loads((Path(context['output']) / 'status.json').read_text())['status'] == 'incompatible_or_unavailable'


def test_compatible_gpu_preflight_needs_no_torch_or_package_install(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    probe = Mock(return_value=subprocess.CompletedProcess([], 0, '0, GPU-test, NVIDIA L4, 8.9, 23034, 580.95.05\n', ''))
    monkeypatch.setattr(subprocess, 'run', probe)
    context = {'root': str(tmp_path), 'output': str(tmp_path)}
    info = SUPPORT['preflight'](context)
    assert info['gpu_name'] == 'NVIDIA L4'
    assert context['preflight_passed'] and not context['installed']
    assert probe.call_count == 1


def test_no_gpu_query_binary_preserves_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(subprocess, 'run', Mock(side_effect=FileNotFoundError('nvidia-smi unavailable')))
    context = {'root': str(tmp_path), 'output': str(tmp_path / 'results')}
    Path(context['output']).mkdir()
    with pytest.raises(RuntimeError, match='nvidia-smi unavailable'):
        SUPPORT['preflight'](context)
    assert (tmp_path / 'attention-lab-results.zip').exists()


def test_embedded_notebook_stops_downstream_cells_on_t4(tmp_path, monkeypatch):
    import tempfile
    notebook, _, _ = notebook_payload()
    code_cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kwargs: str(runtime))
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    probe = Mock(return_value=subprocess.CompletedProcess([], 0, '0, GPU-test, Tesla T4, 7.5, 15360, 580.95\n', ''))
    monkeypatch.setattr(subprocess, 'run', probe)
    namespace = {}
    with pytest.raises(RuntimeError, match='SM75'):
        exec(compile(code_cells[0], '<notebook-preflight>', 'exec'), namespace)
    for cell in code_cells[1:3]:
        with pytest.raises(RuntimeError, match='must pass'):
            exec(compile(cell, '<notebook-downstream>', 'exec'), namespace)
    assert probe.call_count == 1
    assert next(runtime.rglob('attention-lab-results.zip')).exists()
