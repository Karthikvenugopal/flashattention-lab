"""Regression: notebook-inherited backend must not break isolated plot imports."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_plots_override_inherited_unavailable_inline_backend(tmp_path):
    report = {'device': 'cpu', 'dtype': 'float32', 'cases': [{
        'status': 'passed', 'implementation': 'standard',
        'config': {'sweep': 'anchor', 'length': 17, 'dimension': 8, 'batch': 1,
                   'heads': 2, 'query_block_size': 16, 'kv_block_size': 16},
        'median_ms': 1., 'min_ms': .8, 'max_ms': 1.2,
        'memory': {'kind': 'unavailable', 'bytes': None},
        'analytical_two_dense_score_buffers_bytes': 4624}]}
    (tmp_path / 'benchmark.json').write_text(json.dumps(report))
    code = '''import sys
from pathlib import Path
from attention_lab.plots import plot_results
paths = plot_results(sys.argv[1])
import matplotlib
assert matplotlib.get_backend().lower() == 'agg'
assert len(paths) == 4
assert all(Path(p).stat().st_size > 100 for p in paths)
'''
    env = {**os.environ, 'MPLBACKEND': 'module://missing_colab_inline_backend.backend_inline',
           'MPLCONFIGDIR': str(tmp_path / 'mpl'), 'XDG_CACHE_HOME': str(tmp_path / 'cache'),
           'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'latency.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (tmp_path / 'memory.svg').read_text()
