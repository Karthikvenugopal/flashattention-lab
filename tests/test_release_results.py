"""Published claims remain tied to the original measurements and computation."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(str(ROOT / 'scripts/audit_results.py'))


def test_published_claims_and_source_identity():
    report = AUDIT['audit'](ROOT)
    assert report == json.loads((ROOT / 'results/audit.json').read_text())
    assert report['computation_requires_revalidation'] is False


def test_public_results_manifest():
    hashes = json.loads((ROOT / 'results/manifest.json').read_text())
    for path, expected in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_changed_kernel_requires_revalidation(tmp_path):
    # A changed computation must not silently keep the old benchmark claims.
    shutil.copytree(ROOT / 'results/provenance', tmp_path / 'results/provenance')
    shutil.copytree(ROOT / 'src', tmp_path / 'src')
    (tmp_path / 'tests').mkdir()
    shutil.copy2(ROOT / 'tests/test_triton.py', tmp_path / 'tests/test_triton.py')
    kernel = tmp_path / 'src/attention_lab/_triton_kernel.py'
    kernel.write_text(kernel.read_text() + '\n# Changed source invalidates recorded identity.\n')
    with pytest.raises(AssertionError, match='require revalidation'):
        AUDIT['audit'](tmp_path)
