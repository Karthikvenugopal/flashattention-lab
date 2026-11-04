import json

import pytest
import torch

from attention_lab.checks import check
from attention_lab.devices import doctor, resolve_device
from attention_lab.tiny import run_tiny


def test_default_trace_only_last_row_but_checks_every_row(capsys):
    run_tiny()
    output = capsys.readouterr().out
    assert "Query row 2: complete" in output
    assert "Query row 0:" not in output
    assert "Query row 1:" not in output
    assert "All 3 rows" in output
    assert "alpha = exp(m_old-m_new)" in output
    assert "PASS:" in output


def test_optional_rows_print_last_row_first(capsys):
    run_tiny(all_rows=True)
    output = capsys.readouterr().out
    assert output.index("Query row 2: complete") < output.index("Query row 0: complete") < output.index("Query row 1: complete")


def test_explicit_mps_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable in this process"):
        resolve_device("mps")


def test_doctor_identifies_build_separately_from_access():
    report = doctor()
    assert isinstance(report["mps_built"], bool)
    assert isinstance(report["mps_available"], bool)
    assert "mps_operation" in report
    assert "native_metal" in report


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_raw_correctness_report(tmp_path, device, record_testsuite_property):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this execution process")
    path = tmp_path / "correctness.json"
    report = check(device=device, json_path=path)
    saved = json.loads(path.read_text())
    assert saved == report
    assert saved["status"] == "passed"
    assert len(saved["cases"]) == (48 if device == "cpu" else 36)
    assert all(case["status"] == "passed" for case in saved["cases"])
    # Preserve raw results from THIS process in the explicitly requested JUnit
    # artifact, even if a later CLI invocation has different GPU visibility.
    record_testsuite_property(f"correctness_{device}", json.dumps(report))
