"""Deterministic validation, with raw errors saved separately from benchmarks."""

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .attention import blockwise_attention, standard_attention
from .devices import environment, resolve_device
from .reference import TOLERANCES, errors, float64_reference, math_reference


def check(*, device="cpu", json_path=None):
    target = resolve_device(device)
    report = {
        "kind": "correctness (not a benchmark)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(), "device": str(target), "seed": 2205,
        "cases": [], "skips": [],
    }
    generator = torch.Generator().manual_seed(2205)
    failures = 0
    for dtype, (atol, rtol) in TOLERANCES.items():
        if target.type == "mps" and dtype == torch.float64:
            report["skips"].append({"dtype": str(dtype), "reason": "MPS has no float64 support"})
            continue
        # Test each required primitive before classifying this dtype as supported.
        try:
            x = torch.ones((1, 1, 3, 2), dtype=dtype, device=target)
            # Probe the independent library before running our code. A failure
            # in an implementation must not be relabeled as an unsupported dtype.
            math_reference(x, x, x)
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            if target.type != "mps":
                raise
            report["skips"].append({"dtype": str(dtype), "reason": str(exc)})
            continue
        for shape, qb, kb in [((1, 1, 1, 2), 1, 1), ((2, 3, 7, 5), 3, 4), ((1, 2, 17, 8), 5, 3)]:
            inputs = tuple(torch.randn(shape, generator=generator).to(device=target, dtype=dtype) for _ in range(3))
            for causal in (False, True):
                reference = float64_reference(*inputs, causal=causal)
                library = math_reference(*inputs, causal=causal)
                for name, output in (
                    ("standard", standard_attention(*inputs, causal=causal)),
                    ("blockwise", blockwise_attention(*inputs, query_block_size=qb, kv_block_size=kb, causal=causal)),
                ):
                    case = {"implementation": name, "shape": list(shape), "dtype": str(dtype),
                            "causal": causal, "query_block_size": qb, "kv_block_size": kb,
                            "atol": atol, "rtol": rtol, **errors(output, reference),
                            "same_device_reference_errors": errors(output, library)}
                    try:
                        assert bool(torch.isfinite(output).all()), "non-finite output"
                        torch.testing.assert_close(output.cpu().double(), reference, atol=atol, rtol=rtol)
                        torch.testing.assert_close(output, library, atol=atol, rtol=rtol)
                        case["status"] = "passed"
                    except AssertionError as exc:
                        case.update(status="failed", error=str(exc))
                        failures += 1
                    report["cases"].append(case)
    report["status"] = "failed" if failures or not report["cases"] else "passed"
    for dtype in TOLERANCES:
        cases = [c for c in report["cases"] if c["dtype"] == str(dtype)]
        if cases:
            print(f"{dtype}: {len(cases)} cases, max abs={max(c['max_abs_error'] for c in cases):.3e}, max relative L2={max(c['relative_l2_error'] for c in cases):.3e}")
    for skip in report["skips"]:
        print(f"SKIP {skip['dtype']}: {skip['reason']}")
    print(f"{report['status'].upper()}: {len(report['cases'])} cases, {failures} numerical failures")
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Raw correctness results: {path}")
    if report["status"] != "passed":
        raise RuntimeError("Correctness checks failed; inspect the report")
    return report
