# Published measurement artifacts

| Directory | Device / dtype | Cases | Timing samples |
| --- | --- | ---: | ---: |
| `cpu/` | Apple M3 CPU / fp32 | 57 | 570 |
| `mps/` | Apple M3 MPS / fp32 | 57 | 570 |
| `nvidia-l4/` | NVIDIA L4 CUDA / fp16 | 72 | 720 |

Each directory contains raw `benchmark.json`, `summary.csv`, `timings.csv`, and regenerated latency/memory PNG/SVG plots. JSON retains hardware/software, seeds, configurations, warmups/repetitions, first calls, all samples, numerical errors/tolerances, backend evidence, and memory API semantics. `case_id` joins CSV rows to JSON. CPU/MPS also retain their original independent correctness reports.

NVIDIA adds hardware/driver metadata, pre-benchmark environment, dependency versions, and original `correctness.xml` (17 passed tests). The pre-benchmark environment had six CPU threads; the benchmark explicitly set one. `nvidia-smi` and CUDA reported different memory totals; neither was replaced with the other. `mac-dependencies.txt` is a focused subset of the contemporaneous Mac package record, not a complete dependency lock.

`provenance/nvidia-source.zip` contains the unchanged measured Python source/tests/helpers, not executable release instructions. `nvidia-source-sha256.json` verifies every included file. Historical plotting code is retained only as provenance; use the current plotting module. `mac-source-sha256.json` records the original Mac-stage source fingerprints. `provenance/import.json` records original input hashes, the plotting-only wrapper failure, and public-release transformations. Absolute interpreter paths, GPU UUID, and JUnit hostname were removed; **no measurement values changed**. Private wrapper/install/debug logs and duplicate reports are not part of this release.

`audit.json` is derived from the raw evidence using `python scripts/audit_results.py --write`. `manifest.json` hashes every released results file; `python scripts/manifest_results.py` updates it after intentional plot regeneration. `python scripts/audit_results.py` checks claims and computation identity without running a GPU.

These curated directories are intentionally included in Git. New benchmark runs should use `results/generated/`, which is ignored. Regenerating plots never reruns benchmarks. See [methodology](../docs/benchmarks.md) and [release validation](../docs/validation.md).
