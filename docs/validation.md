# Release validation

## Previously executed NVIDIA evidence

The user-executed Google Colab run on NVIDIA L4 passed **17 tests**, including all 16 fused GPU cases, then **72 benchmark cases** with 720 timing samples. Its original JUnit, dependency/hardware records, and source snapshot remain in `results/`. Release preparation inspected these saved artifacts; it did not execute a new GPU test or benchmark. [Exact measured configuration and claims](benchmarks.md)

## Newly executed local checks

The release environment uses Python 3.13.1 on macOS arm64 with **system site-packages disabled**. PyPI access was unavailable, so original cached compatible wheels were used to install torch 2.8.0, pytest 8.3.4, matplotlib 3.10.3 and their dependencies into a fresh virtual environment. This is a newly isolated installation, not reuse of the workspace's system-packages venv. [Installed dependency versions](validation-environment.txt)

- Editable package installation with the `dev` and `bench` extras, and `pip check`.
- `python -m pytest --cpu-only -q -rs`: **98 passed, 49 skipped**. GPU cases and diagnostic probes were explicitly excluded; skipped tests are **not fresh GPU validation**. Existing standard/blockwise and original Triton tests are preserved; only an optional CPU-only selection hook was added to test configuration.
- The plotting regression starts a separate interpreter with an unavailable inherited inline backend and checks real PNG/SVG outputs.
- All **12 published plots** (latency/memory × PNG/SVG × CPU/MPS/NVIDIA) regenerated from verified saved data, including a run with Colab's inherited `MPLBACKEND` present. No benchmarks or GPU code execute during plot regeneration.
- `python scripts/audit_results.py`: all **186 cases / 1860 samples** reconcile with CSV/JSON, saved measurement digests and recomputed statistics. NVIDIA source identity/JUnit and highlighted ratios verify. A negative test confirms changed kernel source invalidates the recorded results.
- Wheel and source-distribution builds succeeded; the extracted source distribution also passed **98 tests with 49 explicit GPU skips**, its data audit, and its public-file/link/secret-pattern checks. The wheel installed into the isolated environment, and an import from outside the checkout resolved to its `site-packages` copy rather than editable source. The installed `attention-lab` entry point and module CLI help both worked.
- Both tiny-example modes and `python examples/tiny.py` passed from the installed wheel. The CPU `check` command passed all **48 comparisons**. A fresh CPU CLI sweep passed **57 cases** using default 3 warmups/10 repeats, then generated all four plot files. These are newly executed CPU smoke checks, not replacements for the historical published measurements.

## Reproduce the local checks

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,bench]'
python -m pip check
python -m pytest --cpu-only -q -rs
python scripts/audit_results.py
python scripts/check_release.py
python -m attention_lab tiny
python examples/tiny.py
python -m attention_lab check --device cpu --json results/generated/correctness.json
python -m attention_lab benchmark --device cpu --output results/generated/cpu
python -m attention_lab plot results/generated/cpu
```

To intentionally regenerate published plot assets and their checksums:

```bash
python scripts/regenerate_plots.py
python scripts/manifest_results.py
python scripts/check_release.py
```

The original measured kernel and benchmark sources are frozen by the audit. Updating computation requires revalidation of affected results; formatting or metadata changes to benchmark source also fail identity checks until reviewed. The documented plotting change is outside the timed computation path.

## Scope and remaining limits

The local release-preparation checks below did not launch additional GPU/cloud jobs. Publication and hosted CPU CI are separate steps. The optional Modal runner imports locally but its service/image path was not executed. GitHub Actions is configured for CPU validation; hosted CI was not run. Online PyPI resolution and a clean Linux install could not be tested from this network-restricted Mac; isolated macOS installation from cached wheels was tested instead. The original source workspace protects Git metadata. The initial public repository is created from an isolated release checkout; kernel and benchmark source are preserved. A temporary external Git directory confirmed that all intended public files, including curated result evidence, survive the ignore rules; generated outputs and private environment files do not. No credential patterns or machine-specific paths were found in those files or their decoded notebook/provenance source payloads. This is a targeted pattern scan, not a guarantee about unknown secret formats.
