# Google Colab

Upload [the notebook](../notebooks/attention_lab_colab.ipynb) at [Google Colab](https://colab.research.google.com/). Choose **Runtime → Change runtime type → GPU**, then run cells in order. Select L4 or A100 when available. Colab does not guarantee an assignment or free access. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

The notebook embeds a checksummed source/test snapshot and requires no GitHub URL, Drive mount, credentials, or Modal. Dependencies are installed into an isolated environment; notebook-kernel packages are unchanged. Results download as a ZIP. Disconnect/delete the runtime when finished.

## Compatibility and execution order

1. Standard-library code and `nvidia-smi` identify the assigned GPU **before installing dependencies**. The current kernel requires SM80+; T4/SM75, P100/SM60, V100/SM70, CPU, and TPU assignments stop with an explanation. The pinned CUDA 12.8 setup conservatively requires Linux x86_64, Python 3.10–3.13, and driver 570.26+. No driver is installed or changed.
2. Install torch 2.8.0+cu128, Triton 3.4.0, pytest 8.3.4, and matplotlib 3.10.3 in a private venv. Pinned pip bootstrapping avoids Debian `ensurepip` availability issues. Each setup command has a 600-second timeout.
3. Run the unchanged `tests/test_triton.py`. A zero exit code, valid JUnit, and all 16 distinct GPU correctness cases passing with no skips/errors/failures are required before the benchmark callback can run. The guard test adds a seventeenth test. Correctness timeout: 240 seconds.
4. Benchmark all four implementations with the existing NVIDIA profile, then generate plots. Total worker timeout: 600 seconds, including correctness and plots. Interruption/timeout kills the process group and bundles available artifacts when the parent remains alive. These bounds do not disconnect Colab or impose a monetary cap.
5. Download the ZIP. Run the last download cell manually after an earlier failure. The Files panel provides an alternative if the browser blocks automatic download. Local Colab storage is temporary.

The notebook imports source through `PYTHONPATH`, allowing its listed runtime Python versions without changing the project's Python 3.13 package requirement. Actual Python/dependencies are recorded. Each worker attempt has a separate directory. See [methodology](benchmarks.md) for timing and memory semantics.

[NVIDIA capability table](https://developer.nvidia.com/cuda/gpus) · [CUDA 12.8 driver requirements](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/) · [PyTorch installation variants](https://pytorch.org/get-started/previous-versions/)

## Snapshot maintenance and verified execution

After changing source/tests, rebuild before distributing the notebook:

```bash
python scripts/build_colab_notebook.py
python -m pytest --cpu-only tests/test_colab_notebook.py tests/test_cloud_checks.py -q
```

The notebook is a self-contained release snapshot, not a link to live repository code. Existing tests verify embedded source bytes/checksum and reject incompatible simulated hardware before setup. The original user-executed L4 run passed 17 tests and 72 benchmark cases; its raw evidence is retained separately. [Measured results](benchmarks.md)

## Recover a completed run's plots

The plotting code now forces the `Agg` file backend **before importing Matplotlib**, overriding Colab's inherited `MPLBACKEND=module://matplotlib_inline.backend_inline`. The worker environment also sets `MPLBACKEND=Agg`. An isolated-process regression test covers an unavailable inherited backend.

If an older notebook reached plotting and then failed, regenerate from the saved JSON; do not repeat GPU work. In that same runtime, with its existing `context`:

```python
import json, os, subprocess
from pathlib import Path
from google.colab import files

root = Path(context['root'])
bench = Path(context['last_attempt']) / 'benchmark'
report = json.loads((bench / 'benchmark.json').read_text())
print('Benchmark status:', report['status'])
env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': str(root / 'src'),
       'MPLCONFIGDIR': str(root / '.mpl')}
subprocess.run([context['python'], '-m', 'attention_lab', 'plot', str(bench)],
               env=env, check=True, timeout=120)
files.download(support['bundle'](context))
```

The ZIP retains the original wrapper failure alongside completed measurements; plot recovery does not rewrite it as a newly successful GPU run. If the runtime is gone, use the earlier downloaded archive's JSON with the local plotting command instead.

## Optional Modal runner

`scripts/modal_runner.py` remains an alternative bounded runner, not a requirement for Colab or the published results. Install `.[cloud]` and authenticate only if choosing Modal. It gates benchmarks on the same correctness checks and uses one L4, at most one single-use container, no configured retries, two CPU cores and eight GiB host memory maximum, 600-second execution and separate 600-second startup timeouts. Build/queue time is outside those limits. It downloads returned artifacts locally. No Modal job has been launched or is required for release; any future cloud run needs explicit authorization and a fresh account/pricing check.
