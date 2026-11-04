"""Build a self-contained notebook snapshot from the unchanged local tests/source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination=None):
    paths = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'tests').glob('*.py')),
             ROOT / 'scripts/colab_support.py', ROOT / 'scripts/colab_gpu_worker.py',
             ROOT / 'pyproject.toml', ROOT / 'README.md', ROOT / 'LICENSE']
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    encoded = '\n'.join(f'    {line!r}' for line in textwrap.wrap(base64.b64encode(payload).decode(), 100))
    code = f'''# Standard library only: GPU detection happens before pip or torch imports.
import base64, hashlib, io, zipfile
PAYLOAD = base64.b64decode(\n{encoded}\n)
SOURCE_SHA256 = {hashlib.sha256(payload).hexdigest()!r}
assert hashlib.sha256(PAYLOAD).hexdigest() == SOURCE_SHA256, 'Embedded source checksum mismatch'
with zipfile.ZipFile(io.BytesIO(PAYLOAD)) as archive:
    support_source = archive.read('scripts/colab_support.py')
support = {{}}
exec(compile(support_source, 'colab_support.py', 'exec'), support)
context = support['initialize'](PAYLOAD, SOURCE_SHA256)
support['preflight'](context)
'''
    cells = []

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})

    def python(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': source.splitlines(keepends=True)})

    markdown('''# FlashAttention lab: existing Triton tests → benchmarks

Upload this notebook to [Google Colab](https://colab.research.google.com/), choose **Runtime → Change runtime type → GPU**, then run cells in order. Nothing has been executed remotely in the saved notebook. No Modal dependency or Modal job is used.

The notebook embeds the current source and **unchanged tests**, with SHA-256 checksums; no GitHub URL, upload, Drive mount, or credentials are required. Rebuild after source changes using `python scripts/build_colab_notebook.py`.

**Compatibility is checked before installing dependencies.** The existing kernel requires NVIDIA **SM80+**. A T4 (SM75), P100 (SM60), V100 (SM70), CPU, or TPU runtime stops with an explanation. An A100 or L4 meets the architecture requirement, but Colab does not guarantee a particular GPU or free access. This pinned stack also conservatively requires Linux x86_64, Python 3.10–3.13, and NVIDIA driver **570.26+** for CUDA 12.8. No driver is installed or changed.

Dependencies go into a private virtual environment: torch **2.8.0+cu128**, Triton **3.4.0**, pytest **8.3.4**, matplotlib **3.10.3**. The notebook kernel's packages are left alone. Tests and benchmarks use the isolated interpreter and `PYTHONPATH=src`; the project's original Python 3.13 packaging requirement is preserved. The actual Colab Python version is recorded, so comparisons with the Mac environment must account for that difference.

The numerical tests run first. A successful pytest exit **and all 16 distinct GPU correctness cases passing with zero skips** are required before benchmarks. Any installation failure, incompatible assignment, test failure, missing report, or timeout stops progress. **On failure, run the last download cell manually** to save the explanation and available logs.

References: [Colab resource availability](https://research.google.com/colaboratory/faq.html), [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus), [CUDA 12.8 driver requirements](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/), [PyTorch 2.8 CUDA wheels](https://pytorch.org/get-started/previous-versions/).
''')
    markdown('## 1. Detect the GPU and reject incompatible assignments\n\nThis cell uses only the standard library and `nvidia-smi`. It creates a fresh workspace and saves preflight results even when compatibility fails.')
    python(code)
    markdown('## 2. Install the pinned stack in an isolated environment\n\nThe GPU is checked again before setup. Install logs are saved. Each setup command has a ten-minute timeout; setup can consume Colab runtime time and network bandwidth. No package installation occurs if preflight fails.')
    python("support['install_dependencies'](context)\n")
    markdown('''## 3. Run the existing tests, then the benchmark sweep

Runs the exact embedded `tests/test_triton.py`, comparing fp16/bf16 causal and noncausal outputs against CPU float64 and CUDA math SDPA. Covers partial tiles, odd dimensions, multiple batches/heads, and future isolation. The existing shared correctness gate blocks the benchmark callback unless every GPU case passes.

The `nvidia` benchmark profile sweeps sequence length through 2048, head dimension, independent query/KV tiles, batch size, and head count, plus causal/partial cases. Compares standard, **educational PyTorch blockwise**, default-dispatch SDPA, and **fused Triton**. Three warmups, ten synchronized timings, first-call cost separately, numerical checks for every case. Actual backend evidence is recorded or labeled unknown. The Triton test stage warms kernel caches; benchmark first-call timing does not promise cold compilation.

The worker has a **600-second total timeout**, including a 240-second correctness subprocess limit. On timeout/interruption the worker process group is stopped and available files are bundled. This does not disconnect the Colab runtime or impose a monetary cap. Disconnect/delete the runtime when finished. Results report CUDA allocator peaks, not total GPU/system memory. Every retry gets a separate artifact directory; earlier measurements cannot masquerade as the latest attempt.
''')
    python("support['run_experiment'](context)\n")
    markdown('''## 4. Download all results — also usable after a failure

The ZIP includes the embedded source snapshot and hashes, hardware/driver report, status, installation log, package versions, correctness logs/JUnit, worker log, and any completed/partial benchmark JSON/CSV plus latency/memory PNG/SVG plots. Failed or incompatible runs contain explanations rather than invented measurements. If the browser blocks automatic download, use Colab's Files panel to download the printed ZIP path. Download before the runtime disconnects; local Colab storage is temporary.
''')
    python("if 'context' not in globals():\n    raise RuntimeError('Run the first cell to create a source snapshot and preflight report.')\narchive_path = support['bundle'](context)\nfrom google.colab import files\nfiles.download(archive_path)\n")
    for index, cell in enumerate(cells):
        cell['id'] = f'attention-lab-{index}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': 'attention_lab_colab.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'cells': cells}
    destination = destination or ROOT / 'notebooks/attention_lab_colab.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1) + '\n')
    return destination


if __name__ == '__main__':
    print(build())
