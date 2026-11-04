# FlashAttention Forward Lab

A **forward-pass FlashAttention reproduction** with a fused Triton kernel, causal masking, partial tiles, and reproducible benchmarks. The kernel keeps online-softmax state within each query-tile program and avoids materializing the full attention matrix in global memory.

Verified on an **NVIDIA L4**: **17 correctness tests passed**, including all 16 fp16/bf16 GPU cases, followed by **72 passed benchmark cases**. The released kernel and benchmark code match the measured source snapshot. [Results and methodology](docs/benchmarks.md) · [Release validation](docs/validation.md)

## Measured performance

**NVIDIA L4 (SM89), `[batch, heads, sequence, head_dim] = [1, 4, 2048, 64]`, fp16, noncausal.** Median synchronized wall latency over 10 repetitions after 3 warmups; one CPU thread. Triton query/KV tiles: 64/32. PyTorch 2.8.0+cu128, Triton 3.4.0, driver 580.82.07.

| Forward implementation | Median latency | Comparison |
| --- | ---: | --- |
| Readable standard attention | 1.995628 ms | Materializes dense scores/probabilities; fp32 working tensors |
| Educational PyTorch blockwise | 796.419448 ms | Separate operations for every tile |
| Optimized PyTorch SDPA | **0.147785 ms** | Observed CUDA Flash Attention backend |
| **Fused Triton** | **0.188362 ms** | **10.6× faster than standard; 27.5% slower than SDPA** |

The 10.6× result compares with the readable standard implementation, **not SDPA**. Across the 18 tested configurations, Triton beat standard in all 18 and was slower than SDPA in 17. These are attention forward timings, not full-model inference measurements. [Raw data](results/nvidia-l4/benchmark.json) · [Timing samples](results/nvidia-l4/timings.csv) · [Latency plot](results/nvidia-l4/latency.png)

## Setup

Python **3.13**. From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,bench]'
python -m pytest --cpu-only -q
python -m attention_lab tiny
python examples/tiny.py
```

CPU and Apple MPS work without Triton. For NVIDIA, use Linux x86_64 with an **SM80+ GPU** and a compatible driver, then install the CUDA stack:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,bench,nvidia]'
python -m pytest tests/test_triton.py -q -rs
```

The Colab notebook checks the GPU/driver **before installing packages**, runs the original correctness tests, and only then benchmarks. It embeds the source and downloads results as a ZIP. T4/P100/V100 assignments are rejected because the existing kernel requires SM80+. [Notebook](notebooks/attention_lab_colab.ipynb) · [Colab guide](docs/colab.md)

## Use and reproduce

Q/K/V layout is `[batch, heads, tokens, head_dimension]`. The fused kernel accepts contiguous fp16/bf16 inputs, head dimensions 1–128, and query/KV tiles of 16/32/64. It is forward-only self-attention; backward, dropout, arbitrary masks, and GQA are outside scope.

```python
import torch
from attention_lab.triton_attention import triton_attention

q, k, v = [torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.float16)
           for _ in range(3)]
out = triton_attention(q, k, v, causal=True,
                       query_block_size=64, kv_block_size=32)
```

Readable `standard_attention`, educational `blockwise_attention`, and optimized `sdpa_attention` are exported from `attention_lab`. The educational blockwise implementation explains online softmax; it is distinct from the fused GPU kernel. [Mathematical walkthrough](docs/learning-notes.md)

```bash
# Local CPU sweep and plots (fresh output; does not overwrite published measurements).
python -m attention_lab check --device cpu --json results/generated/correctness-cpu.json
python -m attention_lab benchmark --device cpu --output results/generated/cpu
python -m attention_lab plot results/generated/cpu

# Inspect saved measurements and regenerate all published plots without a GPU.
python scripts/audit_results.py
python scripts/regenerate_plots.py

# On compatible NVIDIA hardware, after correctness succeeds:
python -m attention_lab benchmark --device cuda --dtype float16 --profile nvidia \
  --implementations standard blockwise sdpa triton --output results/generated/nvidia
python -m attention_lab plot results/generated/nvidia
```

The CLI also supports explicit `--device mps`; device unavailability is an error, never a silent CPU fallback. Benchmarks save synchronized samples, separate first-call costs, numerical errors, backend evidence, and memory semantics. CUDA memory is a peak allocated increment; MPS allocations are **current values, not peaks**. [Methodology and all datasets](docs/benchmarks.md)

## Evidence and validation

The repository retains raw CSV/JSON, hardware/dependency metadata, NVIDIA JUnit, measured-source hashes/snapshot, and an executable claim audit. Machine-specific paths and identifiers were removed without changing measurement values. [Artifact inventory](results/README.md)

Local release checks are separate from the previously executed Colab GPU tests. `--cpu-only` explicitly skips GPU checks; it does not constitute new GPU validation. The plotting backend fix is covered by an isolated-process regression test, and all published plots were regenerated from saved data. [Exact checks and remaining limits](docs/validation.md)

Based on [FlashAttention](https://arxiv.org/abs/2205.14135) and the online-softmax design illustrated in [Triton's fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). MIT licensed; see [LICENSE](LICENSE).
