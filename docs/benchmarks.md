# Benchmark methodology and verified results

The measurements are attention forward passes. They do not measure full models, backward passes, token generation, or training. The custom fused kernel, readable standard implementation, educational blockwise implementation, and default-dispatch PyTorch SDPA are measured separately.

## Highlight: NVIDIA L4, sequence length 2048

| Field | Exact configuration |
| --- | --- |
| GPU | NVIDIA L4, SM89; driver 580.82.07 |
| GPU memory | `nvidia-smi`: 23034 MiB; CUDA reports 23659151360 bytes (22563.125 MiB). These APIs expose different totals; both original values are retained. |
| Host | Linux x86_64, Python 3.13.16; one PyTorch CPU thread during benchmarks |
| Libraries | torch 2.8.0+cu128, CUDA runtime 12.8, Triton 3.4.0, matplotlib 3.10.3 |
| Q/K/V shape | `[B,H,N,D] = [1,4,2048,64]`, matching contiguous self-attention inputs |
| Dtype / masking | fp16 inputs/output; noncausal, dropout disabled |
| Query / KV tiles | 64 / 32 for blockwise and Triton; ignored by standard and SDPA |
| Timing | Median synchronized `perf_counter` wall milliseconds; 10 repetitions after 3 warmups |
| Included | Python dispatch, allocation within the call, launch and synchronization overhead |
| Excluded | Input generation/transfers, references, correctness checks, profiler, separate memory probes |
| Raw case IDs | Standard 12, educational blockwise 13, SDPA 14, Triton 15 |

| Implementation | Median ms | CUDA peak allocated increment |
| --- | ---: | ---: |
| Standard | 1.9956279999746585 | 137.000 MiB |
| Educational blockwise | 796.4194479999946 | 11.191 MiB |
| SDPA / CUDA Flash Attention | 0.14778500002421424 | 1.032 MiB |
| Fused Triton | 0.18836200001715042 | 1.000 MiB |

The highlighted **0.188 ms** is Triton's median rounded to three decimals. Speedup over standard is `1.9956279999746585 / 0.18836200001715042 = 10.594642230348775`, rounded to **10.6×**. Relative to optimized SDPA, Triton takes `0.18836200001715042 / 0.14778500002421424 = 1.274567784188434` times as long: **27.5% higher latency**, rounded to one decimal. These ratios share the exact same hardware, shape, dtype, masking, and timing statistic above.

Standard attention promotes fp16 inputs to fp32 working tensors and explicitly materializes scores and probabilities. The comparison includes those design choices. SDPA is the optimized performance baseline; operator evidence identifies `aten::_scaled_dot_product_flash_attention` in all 18 NVIDIA configurations. Merely enabling a backend would not prove its execution.

## Sweep results

NVIDIA anchor: `[1,4,512,64]`, query/KV tiles 64/32, fp16. One variable at a time covers sequence length 128/512/1024/2048, head dimension 16/32/64/128, batch 1/2/4, heads 1/2/4/8, independent query and KV tiles 16/32/64, plus causal and odd-length 137 cases. This is **18 configurations × four implementations = 72 cases**, not a Cartesian search. Equal shapes reset the CPU input generator to seed 2205 and share rounded inputs.

Every timed case passed CPU float64 forced-math SDPA using the actual quantized input values (fp16 atol=rtol=0.002). Maximum absolute error across the run was 0.0009743257421575002. The separate original GPU suite passed **17 tests**, including all 16 fp16/bf16 causal/noncausal fused cases, with no skipped tests. Tests exercise odd dimensions/lengths, masked tails, asymmetric blocks, multi-batch/multi-head inputs, first causal output, and future isolation.

For the following rows, all highlight fields above remain fixed except N; timings are median milliseconds:

| N | Standard | Educational blockwise | SDPA | Fused Triton |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.244355 | 3.440948 | 0.072713 | 0.116915 |
| 512 | 0.225786 | 40.798632 | 0.070960 | 0.116335 |
| 1024 | 0.336306 | 197.798859 | 0.096334 | 0.166377 |
| 2048 | 1.995628 | 796.419448 | 0.147785 | 0.188362 |

Triton is faster than standard in all 18 configurations and slower than SDPA in 17. Its one median advantage over SDPA, at `[1,2,512,64]`, is about 4.5%, with overlapping timing ranges (SDPA 0.084–0.162 ms; Triton 0.104–0.176 ms). Ten samples do not establish that small difference as repeatable. Educational blockwise is slower than standard at every sequence length; separate tensor operations and Python loops do not reproduce fused GPU execution.

[Raw NVIDIA data](../results/nvidia-l4/benchmark.json) · [Summary CSV](../results/nvidia-l4/summary.csv) · [720 samples](../results/nvidia-l4/timings.csv) · [Latency plots](../results/nvidia-l4/latency.png) · [Memory plots](../results/nvidia-l4/memory.png)

## CPU and MPS measurements

These earlier runs used Apple M3, Mac15,13, 16 GiB unified memory, arm64 macOS 27.2, Python 3.13.1, torch 2.8.0, matplotlib 3.10.3, and one CPU thread. Some standalone processes could not query the hardware; the MPS-capable process recorded the machine identity. Each device has **57 passed cases / 570 timing samples**, fp32, seed 2205, 3 warmups and 10 repetitions. Anchor `[1,2,128,32]`, query/KV tiles 32/32; the small profile sweeps length 32–512 and the same other axes. They use different inputs/dtypes/anchors from NVIDIA and do not constitute an equivalent cross-device comparison.

CPU SDPA operator: `aten::_scaled_dot_product_flash_attention_for_cpu`. MPS SDPA operator: `aten::_scaled_dot_product_attention_math_for_mps`. Backend evidence is retained per case; an unrecognized operator/profiler failure would be `unknown`. GPU visibility varied by launch context, so each raw report retains its own environment. No MPS CPU fallback was enabled.

[CPU raw data](../results/cpu/benchmark.json) · [CPU latency](../results/cpu/latency.png) · [CPU memory](../results/cpu/memory.png) · [MPS raw data](../results/mps/benchmark.json) · [MPS latency](../results/mps/latency.png) · [MPS memory](../results/mps/memory.png)

## Memory and cold-work semantics

CUDA resets allocator peaks and measures a separate call, reporting peak **allocated** bytes minus the live pre-call baseline. The increment includes the output but excludes reserved/driver memory and does not measure SRAM/register traffic. At the highlighted shape, Triton's one-MiB increment equals the output tensor's size. SDPA workspace and kernel selection can make memory nonmonotonic with shape; the records show observed values.

MPS records **current live allocation after a separate call**, pre-call baseline, and driver allocation. These are **not peak memory**, physical residency, or total unified-memory pressure. CPU allocator peaks were unavailable. Dashed curves show only the analytical size of two dense score buffers, not measured total memory. No peak-memory claim is made for CPU/MPS blockwise attention.

First invocations are saved separately from repeated timings and may include compilation/cache setup. Correctness tests warmed some Triton kernels beforehand; no compiler-only or guaranteed cold-cache duration is claimed. Fixed method order, host scheduling, power/clock state, and ten samples limit inference about small differences. Plot range bars show min/max, not confidence intervals.

## Provenance and revalidation

The released NVIDIA kernel, wrapper, attention implementations, reference, benchmark harness, SDPA detector, device metadata code, and original GPU tests match the recorded Python snapshot byte for byte. The post-run plotting fix selects `Agg` before Matplotlib import; it changes no computation or timing path. The original outer wrapper failed at plotting, after benchmark JSON had reached `passed`. That event is recorded in [import provenance](../results/provenance/import.json); all plots were regenerated locally from saved measurements for release.

`python scripts/audit_results.py` checks source hashes, unchanged raw measurement digests, all 186 cases / 1860 individual samples, CSV/JSON agreement, recomputed timing statistics, original GPU JUnit, and the highlighted ratios. Any change to the recorded computation/test sources fails the audit and requires affected GPU results to be revalidated. No new GPU work is needed to audit or plot the current unchanged source.
