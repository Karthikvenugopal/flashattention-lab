# Project summary for a resume or portfolio

**FlashAttention Forward Lab — PyTorch, Triton, CUDA**

- Implemented a fused Triton FlashAttention forward pass with online softmax, causal masking, and partial-tile support; validated all 16 fp16/bf16 GPU correctness cases against independent references.
- Built reproducible attention benchmarks across sequence length, head dimension, tile size, batch size, and head count; retained 72 passed NVIDIA comparisons and raw timing/memory evidence.
- Measured **0.188 ms median latency** and **10.6× speedup over readable dense PyTorch attention** on NVIDIA L4 for `[B,H,N,D]=[1,4,2048,64]`, fp16, noncausal. Optimized PyTorch SDPA measured **0.148 ms** for the same case.

The timing statements refer to an attention forward pass, not full-model inference. The highlighted medians use 10 synchronized wall timings after 3 warmups; query/KV tiles are 64/32. Both the readable baseline and optimized SDPA comparison belong with the performance claim. [Exact methodology and evidence](benchmarks.md)
