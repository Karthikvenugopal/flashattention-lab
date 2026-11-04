"""Experimental fused forward, NVIDIA SM80+, fp16/bf16. No backward.

One program owns a query tile; online max/sum/output stay in local kernel
state across K/V tiles. No N x N tensor is written to global memory.
"""
import torch
from .attention import _validate

# Lazy import keeps CPU/MPS installs independent of Triton.
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    from ._triton_kernel import forward
    _kernel = forward
    return forward


@torch.inference_mode()
def triton_attention(q, k, v, *, causal=False, query_block_size=32, kv_block_size=32):
    _validate(q, k, v)
    if q.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('Triton forward requires NVIDIA CUDA')
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError('Triton forward requires SM80 or newer')
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Triton forward supports float16/bfloat16')
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError('Triton forward requires contiguous Q/K/V')
    if q.shape[-1] > 128:
        raise ValueError('Triton forward supports head dimensions 1..128')
    if any(type(b) is not int or b not in (16, 32, 64) for b in (query_block_size, kv_block_size)):
        raise ValueError('Triton block sizes must be 16, 32, or 64')
    import triton
    out = torch.empty_like(q)
    b, h, n, d = q.shape
    with torch.cuda.device(q.device):
        _get_kernel()[(triton.cdiv(n, query_block_size), b * h)](
            q, k, v, out, n, d, query_block_size, kv_block_size,
            max(16, triton.next_power_of_2(d)), causal, d ** -0.5, num_warps=4)
    return out
