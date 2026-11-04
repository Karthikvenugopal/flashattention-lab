"""Forward-only self-attention in [batch, heads, tokens, head_dimension] layout.

The tiled implementation reproduces the online-softmax mathematics in Algorithm 1
of https://arxiv.org/abs/2205.14135, not its SRAM placement or GPU performance.
"""

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor


SUPPORTED_DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)


def _validate(q: Tensor, k: Tensor, v: Tensor) -> None:
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q, K, V must have shape [batch, heads, sequence, head_dimension]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("This first increment requires matching Q, K, V shapes (self-attention)")
    if any(size <= 0 for size in q.shape):
        raise ValueError("All dimensions must be positive")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in SUPPORTED_DTYPES:
        raise ValueError("Q, K, V must share a supported floating dtype")
    if q.device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Supported device types are cpu, mps, and cuda")
    if q.device.type == "mps" and q.dtype == torch.float64:
        raise ValueError("MPS does not support float64")


def _working_inputs(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Promote before the dot product too: promoting only softmax cannot repair
    # overflow that has already happened in a low-precision matrix multiply.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    return q.to(dtype), k.to(dtype), v.to(dtype)


@torch.inference_mode()
def standard_attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False) -> Tensor:
    """Materialize BOTH quadratic score and probability matrices for readability.

    Inputs must be finite; their dot products must be representable in the
    accumulation dtype. Arbitrary masks, cross-attention, and gradients are not
    part of this API. No implicit device fallback is performed.
    """
    _validate(q, k, v)
    qw, kw, vw = _working_inputs(q, k, v)
    scores = (qw @ kw.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        allowed = positions[:, None] >= positions[None, :]
        scores.masked_fill_(~allowed, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities @ vw).to(q.dtype)


@dataclass(frozen=True)
class TileTrace:
    """Diagnostic snapshots only; never retained when tracing is disabled."""

    query_start: int
    kv_start: int
    scores: Tensor
    old_max: Tensor
    old_sum: Tensor
    old_accumulator: Tensor
    new_max: Tensor
    rescale: Tensor
    exponentials: Tensor
    value_contributions: Tensor
    new_sum: Tensor
    new_accumulator: Tensor


@torch.inference_mode()
def blockwise_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    query_block_size: int = 32,
    kv_block_size: int = 32,
    causal: bool = False,
    trace_callback: Callable[[TileTrace], None] | None = None,
) -> Tensor:
    """Exact online softmax, with separate PyTorch operations on every tile.

    K/V tiles form the outer loop, as in the original paper. Per-row state is
    m = max(scores), l = sum(exp(scores-m)), a = sum(exp(scores-m) * V).
    The paper stores normalized O; this equivalent formulation stores a = l*O.
    The final O = a/l. Operations and tensors are NOT fused into on-chip SRAM.
    """
    _validate(q, k, v)
    for size in (query_block_size, kv_block_size):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("Block sizes must be positive integers")
    qw, kw, vw = _working_inputs(q, k, v)
    batch, heads, length, dimension = q.shape
    row_shape = (batch, heads, length, 1)
    maxima = torch.full(row_shape, -torch.inf, device=q.device, dtype=qw.dtype)
    sums = torch.zeros(row_shape, device=q.device, dtype=qw.dtype)
    accumulators = torch.zeros_like(qw)

    for ks in range(0, length, kv_block_size):
        ke = min(ks + kv_block_size, length)
        key_tile, value_tile = kw[:, :, ks:ke], vw[:, :, ks:ke]
        for qs in range(0, length, query_block_size):
            qe = min(qs + query_block_size, length)
            if causal and ks >= qe:
                continue  # Entire tile is in the future for every query row.
            scores = (qw[:, :, qs:qe] @ key_tile.transpose(-1, -2)) / math.sqrt(dimension)
            if causal:
                q_positions = torch.arange(qs, qe, device=q.device)
                k_positions = torch.arange(ks, ke, device=q.device)
                scores.masked_fill_(k_positions[None, :] > q_positions[:, None], -torch.inf)

            old_m = maxima[:, :, qs:qe]
            old_l = sums[:, :, qs:qe]
            old_a = accumulators[:, :, qs:qe]
            new_m = torch.maximum(old_m, scores.amax(dim=-1, keepdim=True))
            # An all-masked, not-yet-initialized row has m=-inf. Subtract zero
            # there, avoiding (-inf)-(-inf); its exponentials and alpha are zero.
            safe_m = torch.where(torch.isneginf(new_m), torch.zeros_like(new_m), new_m)
            alpha = torch.exp(old_m - safe_m)
            p = torch.exp(scores - safe_m)
            new_l = alpha * old_l + p.sum(dim=-1, keepdim=True)
            new_a = alpha * old_a + p @ value_tile

            if trace_callback is not None:
                trace_callback(TileTrace(
                    query_start=qs, kv_start=ks,
                    scores=scores.detach().cpu().clone(),
                    old_max=old_m.detach().cpu().clone(),
                    old_sum=old_l.detach().cpu().clone(),
                    old_accumulator=old_a.detach().cpu().clone(),
                    new_max=new_m.detach().cpu().clone(),
                    rescale=alpha.detach().cpu().clone(),
                    exponentials=p.detach().cpu().clone(),
                    value_contributions=(p.unsqueeze(-1) * value_tile.unsqueeze(-3)).detach().cpu().clone(),
                    new_sum=new_l.detach().cpu().clone(),
                    new_accumulator=new_a.detach().cpu().clone(),
                ))
            maxima[:, :, qs:qe] = new_m
            sums[:, :, qs:qe] = new_l
            accumulators[:, :, qs:qe] = new_a

    # Every row has at least its first key, or its own key in causal mode.
    return (accumulators / sums).to(q.dtype)
