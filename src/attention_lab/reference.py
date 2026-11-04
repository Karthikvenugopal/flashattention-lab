"""Independent library reference and explicit numerical error reporting."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


TOLERANCES = {
    torch.float64: (1e-10, 1e-10),
    torch.float32: (1e-5, 1e-4),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


@torch.inference_mode()
def math_reference(q, k, v, *, causal=False):
    """Force math SDPA: never use one of our implementations as the oracle."""
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


@torch.inference_mode()
def float64_reference(q, k, v, *, causal=False):
    # Copy the actual, possibly quantized input values, not an unrounded source.
    return math_reference(*(x.detach().cpu().double() for x in (q, k, v)), causal=causal)


def errors(actual, expected):
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    diff = a - e
    denominator = torch.linalg.vector_norm(e).item()
    numerator = torch.linalg.vector_norm(diff).item()
    return {
        "max_abs_error": diff.abs().max().item(),
        "relative_l2_error": numerator / denominator if denominator else (0.0 if numerator == 0 else float("inf")),
    }
