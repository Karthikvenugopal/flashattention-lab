"""Production PyTorch dispatch baseline; backend evidence is collected separately."""
import torch
import torch.nn.functional as F
from .attention import _validate


@torch.inference_mode()
def sdpa_attention(q, k, v, *, causal=False):
    _validate(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def identify_backend(fn):
    """Observe dispatched ATen operators, never infer from hardware or flags."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            fn()
        events = sorted({e.key for e in prof.key_averages() if 'attention' in e.key})
        known = {
            'aten::_scaled_dot_product_flash_attention_for_cpu': 'flash_attention_for_cpu',
            'aten::_scaled_dot_product_flash_attention': 'flash_attention_cuda',
            'aten::_scaled_dot_product_efficient_attention': 'efficient_attention',
            'aten::_scaled_dot_product_cudnn_attention': 'cudnn_attention',
            'aten::_scaled_dot_product_attention_math': 'math',
            'aten::_scaled_dot_product_attention_math_for_mps': 'math_for_mps',
        }
        matches = [value for key, value in known.items() if key in events]
        return {'backend': matches[0] if len(matches) == 1 else 'unknown', 'operators': events}
    except (RuntimeError, NotImplementedError) as exc:
        return {'backend': 'unknown', 'error': str(exc)}
