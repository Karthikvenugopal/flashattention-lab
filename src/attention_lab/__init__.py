"""Readable attention algorithms; educational tiling is not a fused GPU kernel."""

from .attention import blockwise_attention, standard_attention

__all__ = ["standard_attention", "blockwise_attention", "sdpa_attention"]

from .sdpa import sdpa_attention
