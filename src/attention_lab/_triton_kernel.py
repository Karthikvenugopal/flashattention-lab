"""Imported only on explicit NVIDIA execution."""
import triton
import triton.language as tl

@triton.jit
def forward(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
            CAUSAL: tl.constexpr, SCALE: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    base = bh * N * D
    q = tl.load(Q + base + rows[:, None] * D + ds[None, :],
                (rows[:, None] < N) & (ds[None, :] < D), 0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    l = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, BD), 0., tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (tile + 1) * BM)
    for start in range(0, end, BN):
        keys = start + cols
        k = tl.load(K + base + keys[None, :] * D + ds[:, None],
                    (keys[None, :] < N) & (ds[:, None] < D), 0)
        scores = tl.dot(q, k) * SCALE
        allowed = keys[None, :] < N
        if CAUSAL:
            allowed = allowed & (keys[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float('inf'))
        new_m = tl.maximum(m, tl.max(scores, 1))
        p = tl.exp(scores - new_m[:, None])
        alpha = tl.exp(m - new_m)
        v = tl.load(V + base + keys[:, None] * D + ds[None, :],
                    (keys[:, None] < N) & (ds[None, :] < D), 0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, 1)
        m = new_m
    result = acc / l[:, None]
    tl.store(O + base + rows[:, None] * D + ds[None, :], result,
             (rows[:, None] < N) & (ds[None, :] < D))
