"""A complete numerical trace, one query row at a time."""

import math

import torch

from .attention import TileTrace, blockwise_attention, standard_attention
from .devices import resolve_device
from .reference import TOLERANCES, errors, float64_reference


def tiny_inputs(device="cpu", dtype=torch.float64):
    return tuple(torch.tensor(values, device=device, dtype=dtype)[None, None] for values in (
        [[1, 0], [0, 1], [1, 1]],
        [[0, 0], [1, 0], [3, 1]],
        [[1, 0], [0, 2], [3, 1]],
    ))


def number(x):
    return f"{float(x):.6f}"


def vector(x):
    return "[" + ", ".join(number(v) for v in x) + "]"


def run_tiny(*, device="cpu", all_rows=False):
    target = resolve_device(device)
    dtype = torch.float32 if target.type == "mps" else torch.float64
    q, k, v = tiny_inputs(target, dtype)
    print(f"Tiny self-attention: device={target}, dtype={dtype}; row indices start at 0.")
    print("Tokenization -> embeddings -> learned Q/K/V projections -> attention.")
    print("Here Q/K/V are illustrative inputs, not trained word representations.")
    for name, x in (("Q", q), ("K", k), ("V", v)):
        print(f"{name} = {x[0, 0].cpu().tolist()}")
    print("A query compares with keys; softmax weights mix the value vectors.")
    rows = [2, 0, 1] if all_rows else [2]
    qc, kc, vc = (x[0, 0].cpu().double() for x in (q, k, v))
    ordinary = standard_attention(q, k, v)

    for row in rows:
        print(f"\n=== Query row {row}: complete ordinary calculation ===")
        scores = []
        for j in range(3):
            raw = (qc[row] * kc[j]).sum().item()
            score = raw / math.sqrt(2)
            scores.append(score)
            products = " + ".join(f"{number(qc[row, d])}*{number(kc[j, d])}" for d in range(2))
            print(f"key {j}: dot = {products} = {number(raw)}; score = {number(raw)}/sqrt(2) = {number(score)}")
        maximum = max(scores)
        exponentials = [math.exp(s - maximum) for s in scores]
        total = sum(exponentials)
        print(f"row maximum = {number(maximum)} (subtracting it preserves softmax ratios)")
        for j, (score, exponential) in enumerate(zip(scores, exponentials)):
            print(f"key {j}: exp({number(score)} - {number(maximum)}) = {number(exponential)}")
        print(f"normalization sum = {' + '.join(number(e) for e in exponentials)} = {number(total)}")
        manual = torch.zeros(2, dtype=torch.float64)
        for j, exponential in enumerate(exponentials):
            weight = exponential / total
            contribution = weight * vc[j]
            manual += contribution
            print(f"key {j}: weight = {number(exponential)}/{number(total)} = {number(weight)}; weight*V[{j}] = {number(weight)}*{vector(vc[j])} = {vector(contribution)}")
        print(f"output[{row}] = sum of those contributions = {vector(manual)}")
        atol, rtol = TOLERANCES[dtype]
        torch.testing.assert_close(manual, ordinary[0, 0, row].cpu().double(), atol=atol, rtol=rtol)

        print(f"\n=== Query row {row}: same calculation in KV blocks of 2 ===")
        print("Start: m=-inf, l=0, a=[0, 0]. a is an UNNORMALIZED weighted-value sum.")

        def trace(tile: TileTrace):
            if tile.query_start != row:
                return
            old_m, new_m = tile.old_max[0, 0, 0, 0], tile.new_max[0, 0, 0, 0]
            alpha = tile.rescale[0, 0, 0, 0]
            print(f"KV block {tile.kv_start}:{tile.kv_start + tile.scores.shape[-1]}: scaled scores = {vector(tile.scores[0, 0, 0])}")
            print(f"  m_new = max({number(old_m)}, {number(tile.scores.max())}) = {number(new_m)}")
            print(f"  alpha = exp(m_old-m_new) = exp({number(old_m)}-{number(new_m)}) = {number(alpha)}")
            p = tile.exponentials[0, 0, 0]
            for j, exponential in enumerate(p):
                key = tile.kv_start + j
                print(f"  key {key}: p=exp({number(tile.scores[0, 0, 0, j])}-{number(new_m)})={number(exponential)}; p*V={vector(tile.value_contributions[0, 0, 0, j])}")
            old_l = tile.old_sum[0, 0, 0, 0]
            old_a = tile.old_accumulator[0, 0, 0]
            added_a = tile.value_contributions[0, 0, 0].sum(dim=0)
            print(f"  l_new = {number(alpha)}*{number(old_l)} + {number(p.sum())} = {number(tile.new_sum[0, 0, 0, 0])}")
            print(f"  a_new = {number(alpha)}*{vector(old_a)} + {vector(added_a)} = {vector(tile.new_accumulator[0, 0, 0])}")
            if tile.kv_start == 2:
                print(f"  final output = a/l = {vector(tile.new_accumulator[0, 0, 0])}/{number(tile.new_sum[0, 0, 0, 0])}")

        blocked = blockwise_attention(q, k, v, query_block_size=1, kv_block_size=2, trace_callback=trace)
        print(f"blockwise output[{row}] = {vector(blocked[0, 0, row].cpu())}")

    reference = float64_reference(q, k, v)
    for name, output in (("ordinary", ordinary), ("blockwise", blocked)):
        torch.testing.assert_close(output.cpu().double(), reference, atol=atol, rtol=rtol)
        print(f"\nAll 3 rows vs independent CPU float64 SDPA ({name}): {errors(output, reference)}")
    torch.testing.assert_close(ordinary, blocked, atol=atol, rtol=rtol)
    print("PASS: ordinary, blockwise, and independent reference agree on every row.")
    if not all_rows:
        print("Use --all-rows to print rows 0 and 1 too, after row 2.")
    print("Learning checkpoint: pause here before benchmarking, Triton, or cloud execution.")
