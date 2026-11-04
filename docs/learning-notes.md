# Attention and online softmax

## 1. What enters attention?

In a language model, a tokenizer maps text to token IDs. An embedding table maps IDs to vectors. Learned matrices project those vectors into queries, keys, and values: `Q=XWq`, `K=XWk`, `V=XWv`. Our experiment starts after those projections, using fixed numbers. No tokenizer, embedding training, or projection training is involved.

A query asks which keys are relevant. Values contain the vectors we will mix. A head performs this operation independently of the other heads.

## 2. What is a score?

For the last query `[1,1]` and last key `[3,1]`, the dot product is `1*3 + 1*1 = 4`. With head dimension 2, its scaled score is `4/sqrt(2) = 2.828427`.

Dot-product variance tends to grow with dimension under common assumptions about the inputs. Scaling by `1/sqrt(d)` helps keep scores from growing solely because there are more components. We use the conventional scaled-dot-product operation throughout; this scale can be absorbed into Q when connecting it to Algorithm 1's notation.

## 3. How do scores become a mixture?

Our last row has scaled scores `[0, 0.707107, 2.828427]`. Softmax exponentiates scores and divides by their sum. Subtracting the same maximum from all scores gives the same mathematical weights while avoiding large positive exponentials.

The resulting weights are about `[0.050133, 0.101675, 0.848192]`. The first contribution is `0.050133*[1,0]`, the second is `0.101675*[0,2]`, and the third is `0.848192*[3,1]`. Add them to obtain `[2.594708,1.051542]`. The default example prints each step.

## 4. How can we combine blocks without keeping every weight?

For one row, maintain:

```text
m = largest score seen
l = sum(exp(score - m))
a = sum(exp(score - m) * value)
output = a / l
```

After the first two keys, `m=0.707107`, `l=1.493069`, and `a=[0.493069,2]`. The final key has score `2.828427`, so the maximum increases.

To express the old contributions relative to the new maximum, multiply them by:

```text
alpha = exp(0.707107 - 2.828427) = 0.119873
```

This follows from `exp(s-m_new) = exp(s-m_old)*exp(m_old-m_new)`. Apply the same factor to both `l` and `a`, then add the new key's contribution. Now `l=1.178979`, `a=[3.059106,1.239747]`, and `a/l` gives the same output. Normalizing each block separately and adding its outputs would give the wrong result.

These running exponentials are not final softmax probabilities: a later block can change their common scale and denominator.

## 5. How does this connect to the paper?

[Algorithm 1 of the paper](https://arxiv.org/pdf/2205.14135) uses tiled scores and online normalization. It stores a normalized output `O`; we store the equivalent unnormalized accumulator `a=l*O`, then divide once at the end. Both retain the key/value outer-loop order. Explicit block sizes replace the paper's hardware-SRAM-capacity-based choices in this teaching code.

The standard implementation stores two tensors of shape `[B,H,N,N]`: scores and probabilities. The teaching implementation keeps row statistics, an accumulator shaped like the output, and temporary score tiles of at most `[B,H,Bq,Bkv]`. It may also promote entire inputs and allocate multiple tile temporaries. The diagnostic callback additionally copies tensors to CPU; never enable it for performance measurement.

Both still do quadratic attention arithmetic. The original GPU algorithm reduces traffic between large GPU memory and small, fast on-chip memory through fusion and controlled tiling. Separate PyTorch operations cannot claim the same placement or IO savings. On the M3, unified system memory changes the physical memory setting further. The benchmark results measure these implementations separately.

## 6. What does a causal mask mean?

For query position `i`, only keys `j <= i` are allowed. Future scores become `-inf`, so their exponentials are zero. The first query has only one allowed value, so its output is exactly `V[0]`. Global positions matter: restarting positions at zero inside each tile would break the mask. Tests change future keys/values and check that earlier outputs remain fixed.

The fused Triton kernel applies the same running-state mathematics inside one GPU program per query tile. See the benchmark methodology for the measured execution differences.
