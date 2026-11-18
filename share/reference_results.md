# Reference Results

> Back to [public report](public_report.md) · [readme](../readme.md). See also [experiment plan](experiment_plan.md) · [conclusions](conclusion.md) · [guide](../guide.md)

Numbers produced by this repository, for external comparison. All values are aggregated by scripts from raw `jsonl` records; none are hand-entered.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6140.5 MB |
| Driver | 566.24 |
| Clock | Not locked. `nvidia-smi -lgc` was refused for lack of permission; measured SM clock ≈ 2310 MHz and drifts with load. Architecture comparisons therefore use the same-process alternating measurements of `eval_paired.py` and report only paired deltas. |
| Python / torch | 3.13 / 2.10.0+cu126 |
| Precision | BF16 autocast |
| Batch size | 1 for layer benchmarks, 8 × grad-accum 2 for training |

## Model-level training

Common skeleton: `d_model=768`, `n_layers=8`, `n_heads=8`, `head_dim=96`, SwiGLU hidden 2432, vocab 6400, train length 512, 150,000,000 tokens, 18,310 steps of 8,192 tokens.

| Skeleton | Params | Val PPL (42 / 43) | Mean | Test PPL (42 / 43) | Wall (h) | Throughput (tok/s) | Peak mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MHA | 68,630,784 | 9.1557 / 9.1246 | 9.1402 | 8.4975 / 8.5976 | 2.434 / 2.437 | 17,129 | 1553.0 |
| GQA-4 | 63,912,192 | 9.3272 / 9.2341 | 9.2807 | 8.5868 / 8.5361 | 2.163 / 2.163 | 19,275 | 1467.2 |
| MLA | 68,925,696 | 10.5606 / 10.3130 | 10.4368 | 9.7085 / 9.4853 | 2.734 / 2.733 | 15,247 | 1571.0 |
| MLA, QK-Norm disabled (retired) | 68,924,160 | 10.5707 / 10.4115 | 10.4911 | 9.7535 / 9.6315 | 2.195 / 2.286 | 18,992 / 18,294 | 1559.2 |

The MLA row is the QK-Norm variant, which is the one this project reports as MLA; the no-QK-Norm variant is kept only as a reference for the absorbed inference path and its weights were deleted. Seed spread runs from 0.03 (MHA) to 0.25 (MLA) perplexity, so the MLA-vs-MHA gap (~1.35) is repeatable while the GQA-4-vs-MHA gap (~0.14) is not distinguishable from noise. QK-Norm moves perplexity by 0.054, also below the spread.

## Model-level inference

Sequence length 512, 20 samples, 64 decode steps each.

| Skeleton | First token (ms) | Per token (ms) | Throughput (tok/s) | Peak mem (MB) |
| --- | --- | --- | --- | --- |
| MHA | 14.377 / 14.644 | 9.760 / 10.348 | 854.6 / 800.9 | 347.93 |
| GQA-4 | 15.894 / 15.663 | 10.782 / 10.522 | 777.3 / 802.2 | 302.96 |
| MLA (naive path) | 24.754 / 22.043 | 23.865 / 21.449 | 355.2 / 381.6 | 308.49 |
| MLA, absorbed path (retired weights) | 20.579 / 19.074 | 10.384 / 9.783 | 854.2 / 783.4 | 324.23 |

## KV cache footprint

Model-level, `head_dim=96`:

| Skeleton | Elements per token per layer | Bytes (BF16) | Ratio vs MLA |
| --- | --- | --- | --- |
| MHA | 1536 | 3072 | 3.556× |
| GQA-4 | 768 | 1536 | 1.778× |
| MLA | 432 | 864 | 1.000× |

Layer-level, `head_dim=64`, `d_c=256`, `d_rope=32`: MHA 1024 elements, MLA 288 elements, ratio 3.556×. Cache is identical for naive and absorbed MLA; only the decode compute path differs.

Crossover with weight size: at 130.9 MiB (MHA) and 131.5 MiB (MLA) of weights, the cache exceeds the weights past roughly 5.6K tokens (MHA) and 19.9K tokens (MLA).

## Layer-level decode latency

Single layer, batch 1, BF16, `d_c=256`, `n_heads=8`, median of 30 rounds, milliseconds.

| Context | MHA | Naive MLA | Absorbed MLA | Absorbed / naive |
| --- | --- | --- | --- | --- |
| 2,112 | 1.864 | 2.446 | 3.197 | 1.307 |
| 16,384 | 1.532 | 2.212 | 2.964 | 1.340 |
| 32,768 | 2.573 | 3.786 | 5.115 | 1.351 |

## Layer-level prefill latency

Same layer configuration, chunked prefill with 1024-token blocks, milliseconds.

| Context | MHA | Naive MLA | Absorbed MLA | Absorbed / naive |
| --- | --- | --- | --- | --- |
| 16,384 | 49.81 | 85.85 | 187.05 | 2.179 |
| 32,768 | 121.54 | 213.52 | 502.34 | 2.353 |

## `d_c` sweep

`n_heads=8` fixed, ratio is absorbed / naive decode latency.

| `d_c` | Key width | Compression vs MHA | 16K | 32K |
| --- | --- | --- | --- | --- |
| 384 | 416 | 2.46× | 1.602 | 1.686 |
| 256 | 288 | 3.56× | 1.340 | 1.351 |
| 192 | 224 | 4.57× | 1.235 | 1.246 |
| 128 | 160 | 6.40× | 1.158 | 0.965 |

Absorption breaks even at `d_c < (4·n_h·head_dim − n_h·d_rope)/(2·n_h − 1)`, which evaluates to 179.2 for `n_h=8, head_dim=64, d_rope=32`. The measured crossing sits between 128 and 192, consistent with the criterion in order of magnitude.

## `n_heads` sweep

`d_c=256` fixed.

| `n_heads` | 16K | 32K |
| --- | --- | --- |
| 8 | 1.340 | 1.351 |
| 16 | 1.421 | 1.404 |
| 32 | 1.508 | invalid |

## Invalid measurement points

| Point | Observed | Cause |
| --- | --- | --- |
| `d_c=171` @ 32K | Absorbed 150.78 ms, ratio 38.08 | Key width 203 is not a multiple of 8; SDPA falls back to a slow path |
| `n_heads=32` @ 32K | Absorbed 198.79 ms | Same class of kernel degradation at large key width |

Both are excluded from figures; `scripts/layer/plot_layer.py` drops configurations whose key width is not 8-aligned and states the exclusion in the figure title.

## Known deviations from DeepSeek-V2

- The query-side low-rank branch (`W^DQ` followed by `W^UQ` and `W^QR`) is not implemented. Queries are produced by full-rank `w_q` and `w_qr`, so the query parameter count is larger than the paper's structure. KV-side ratios match the paper: `d_c/head_dim = 4`, `d_rope/head_dim = 0.5`.
- Batch sizes above 1 are not measured, so absorbed MLA throughput under serving load is not covered.
- Context lengths stop at 32K.
