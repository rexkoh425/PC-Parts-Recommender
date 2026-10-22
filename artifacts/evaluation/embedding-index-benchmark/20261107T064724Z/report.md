# Embedding-index benchmark

Warm-start timings exclude model loading and input/index copying.

| Change rate | Full rebuild (s) | Incremental (s) | Speedup | Recomputed | Reused | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 8.435 | 1.018 | 8.29x | 0 | 25666 | 2100.9 MiB |
| 1% | 8.448 | 1.080 | 7.82x | 257 | 25409 | 2124.1 MiB |
| 2% | 8.420 | 1.167 | 7.22x | 513 | 25153 | 2125.4 MiB |
| 5% | 8.427 | 1.563 | 5.39x | 1283 | 24383 | 2124.2 MiB |
| 10% | 8.504 | 1.770 | 4.81x | 2567 | 23099 | 2126.2 MiB |
| price-only | 8.393 | 1.164 | 7.21x | 0 | 25666 | 2112.9 MiB |

All runs passed ID-order, shape, content-hash, and vector allclose verification.
