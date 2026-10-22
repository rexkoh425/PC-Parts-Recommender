# Embedding-index benchmark

Warm-start timings exclude model loading and input/index copying.

| Change rate | Full rebuild (s) | Incremental (s) | Speedup | Recomputed | Reused | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 8.392 | 1.200 | 7.00x | 0 | 25666 | 0.0 MiB |
| 1% | 8.576 | 1.079 | 7.95x | 257 | 25409 | 0.0 MiB |
| 2% | 8.393 | 1.137 | 7.38x | 513 | 25153 | 0.0 MiB |
| 5% | 8.406 | 1.379 | 6.09x | 1283 | 24383 | 0.0 MiB |
| 10% | 8.391 | 1.762 | 4.76x | 2567 | 23099 | 0.0 MiB |
| price-only | 8.636 | 0.954 | 9.05x | 0 | 25666 | 0.0 MiB |

All runs passed ID-order, shape, content-hash, and vector allclose verification.
