# Incremental embedding-index benchmark

`scripts/benchmark_embedding_index.py` measures the repository's real
`build_embedding_index()` implementation. It compares a fresh build of each
deterministic snapshot with an incremental refresh from an independently built
original index. It never includes model loading or copying the starting index
in the measured refresh interval.

Run the pinned BuildCores experiment:

```powershell
$Input = 'data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/full/records.jsonl'
uv run --no-sync python scripts/benchmark_embedding_index.py --input $Input --device cuda --batch-size 128 --repetitions 5
```

The tool creates a timestamped directory below
`artifacts/evaluation/embedding-index-benchmark/` containing `runs.csv`,
`summary.json`, and `report.md`. It uses a deterministic selection seeded by
`--seed`, tests 0%, 1%, 2%, 5%, 10%, and price-only changes, and fails on ID
order, shape, vector tolerance (`rtol=atol=1e-6`), content-hash, or reuse-count
mismatches. Semantic mutations change a field used by `build_product_embedding_text`; the
price control changes an unused price field and asserts text equality first.

Timings are warm-start wall-clock times: model warm-up is reported separately
in `summary.json`. Peak RAM is sampled process working set, not GPU memory.
The incremental implementation still loads and validates the old full matrix,
allocates and writes a replacement full matrix, and rewrites its ID map; low
change rates can therefore avoid encoding work without delivering proportional
end-to-end speedups. Results are local experimental evidence, not a production
latency guarantee.
