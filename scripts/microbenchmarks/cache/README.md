# Cache-effects benchmark

[`DESIGN.md`](DESIGN.md) defines the 8-GiB normal-read experiment. The checked-in
`cache_inventory.json` is **unreviewed**; it is a field template, not permission
to drop caches or remount BeeGFS. Read
[`IMPLEMENTATION_RULES.md`](../IMPLEMENTATION_RULES.md) before extending this code.

`run_cache.py` creates the 15-unit plan or a four-unit pilot plan, builds the
single-rank IOR argv, and classifies saved network/backend/residency evidence.
`visualize_results.py` reads the manifest, native IOR JSON, and telemetry directly.
It plots validated read throughput by achieved cache state and network/backend
byte ratios; unverified states never appear as confirmed hits. For an existing
raw run:

```bash
python3 scripts/microbenchmarks/cache/visualize_results.py \
  results/microbenchmarks/runs/<cache-run-id>
```

`plots/plot_manifest.json` points to the complete PNG generation under
`plots/generations/<id>/`. The execution
transaction and independent cache-mode/pool restoration are not yet implemented.
