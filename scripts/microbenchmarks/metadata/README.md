# BeeGFS metadata benchmark

[`DESIGN.md`](DESIGN.md) defines one mdtest invocation as the restartable unit.
`metadata_inventory.json` is an **unreviewed field template**. The four pilot
cases retain the full 100,000 items per rank; admit the pilot only after measured
pre-pilot timing shows the work, validation, and cleanup fit under 20 minutes.

`run_mdtest.py` supplies the 90-unit plan, argv, and owned-namespace checks;
`visualize_results.py` reads native stdout and produces one plot per phase under a
pilot-approved version-specific schema, keeping file/directory operations distinct:

```bash
python3 scripts/microbenchmarks/metadata/visualize_results.py \
  results/microbenchmarks/runs/<metadata-run-id>
```

`plots/plot_manifest.json` lists the current PNGs in
`plots/generations/<id>/`. The static cleanup prototype cannot authorize live namespace deletion;
remote rank supervision and race-safe owned cleanup must be implemented first.

See [`IMPLEMENTATION_RULES.md`](../IMPLEMENTATION_RULES.md) for write boundaries,
deadline control and execution gates.
