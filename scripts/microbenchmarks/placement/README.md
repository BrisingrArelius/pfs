# End-to-end placement benchmark

The end-to-end benchmark design is [`DESIGN.md`](DESIGN.md). The new
`run_placement.py` supplies a fixed 6,480-unit baseline-capacity plan, an
approved-band capacity extension, a six-case pilot proposal, and exact IOR argv.
`visualize_results.py` reads saved native IOR, layouts and telemetry directly and
creates per-workload, chooser, capacity and completion-reason figures. Run:

```bash
python3 scripts/microbenchmarks/placement/visualize_results.py \
  results/microbenchmarks/runs/<placement-run-id>
```

`plots/plot_manifest.json` lists the complete PNG generation under
`plots/generations/<id>/`. `placement_inventory.json` is an **unreviewed
template**; neither the planner nor the historical shell scripts authorize live
pool changes. Read [`IMPLEMENTATION_RULES.md`](../IMPLEMENTATION_RULES.md) and
the [global specification](../../../docs/specs/Global.md) before execution.

Helper scripts for configuring BeeGFS storage pools used by the workload pipeline.

These scripts contain historical target inventories and are not verified against
the current cluster. See [methodology](../../../docs/methodology.md) for D/S/H.
The full experiment design is not implemented by these helpers.

## Files

- `configure_pools.sh` — create or update HDD/SSD storage pools and move hard-coded targets
- `reset_pools.sh` — move hard-coded targets to Default and remove named HDD/SSD pools

## Historical usage

Do not run these scripts as the benchmark's setup or restoration mechanism.
They encode old target memberships and cannot restore the exact live baseline.
The end-to-end design requires a reviewed inventory and reversible transaction
with an independent coordinator-loss watchdog instead.

## Notes

- These scripts are environment-specific and assume BeeGFS administrative privileges.
- The scripts move target membership, not existing file contents or directory patterns.
- Their 4xx targets and target 104 media assignment conflict with historical
  context notes. Resolve live inventory separately before treating them as current setup.
