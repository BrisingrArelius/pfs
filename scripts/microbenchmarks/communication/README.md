# BeeGFS NetBench communication benchmark

[`DESIGN.md`](DESIGN.md) defines the eight-case pilot and 240-unit full matrix.
`communication_inventory.json` is an **unreviewed template**, not live target or
NetBench state. Read [`IMPLEMENTATION_RULES.md`](../IMPLEMENTATION_RULES.md),
[`Global.md`](../../../docs/specs/Global.md), and
[`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md) before execution.

`run_communication.py` supplies fixed IOR argv and checks a recorded NetBench
transition; `visualize_results.py` reads native IOR, saved commands, state and
telemetry directly into separate synthetic read/write and backend-evidence plots:

```bash
python3 scripts/microbenchmarks/communication/visualize_results.py \
  results/microbenchmarks/runs/<communication-run-id>
```

`plots/plot_manifest.json` identifies the complete PNG generation under
`plots/generations/<id>/`. The pilot must
establish and fingerprint a finite `backend_data_ratio_max` before full-run
evidence can validate. Pilot plots are labelled `calibration_only`;
the separately approved full protocol records `calibrated_from` with that pilot
plan fingerprint. A watchdog-backed live NetBench/pool/chooser transaction and
read preparation are still required.
