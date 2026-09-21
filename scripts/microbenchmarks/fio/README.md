# Local-storage FIO test

Implemented in `run_fio.py` and `run_support.py`; cluster pilot validation is
still pending. The previous scripts remain
[archived](../../../archive/legacy-code/fio-abandoned/README.md).

- [DESIGN.md](DESIGN.md): standalone requirements, function contracts,
  two-module call flow, target-selectable pilot, timing fields, balanced ordering,
  reservation handling, measurement-level resume and the results report.
- [fio_config.json](fio_config.json): workload and planning settings.
- [target_inventory.json](target_inventory.json): recorded host/target mapping.

The design incorporates `Obsidian/DaSH/BeeGFS/Specs/MicroBenchmarks.md` and
`Global.md` from the Obsidian vault, with the user-directed per-target-only scope
and shared-file preparation policy. It covers five workloads and five repetitions.
Dataset size, queue depth and timing choices remain subject to a cluster pilot.

Each measured job stops after **10 GiB of I/O or 60 seconds, whichever comes
first**, with no ramp period. The configured plan has 175 measured invocations
per host, 700 across four hosts, plus one full-data preparation per target:
728 FIO invocations total before retries. Reuse that file for all 25 measurements,
retain it across reservation stops, and delete it after the target is complete.
Measurement-level checkpoint/resume preserves successful repetitions and reuses
the retained file after validating its mount and identity.

Admission uses pilot/observed durations with a margin, not hard timeouts. Fill
the null preparation estimates in `fio_config.json` from the pilot. A hard
timeout is an unexpected failure: stop the session without automatic retries.

## Run the pilot

Run from the repository root on an inventoried `colva` host, with Python 3.9+,
FIO/libaio and findmnt installed. The account needs write access to the target mounts and a
persistent results directory. No page-cache drops or privileged device access
are used. Keep one benchmark instance active per host.

**First set preparation estimates** in `fio_config.json`:
`planning.prepare_seconds.hdd` and `.nvme` are deliberately `null`. Supply positive
wall-clock seconds for writing and syncing the full 10-GiB file. For the first
pilot, use provisional estimates based on the hardware; afterward replace them
with observed setup times. These estimates control admission, not FIO duration.
Missing estimates produce an explicit error; the 600-second hard timeout is
never used as an estimate. Empty measurement estimates fall back to 60 seconds.

On **colva1**, target 101 is HDD and 104 is NVMe:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --targets 101,104 --time-limit 2h
```

This executes 50 measurements and two preparations. Each measurement transfers
10 GiB or stops normally at 60 seconds. The file remains the same across its 25
measurements; explicit `overwrite=1` and `fallocate=none` avoid a fresh allocation
phase in each measured job. The runner verifies file device/inode/size around I/O.

Resume after reacquiring a reservation:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --resume --time-limit 2h
```

The saved target selection is reused. Completed repetitions are skipped; an
interrupted repetition restarts from its beginning. Missing/incomplete prepared
files are prepared again once; an unexpected replacement file is an error.
Cleanup failures retry cleanup without repeating successful measurements.

To extend an **active** reservation from another terminal:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --extend-deadline 2h
```

Alternatively use `--deadline` with a timezone-qualified ISO-8601 timestamp.
`--cleanup-buffer` defaults to five minutes. Exit 0 means completion or an expected
budget stop; check the manifest session outcome to distinguish them. Hard timeout
or command/validation failure exits nonzero and stops the session. Ctrl-C/SIGTERM
exits 130. Diagnose a failure before explicitly resuming.

## Evidence and pilot checks

`manifest.json` records settings, ordered cases, sessions, timing estimates,
attempts, file identity, capacity and target-device diskstats. Each raw attempt
directory retains `job.fio`, native `fio.json`, `stdout` and `stderr`. Preparation
has separate directories; measurement timing excludes preparation and final setup
sync. No analysis result is needed to resume a measurement.

Check pilot file reuse, byte/time completion reasons, timing/latency fields and
repeat-to-repeat spread. Use `command_wall_seconds` for planning; use native FIO
statistics for performance. Scientific settings/target selection/FIO version
must match on resume; planning estimates may be updated. For the full experiment,
choose a new results directory and omit `--targets` to select all local targets.

## Developer verification (no benchmark I/O)

```bash
python3 -B -m unittest discover -s scripts/microbenchmarks/fio/tests -v
```

Tests use `/tmp/opencode`, sparse fixture files, mocked mount/FIO operations and
harmless Python subprocesses. They do not run FIO, findmnt, sudo or cache drops.
