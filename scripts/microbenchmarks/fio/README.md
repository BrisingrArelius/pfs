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

Admission uses pilot/observed durations with a margin, not hard timeouts. Current
estimates come from the colva1 target-101/104 smoke pilot: 130 seconds for HDD
preparation, 6 seconds for NVMe preparation, and rounded-up command times by
workload. A hard timeout is an unexpected failure with no automatic retries.

## Run the pilot

Run from the repository root on an inventoried `colva` host, with Python 3.9+,
FIO/libaio and findmnt installed. The account needs write access to the target mounts and a
persistent results directory. No page-cache drops or privileged device access
are used. Keep one benchmark instance active per host.

Preparation estimates control admission, not FIO duration. Review them if another
host is slower; admission applies the configured margin. The 600-second hard
timeout is never used as an estimate.

On **colva1**, target 101 is HDD and 104 is NVMe:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --targets 101,104 --pilot --time-limit 30m
```

This smoke pilot executes 10 measurements and two preparations: every workload
once on each target. Its configured measurement time is at most 10 minutes.
Each measurement transfers
10 GiB or stops normally at 60 seconds. The file remains the same across its 25
measurements; explicit `overwrite=1` and `fallocate=none` avoid a fresh allocation
phase in each measured job. The runner verifies file device/inode/size around I/O.

Resume after reacquiring a reservation:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --resume --time-limit 30m
```

The saved target selection is reused. Completed repetitions are skipped; an
interrupted repetition restarts from its beginning. Missing/incomplete prepared
files are prepared again once; an unexpected replacement file is an error.
Cleanup failures retry cleanup without repeating successful measurements.
Pilot mode is also restored from the manifest; `--pilot` is optional on resume.

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

## Write-safety boundary

The runner may modify only the selected results directory and this exact generated
file on a verified target mount:

```text
<mount>/.local-fio/<32-character-run-id>/target-<inventory-id>/data
```

FIO receives that explicit regular-file path, never an inventory device path or
anything below `beegfs_storage`. It runs with its working/auxiliary directory in
the corresponding raw results folder. Cleanup validates the generated path,
mount, symlink-free regular-file identity and inode before unlinking exactly
`data`; it does not use globs or recursive deletion.

Tests reject tampered manifest traversal, artifact paths outside results, symlink
redirection and unexpected replacement files. They also place sentinels in
`beegfs_storage`, the mount root and adjacent `.local-fio` paths, then verify all
remain unchanged through preparation, measurements and cleanup.
