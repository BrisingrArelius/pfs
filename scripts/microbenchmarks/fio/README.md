# Local-storage FIO test

Implemented in `run_fio.py` and `run_support.py`. The earlier single-job protocol
completed a cluster pilot and four-host run; protocol 5 changes the measurement
to four sustained parallel jobs and therefore requires a new pilot. Previous scripts remain
[archived](../../../archive/legacy-code/fio-abandoned/README.md).

- [DESIGN.md](DESIGN.md): standalone requirements, function contracts,
  two-module call flow, target-selectable pilot, timing fields, balanced ordering,
  reservation handling, measurement-level resume and the results report.
- [fio_config.json](fio_config.json): workload and planning settings.
- [target_inventory.json](target_inventory.json): recorded host/target mapping.

The design incorporates `Obsidian/DaSH/BeeGFS/Specs/MicroBenchmarks.md` and
`Global.md` from the Obsidian vault, with the user-directed per-target-only scope
and shared-file preparation policy. It covers five workloads and five repetitions.
The old pilot does not validate protocol 5's larger dataset or timing estimates.

Each measured invocation runs four jobs concurrently for **60 measured seconds
after a 5-second ramp**. Every job owns a disjoint 10-GiB region of one 40-GiB
file, uses `iodepth=32`, and repeatedly accesses its region until runtime expires.
The resulting bandwidth and IOPS are aggregate per-OST values across the four
jobs. The configured plan has 175 measured invocations per host, 700 across four
hosts, plus one full-data preparation per target: 728 FIO invocations total before
retries. Reuse the file for all 25 measurements, retain it across reservation
stops, and delete it after the target is complete.
Measurement-level checkpoint/resume preserves successful repetitions and reuses
the retained file after validating its mount and identity.

Admission uses pilot/observed durations with a margin, not hard timeouts. Protocol
5 starts with conservative estimates of 360 seconds for HDD preparation, 20
seconds for NVMe preparation and 66 seconds per measured invocation. Replace
these estimates with observed pilot timings if they are insufficient. A hard
timeout is an unexpected failure with no automatic retries.

## Run the pilot

Run from the repository root on an inventoried `colva` host, with Python 3.9+,
FIO/libaio and findmnt installed. The account needs write access to the target mounts and a
persistent results directory. No page-cache drops or privileged device access
are used. Keep one benchmark instance active per host.

Preparation estimates control admission, not FIO duration. Review them if another
host is slower; admission applies the configured margin. The 1,800-second hard
timeout is never used as an estimate.

On **colva1**, target 101 is HDD and 104 is NVMe:

```bash
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir results/microbenchmarks/runs/local-fio-pilot/colva1 \
  --targets 101,104 --pilot --time-limit 30m
```

This smoke pilot executes 10 measurements and two preparations: every workload
once on each target. Each invocation has a 5-second ramp and 60 measured seconds.
The same prepared 40-GiB file is split into four non-overlapping 10-GiB job
regions and remains in place across all measurements; explicit `overwrite=1` and
`fallocate=none` avoid a fresh allocation phase. The runner verifies file
device/inode/size around I/O.

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

## Run the full experiment

These commands run protocol 5's four-job sustained aggregate-throughput
experiment. Four jobs increase offered concurrency but do not prove an absolute
hardware maximum; use the pilot to confirm the OST is saturated and results are
stable. Use the same repository revision and run name on all four hosts. Home directories on the
storage nodes may be node-local, so retrieve or stage every host's results before
discarding the reservation or node state.

Paste this complete block from the repository checkout on each of `colva1`,
`colva2`, `colva3` and `colva4`. Change `REPO` if the checkout is elsewhere and
choose a new `RUN` name rather than reusing an existing result directory:

```bash
set -euo pipefail
REPO="$HOME/pfs"
RUN="local-fio-full-03"
HOST="$(hostname -s)"
case "$HOST" in colva1|colva2|colva3|colva4) ;; *) echo "Unexpected host: $HOST" >&2; exit 1 ;; esac
cd "$REPO"
printf 'Host: %s\nRevision: %s\n' "$HOST" "$(git rev-parse HEAD)"
fio --version
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir "$HOME/pfs-results/fio/$RUN/$HOST" \
  --time-limit 5h
```

The command intentionally omits `--targets`, selecting every inventory target on
the current host. A zero exit can also mean a planned budget stop. Confirm the
last line says `completed`; if it says `budget_stop`, reacquire time and paste:

```bash
set -euo pipefail
REPO="$HOME/pfs"
RUN="local-fio-full-03"
HOST="$(hostname -s)"
cd "$REPO"
python3 scripts/microbenchmarks/fio/run_fio.py \
  --results-dir "$HOME/pfs-results/fio/$RUN/$HOST" \
  --resume --time-limit 5h
```

After a host reports `completed`, create a checksummed archive and stage it on
`anjuna3`. Paste this block on that `colva` host; rerunning it replaces only that
host's archive with a newly generated copy:

```bash
set -euo pipefail
RUN="local-fio-full-03"
HOST="$(hostname -s)"
ARCHIVE="$RUN-$HOST.tar.gz"
ssh pfs@anjuna3 "mkdir -p ~/pfs-results/fio-staging/$RUN"
tar -C "$HOME/pfs-results/fio/$RUN" -czf "$HOME/pfs-results/fio/$ARCHIVE" "$HOST"
(cd "$HOME/pfs-results/fio" && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256")
scp "$HOME/pfs-results/fio/$ARCHIVE" "$HOME/pfs-results/fio/$ARCHIVE.sha256" \
  "pfs@anjuna3:~/pfs-results/fio-staging/$RUN/"
```

After staging all four hosts, paste this block on the PC from the repository root.
It downloads the archives, verifies their checksums, extracts the four host
directories, validates and normalizes all native evidence, and creates the five
per-access-pattern plots:

```bash
(
  set -euo pipefail
  RUN="local-fio-full-03"
  JUMP="dashlab@lab.dashlab.in"
  STAGING_HOST="pfs@anjuna3.dashlab.in"
  DOWNLOAD="$HOME/fio-result-downloads/$RUN"
  DEST="results/microbenchmarks/runs/$RUN"
  mkdir -p "$DOWNLOAD" "$DEST"

  for HOST in colva1 colva2 colva3 colva4; do
    REMOTE="pfs-results/fio-staging/$RUN/$RUN-$HOST.tar.gz"
    scp -J "$JUMP" \
      "$STAGING_HOST:$REMOTE" "$STAGING_HOST:$REMOTE.sha256" \
      "$DOWNLOAD/"
  done

  (cd "$DOWNLOAD" && sha256sum -c -- *.sha256)
  for ARCHIVE in "$DOWNLOAD"/*.tar.gz; do
    tar -xzf "$ARCHIVE" -C "$DEST"
  done
  for HOST in colva1 colva2 colva3 colva4; do
    test -f "$DEST/$HOST/manifest.json"
  done
  python3 scripts/microbenchmarks/fio/parse_results.py \
    "$DEST" --output-dir "$DEST/analysis"
  python3 scripts/microbenchmarks/fio/visualize_results.py \
    "$DEST/analysis/measurements.csv" \
    --output-dir "$DEST/analysis/plots"
)
```

The parentheses run strict error handling in a child shell. If a download,
checksum, extraction, parse or plot step fails, that child stops at the failing
command but the interactive terminal remains open and displays the error.

The parser must report 700 measurements, 28 preparations and no errors or
warnings for a complete four-host run. Review `$DEST/analysis/parse_report.json`
before interpreting the CSV summaries or figures.

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

## Parse and visualize results

Create normalized CSV files, a Markdown summary and a validation report from one
or more copied host result directories. The output directory may be below the
input tree because the parser explicitly excludes it from evidence discovery:

```bash
RUN="local-fio-full-03"
python3 scripts/microbenchmarks/fio/parse_results.py \
  "results/microbenchmarks/runs/$RUN" \
  --output-dir "results/microbenchmarks/runs/$RUN/analysis"
```

Create five per-OST bandwidth figures from the normalized measurements, one for
each access pattern. Each figure places OST IDs on the horizontal axis and measured
bandwidth in MiB/s on the vertical axis. This requires Matplotlib:

```bash
RUN="local-fio-full-03"
python3 scripts/microbenchmarks/fio/visualize_results.py \
  "results/microbenchmarks/runs/$RUN/analysis/measurements.csv" \
  --output-dir "results/microbenchmarks/runs/$RUN/analysis/plots"
```

Both commands treat the native run evidence as read-only. `parse_report.json`
records validation errors and warnings; `plot_manifest.json` records every plot
and its semantics. Each host result directory contains a generated `RUN.md`, and
the parser writes `analysis/run_configuration.md` as a readable multi-host record
of the exact configuration. Each figure keeps every OST in its own horizontal-axis
category: small ticks show the five repetition values, whiskers show their
min-max range and a colored bar shows the OST median.
HDD and NVMe are distinguished by color but are never aggregated.
Color-matched dotted lines label the mean of the per-OST medians for HDD and
NVMe, providing explicit MiB/s reference values without pooling repetitions.

## Developer verification (no benchmark I/O)

```bash
python3 -B -m unittest discover -s scripts/microbenchmarks/fio/tests -v
```

Tests use system-managed temporary directories, sparse fixture files, mocked
mount/FIO operations and harmless Python subprocesses. They do not run FIO,
findmnt, sudo or cache drops.

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
