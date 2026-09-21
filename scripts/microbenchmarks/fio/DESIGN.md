# Local-storage FIO: implementation design

**Status: implemented; cluster pilot pending.** `run_fio.py` implements the
experiment and `run_support.py` owns process/deadline handling. Recovery checks
use fake FIO and harmless subprocesses; the previous scripts remain archived.

## 1. Research question and scope

> Under a fixed I/O configuration, how do individual local target filesystems
> compare across five access patterns, and how much do their results vary?

Measure files on the inventoried XFS target mounts on `colva1`–`colva4`. The path
includes the filesystem, controller and device. Results provide a backend
reference for later BeeGFS experiments, not universal device peaks, sustained
SSD write performance, or an overhead subtractable from BeeGFS throughput.

This design incorporates `Obsidian/DaSH/BeeGFS/Specs/MicroBenchmarks.md` (§3, §9–12)
and `Global.md` from `/home/arelius/projects/obsidian`, with these explicit
user-directed refinements:

- **Per-target only.** No simultaneous-target experiment or group abstraction.
- Prepare **one file per target benchmark**, reuse it across all workloads and
  repetitions on that target, then delete it.
- Keep measurement-level checkpointing, but preserve the prepared file across
  reservation boundaries when it remains valid.

These narrow the broader notes' concurrency scope and fresh-preparation rules.
The source notes have not been edited. The requirements here are standalone.

| Workload | FIO `rw` | FIO `bs` |
|---|---|---|
| Sequential read | `read` | `1m` |
| Sequential write | `write` | `1m` |
| Small random read | `randread` | `4k` |
| Small random write | `randwrite` | `4k` |
| Large random read | `randread` | `128k` |

Run five separate repetitions of each workload. The 128-KiB case characterizes
larger random requests; it is not a special BeeGFS boundary. Targets within a
host run sequentially; hosts may run in parallel with separate output directories.

## 2. Fixed protocol and interpretation

Use one worker, one 10-GiB file, `ioengine=libaio`, `direct=1`, and `iodepth=32`.
Freeze these settings after a small pilot rather than sweeping every parameter.

Each measurement stops after **10 GiB of I/O or 60 seconds, whichever comes
first**: `time_based=0`, `runtime=60`, `ramp_time=0`. Random operations have the
same byte budget; this is not a guarantee of visiting every block exactly once.
The measured byte budget restarts at each invocation; it is not shared by runs.
A job that transfers 10 GiB in two seconds ends after those two seconds. It does
not wait for 60 seconds, repeat the transfer, or require a minimum duration.

This is a bounded-workload comparison. A short SSD measurement samples different
time behavior from a 60-second HDD measurement. Always retain actual bytes,
duration and completion reason (`byte_limit` or `time_limit`). A short write can
fit within a device's internal cache. Direct I/O bypasses the normal data page
cache, not controller/device caches, and does not establish write durability.
No system-wide cache drops, routine Darshan, or persistent-write claims.

Use native bandwidth, IOPS and repeat-to-repeat variation as primary results.
Retain mean latency and completion-latency p50/p95/p99/p99.9, but interpret tails
alongside operation count and duration. A short run's p99.9 can represent very
few observations. Separate within-target variation from differences between
targets; different devices are not interchangeable repetitions of one device.

### One file for the entire target benchmark

1. Verify the target mount, source device, XFS and available space.
2. Create `<mount>/.local-fio/<run-id>/target-<id>/data`, outside `beegfs_storage`.
3. Write the full 10 GiB once with sequential 1-MiB direct I/O and `end_fsync=1`.
   Preparation is size-based and has no 60-second measurement cap.
4. Verify successful full-data preparation and checkpoint the file's identity.
5. Run all 25 measurements against that same existing file. Reads reuse its
   data; measured writes overwrite it. Do not truncate, unlink or reinitialize
   it between workloads or repetitions. Set measured jobs to `allow_file_create=0`.
   This creation guard alone does not prove overwrite behavior: verify the same
   device/inode and length before/after measured jobs in the pilot. Confirm the
   installed FIO version does not truncate/recreate the file or repeat allocation;
   use explicit overwrite/allocation options if its defaults require them.
6. After all 25 measurements have durable valid results, remove the file and
   record target cleanup completion. A cleanup failure must not invalidate or
   rerun successful measurements; resume retries cleanup separately.

Preparation is excluded from measured metrics. The five executions share file
allocation and device history; they are not claimed to start from identical
physical device states. Record execution order and allocation/session boundaries.
Contents may change during write workloads; this tests I/O, not data integrity
or identical-payload replay. Device-state reset and file checksumming are outside
this experiment.

Reserve 5 GiB beyond the new file before preparation; once it exists, require
only the reserve. Record free bytes, free percent, free inodes and device counters
before/after measurements. This headroom is not a high/low-capacity study.

### Ordering without a scheduling framework

Use `order_seed` to generate and save a reproducible target order. Finish one
target before moving to the next. Shuffle the five workloads once, then rotate
that sequence left by one position for each of five repetition rounds. For example,
`ABCDE`, `BCDEA`, `CDEAB`, `DEABC`, `EABCD`: every workload occupies each position
once. Use the same round orders across targets. This balances position, not every
possible preceding-workload effect, and does not reset device history.

Set FIO's random seed from the repetition, independently of the ordering RNG.
Resume follows the saved plan, not a newly generated order. No new measurements
or scheduling framework are needed.

## 3. Counts and realistic time planning

| Scope | Measurements | Preparations | Total FIO invocations |
|---|---:|---:|---:|
| One target | 5 workloads × 5 repetitions = 25 | 1 | 26 |
| One seven-target host | 175 | 7 | 182 |
| Four hosts / 28 targets | 700 | 28 | 728 |

These counts exclude retries and replacement of lost/incomplete prepared files.
Normal preparation writes **280 GiB total**, not a fresh 10 GiB for every run.
The measurement intervals total at most **2 h 55 m per host** (175 × 60 seconds);
fast byte-completing cases take less. Preparation, startup, I/O drain and cleanup
are additional. The 60-second FIO cap is not an exact process wall-time bound.

```text
measurement time ≈ min(10 GiB / workload-specific observed rate, 60 seconds)
target time      ≈ one preparation + 25 measurement times + overhead
host time        ≈ sum of its seven target times
parallel elapsed ≈ slowest host time
```

At 6 GB/s, 10 GiB takes about 1.79 seconds; at 6 Gbit/s, about 14.32 seconds.
A sequential rating does not predict random-I/O performance. At 150 MiB/s,
preparation takes about 68.27 seconds. If all seven preparations take that long
and every measurement hits 60 seconds, a host takes about **3 h 3 m plus overhead**.
This is an illustration, not a promised reservation duration.

### Estimate, hard timeout and reservation deadline are different

- **Estimate:** expected phase wall time, used to decide whether to start it.
  Use recent successful wall times for the same target/workload when available;
  otherwise use pilot estimates for its media/workload. A measurement without
  an estimate uses 60 seconds as a conservative fallback. Preparation requires
  a supplied pilot estimate for that media before a full run.
- **Admission budget:** estimated time × 1.25 + 5 seconds of overhead margin.
  These configurable margins are planning choices, not scientific factors.
  Before initial setup, budget preparation plus the first pending measurement.
  Thereafter budget only the next measurement. Always subtract the cleanup buffer
  from available reservation time. Never budget the 600-second failure timeout
  as if it were expected preparation time.
- **Hard timeout:** exceptional failure protection. Initially allow 600 seconds
  for preparation and 75 seconds for measurement wall time. Pilot the limits;
  normal commands should never reach them. If reached, kill/reap the child,
  mark the attempt failed, preserve evidence and **abandon the current session**.
  No automatic retry and no continuing to the next measurement. Diagnose before
  explicitly resuming. Do not discard already completed measurements.
- **Reservation deadline:** enforced independently while the process runs. If an
  estimate is wrong and the reservation boundary is reached, stop and checkpoint
  an interrupted measurement. That is recoverable, not a valid timed result.

FIO's normal 60-second `runtime` completion is successful measurement termination,
not a hard timeout. Extending a reservation changes the allocation deadline,
not the command failure timeout. Estimates update from observed wall times without
changing the scientific workload. There is no sophisticated prediction engine.

Record distinct timing fields rather than one ambiguous `duration`:

| Field | Meaning and use |
|---|---|
| `fio_runtime_ms` | Native FIO measurement duration; keep native units/fields in raw JSON. Performance statistics come from FIO, not wrapper wall time. |
| `command_wall_seconds` | Monotonic elapsed time from child launch through reaping, including startup/drain. This supplies planning observations. |
| `estimated_wall_seconds` | Selected expected command wall time before admission. |
| `admission_seconds` | Estimate after planning margin and overhead. |
| `remaining_seconds` | Allocation time left after reserving cleanup time, sampled at admission. |
| `hard_timeout_seconds` | Exceptional command wall-time limit, not a scheduling estimate. |

Also record target preparation, cleanup and session wall times using monotonic
clocks for elapsed intervals and UTC timestamps for provenance. These few timings
explain overhead without implementing a profiling system. Data preparation's final
sync is setup time; it must not enter a measured workload's bandwidth calculation.

## 4. Small implementation with clear ownership

```text
fio/
  run_fio.py             experiment loop, file lifecycle, evidence validation
  run_support.py         deadline/process handling and atomic JSON writes
  fio_config.json        fixed FIO options, workloads and planning settings
  target_inventory.json  host -> target ID, mount, device and media
```

Use standard-library Python, FIO and findmnt. `run_fio.py` imports support, never
the reverse. Use plain dictionaries and paths, one manifest, one process at a
time and no metric aggregation inside the runner. Keep the code tight through
clear ownership and reuse, not compressed statements. There is **no line-count
or function-count quota**. No generic matrix framework, plugins, scheduler,
automatic retry engine, or additional helper-package hierarchy.

### Function contracts: `run_fio.py`

| Function | Responsibility |
|---|---|
| `parse_args()` | Parse the small CLI in section 7 and reject incompatible options. |
| `plan_cases(config, targets)` | Pure function: seed and record target order and per-round workload order; produce stable target/workload/repetition IDs. Validate unique target IDs and workload names. |
| `load_run(args)` | Load config/inventory, detect host and FIO version, validate target selection, create or restore the manifest, check compatibility and recover abandoned attempts. Validate completed evidence and record a new session. |
| `check_target(target)` | Verify exact mount path, expected source and XFS with bounded findmnt; return identity, statvfs capacity/inodes and raw diskstats snapshot. Callers save the snapshot and enforce headroom. |
| `build_job(config, target, workload, file_path, phase)` | Pure function: one target-named FIO job, shared defaults plus `rw`/`bs`. Override options for preparation or measurement as specified below; return job text. |
| `validate_result(path, expected)` | Require one expected job, zero errors and plausible bytes/duration. Preparation must write the full extent; measurement must reach its byte budget or normal runtime cap. Return decoded native JSON and completion reason; do not aggregate metrics. |
| `estimate_seconds(run, target, workload, phase)` | Select observed or pilot phase wall time and apply the documented admission margin. No disk I/O or process control. |
| `run_target(run, target, cases)` | Own file lifecycle: mount/headroom checks, prepare once or validate retained file, execute pending cases, then delete only when all are complete. Checkpoint setup and cleanup separately. |
| `run_case(run, case)` | Admit one measurement, checkpoint a new attempt, capture snapshots, run/validate FIO and save its outcome. Never prepare or delete the shared target file. |
| `execute_fio(...)` | Shared preparation/measurement invocation: durable artifacts, native validation, identity/capacity snapshots and attempt outcome. |
| `main()` | Handle extension-only mode or acquire the run-directory lock, load state, initialize the deadline, and call `run_target` sequentially. Own session exit status and signal handling. |

Give each function a short purpose/side-effect docstring. The contracts establish
boundaries; small extractions are allowed when they make the implementation clearer.

### Function contracts: `run_support.py`

| Function | Responsibility |
|---|---|
| `atomic_json(path, value)` | Write a temporary sibling, flush/fsync, replace destination and fsync parent. Used for manifest/deadline; clean temporary files on error. |
| `set_deadline(path, *, time_limit=None, deadline=None, extend=None)` | Write an absolute UTC epoch deadline atomically. Extension adds to an existing unexpired deadline. One updater at a time is assumed. |
| `remaining_seconds(path, cleanup_seconds)` | Read the current deadline and subtract wall time and cleanup reserve. Missing/malformed/non-finite state is an error, never unlimited time. |
| `run_command(argv, stdout_path, stderr_path, deadline_path, cleanup_seconds, timeout_seconds)` | Own a subprocess group, file-backed stdout/stderr and one-second polling. Enforce the live reservation boundary and monotonic command timeout, flush/fsync outputs, terminate/reap on exceptions and raise on nonzero exit. Return launch-to-reap wall seconds on success; attach elapsed time to command failures/timeouts. |

Use one custom `BudgetExpired` exception for reservation exhaustion, standard
`TimeoutError` for hard timeouts, and `CalledProcessError` for command errors.
Handle SIGINT/SIGTERM through the same interruption path. Terminate the group with
TERM, wait up to ten seconds, then KILL/reap; tolerate an already-exited child.
Metadata probes use short explicit timeouts. These helpers know nothing about
target IDs, workloads, progress schema or retries.

## 5. Control flow

```text
main
  parse_args; handle deadline-update-only mode
  lock output directory; load_run; set_deadline
  for each target in saved order
    run_target
      check_target; save identity/capacity
      if completed measurements but cleanup pending: retry cleanup only
      if pending measurements:
        validate retained file, or admit and prepare one full file
        for each pending case in saved order
          run_case
            check estimated duration against remaining reservation
            save running attempt and pre-measurement snapshot
            build_job(measure) -> run_command -> validate_result
            save post-measurement snapshot and durable outcome
        delete file only after all target measurements are complete
      on stop/failure: retain valid prepared file and unfinished case state
  save session outcome; release lock
```

`run_target` owns the data file; `run_case` owns measurement evidence;
`run_command` owns the process; `atomic_json` owns durable state replacement.
Record exact paths before creating data. Reject symlinked work directories/files,
verify mount identity before any cleanup, and never delete via a broad glob.
The implementation derives and revalidates the sole writable data pathname from
the verified mount, run ID and target ID; a manifest cannot redirect it. Raw
artifacts are likewise confined beneath the selected results directory. FIO runs
with that artifact directory as its process/auxiliary directory and receives no
device or `beegfs_storage` pathname.

If admission fails, leave the next case pending and stop normally. After valid
preparation, recheck time before the first measurement; if time is insufficient,
save the prepared state and exit without deleting it. The next reservation reuses
the file. Unexpected errors propagate out of the loop rather than being caught
and silently converted into another measurement.

## 6. Evidence and checkpoint recovery

Use output storage that survives releasing the allocation, one directory per host:

```text
results/microbenchmarks/runs/<run-id>/<host>/
  .lock
  manifest.json
  deadline.json
  raw/target-<id>/prepare-<generation>/job.fio, fio.json, stdout, stderr
  raw/<case-id>/attempt-<n>/job.fio, fio.json, stdout, stderr
```

The manifest is the only progress record. It holds:

- Scientific config/inventory snapshots, protocol/FIO versions and a canonical
  JSON SHA-256 fingerprint. Include workload, preparation and ordering policy.
  Include the selected target IDs; subset and full runs are distinct plans.
  Exclude mutable deadlines and duration estimates from scientific compatibility.
- Host/kernel, memory, known/unknown controller/device-cache settings, and session
  records: allocation ID or unknown, timestamps, deadline and exit reason.
- Saved case order and stable IDs: `<target-id>__<workload-name>__r<repetition>`.
- Per-target file path, preparation generation/attempts, successful setup evidence,
  file identity (device, inode, size), readiness and cleanup state.
- Measurement attempts: session, state, commands, artifact paths, timestamps,
  wall duration, snapshots, completion reason, exit status and error if any.

No attempt means `pending`. Attempts transition `running -> completed`, `failed`
or `interrupted`; retries create a new attempt, preserving the old evidence.
Save `running` before launching FIO. Publish `completed` only after native output
validation, required snapshots and durable raw evidence. Fsync generated job files,
outputs and new artifact directories before checkpointing their references.
Measurement completion does **not** wait for deletion of the shared data file.

### When a reservation ends

1. Stop/reap active FIO if necessary; mark that attempt interrupted, not completed.
2. Preserve completed measurements and the successfully prepared target file.
3. On explicit `--resume`, check the scientific fingerprint and mount identity.
   Convert abandoned `running` attempts to interrupted; validate completion evidence.
4. If preparation succeeded and the recorded regular file still has the same
   device/inode/size, reuse it. Measured overwrites do not invalidate preparation;
   do not require unchanged modification time or contents.
5. If the file is absent or preparation never completed, create a new preparation
   generation once for that target's remaining cases. Never accept size alone as
   evidence that a partial setup completed. An unexpected replacement/path identity
   mismatch is an error to investigate, not permission to overwrite an unknown file.
6. Skip valid completed measurements and restart the interrupted measurement from
   its beginning on the prepared file, not from a saved byte offset. Record the new
   session and any preparation generation change with subsequent measurements.
7. Delete the file after all remaining measurements finish. If a crash occurs after
   unlink but before cleanup checkpointing, absence of that known path completes
   cleanup; it does not trigger preparation or rerun measurements.

A normal FIO stop at 60 seconds is complete even below 10 GiB. External timeout or
reservation interruption is not. Validate duration/byte tolerances against the
installed FIO version in the pilot; short zero-error output with neither limit
reached is invalid. Missing/corrupt required raw evidence invalidates only the
affected measurement. Analysis failure alone never invalidates measured evidence.

Hard timeout abandons the session without an automatic retry. Retain failed
evidence and the target file for diagnosis; a later explicit resume follows the
same checks. Incomplete preparation files are not reusable and can be removed
only after verifying their recorded ownership/path. SIGKILL or machine loss may
leave them behind; recovery relies on recorded state, not guaranteed cleanup.

## 7. CLI and configuration

Commands (run on the inventoried host):

```bash
python3 run_fio.py --results-dir /persistent/pilot/colva1 --targets 101,104 --pilot --time-limit 30m
python3 run_fio.py --results-dir /persistent/run/colva1 --time-limit 5h
python3 run_fio.py --results-dir /persistent/run/colva1 --resume --time-limit 5h
python3 run_fio.py --results-dir /persistent/run/colva1 --extend-deadline 2h
```

Only `--results-dir`, `--targets`, `--pilot`, `--resume`, `--time-limit`, `--deadline`,
`--extend-deadline`, and `--cleanup-buffer` (default `5m`). Read config/inventory
beside the script and detect the actual host. Require one positive duration or
timezone-qualified absolute deadline for execution. Duration suffixes are `s`,
`m`, `h`. Extension mode only updates an existing deadline, not the run manifest.

`--targets` is a comma-separated list of IDs from the detected host's inventory.
Reject unknown, duplicate or other-host IDs before touching target files. A new
run defaults to every target on that host. On resume, omission means reuse the
saved selection; an explicit list must match it. Pilot subsets use the exact same
preparation, execution and checkpoint path, without editing the inventory. Two
selected targets in `--pilot` mode produce 10 measurements and two preparations:
all five workloads once on each target. This smoke test validates mechanics and
rough timing, not five-run variability. Pilot mode is fingerprinted and restored
automatically on resume; it cannot convert a full run. A separate full-run
directory preserves the pilot evidence. Without `--pilot`, five repetitions remain fixed.

The operator guarantees one active benchmark instance per host and controls
competing activity. Record this execution assumption; do not add another host
lock or distributed coordination system. Different hosts may run concurrently.

Use `fcntl.flock` to prevent two execution writers in one results directory. The
active runner rereads `deadline.json` every second; one extension writer is
assumed. Stop the child at deadline minus cleanup buffer, independently of its
hard timeout. Missing/malformed live deadline data is a failure, never an unlimited
reservation. The buffer covers termination and flushing; a reusable target file
is intentionally retained on reservation stop, not treated as cleanup debris.

Exit 0 for full completion or a planned budget stop, distinguished in the manifest;
130 for SIGINT/SIGTERM; nonzero for other failures. Stop the session on the first
failure. No automatic retry, target skipping, or hidden workload reduction.

`fio_config.json` contains:

- Shared native FIO options, five `name`/`rw`/`bs` workload entries, repetitions
  and `order_seed`; no simultaneous groups or arbitrary parameter matrix.
- Preparation options and hard timeout, measurement hard timeout and free-space
  reserve. These are distinct from the expected phase durations.
- Planning estimates by media and workload, multiplier and overhead margin.
  Empty measurement estimates use the 60-second fallback. Null preparation
  estimates must be filled from the pilot before full execution; do not substitute
  a hard timeout. Observed successful wall times can refine admission estimates.

Estimate lookup is `planning.prepare_seconds[media]` for setup and
`planning.measurement_seconds[media][workload_name]` for a measurement. Values
are positive wall-clock seconds. Prefer the maximum successful elapsed time
already observed for that target/workload within this benchmark, then apply the
margin; otherwise use the pilot value/fallback. Record which estimate admitted
each phase so an underestimated reservation stop can be explained.

`build_job` serializes FIO options directly, avoiding an option-translation layer.
The wrapper owns filenames, job name and output paths. Preparation removes
`runtime`/`ramp_time`, sets `time_based=0`, enables file creation, and applies
sequential write options with final sync. Measurement disables creation and keeps
the size/runtime caps, overlaying only workload `rw`/`bs` and repetition seed.
Planning fields and hard timeouts are wrapper settings, never passed to FIO.
The implementation fixes `overwrite=1` and `fallocate=none` in both phases and
includes that policy in scientific compatibility. Native JSON uses FIO's explicit
output file, separate from captured stdout/stderr. Runtime-cap validation allows
one second of native accounting tolerance; the pilot must confirm that tolerance
against the installed FIO version. Target diskstats snapshots retain the matching
device/partition row, not a full host telemetry log.

## 8. Pilot, checks and research acceptance

Before the full 700-measurement experiment:

1. Pilot one HDD and one NVMe with these same FIO options across all five workloads
   once using `--targets ... --pilot`. Record preparation time and per-workload
   wall time. Bootstrap preparation planning with a documented provisional estimate
   from available device throughput evidence, then replace it with observed timing;
   an unknown estimate is not silently replaced by the hard timeout.
2. Confirm full-file setup, reuse without truncation, existing-file overwrites,
   native JSON semantics, actual bytes/duration, latency units, backend activity,
   capacity snapshots and cleanup. Check whether the fastest runs are stable.
3. If repeat-to-repeat stability itself still needs a pilot, start a separate
   two-target full five-repetition run; do not treat the smoke pilot as variability
   evidence. Freeze the scientific protocol after the required checks.
4. Extend only demonstrated problematic cases
   through an explicit protocol revision, not an automatic runtime adjustment.
5. Supply planning estimates and run the fixed per-target matrix across as many
   reservations as necessary. Keep settings and monitoring consistent.

Local verification uses pure checks and fake processes, never benchmark I/O:
deterministic saved ordering, byte geometry, single-job generation, normal cap
completion versus hard timeout, live extension, interrupts and child cleanup.
Exercise resume after setup, between measurements, during writes, after raw output
but before completion, and after deletion but before cleanup checkpointing. Confirm
one preparation serves all 25 normal measurements, retained files survive stops,
valid results are never rerun, and timeout stops rather than retries the session.

Record raw diskstats before/after each measurement as coarse backend evidence and
note competing I/O. Device/controller caches, allocation history and short sampling
windows remain interpretation limits. No capacity-band study, queue-depth sweep,
simultaneous-target study or historical-results importer belongs in this runner.
Audit compatible historical evidence before scheduling supplementary work; missing
identity or tail-latency provenance cannot be reconstructed from bandwidth alone.

## 9. Results to produce

Keep analysis separate from the runner. Native FIO JSON and the manifest must
support this small, predefined report:

| Level | Required presentation |
|---|---|
| Target × workload | All five individual values plus median and min–max range for bandwidth and IOPS. Show points, not just an aggregate bar. |
| Measurement | Actual bytes, I/O count, FIO duration and byte/time completion reason; retain command wall time separately. |
| Latency | Native mean and selected completion-latency percentiles, with duration/I/O count available to judge short-run tails. Do not average percentiles into a purported pooled percentile. |
| Provenance | Host/target/device identity, scientific settings, session, saved execution order and preparation generation. |

Label units explicitly. Use FIO's native bandwidth/IOPS accounting, not bytes
divided by runner wall time. Report within-target spread separately from
between-target differences; do not pool unlike devices into five fictional
replicates. With five repetitions, avoid precise-looking significance claims.
This report characterizes the fixed bounded workload; it does not require an
SSD job that finished in two seconds to continue to sixty seconds.
