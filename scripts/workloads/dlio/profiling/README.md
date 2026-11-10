# DLIO phase profiling

Instrumentation-only companion to the existing runners. It does not change
their files or the DLIO reader/checkpoint implementations. The launcher
preserves eight ranks, five epochs, dataset sizes, seeds, batching, threads,
prefetch and all-rank checkpoints after epochs 2/4. It runs **one invocation**
for each selected configuration, in a separate output tree.
Keep that tree on client-local storage: preflight rejects BeeGFS/NFS and
other known network filesystems for trace output, avoiding trace-write traffic
being confused with application traffic. Use `--output /local/path/profile-ID`
if `~/tejas` is network-mounted. Datasets/checkpoints remain on BeeGFS as before.

| Workload | Normal | Throttled | Invocations |
|---|---|---|---:|
| CosmoFlow | pools 1, 3, 6 | pools 14, 16, 18 (4/0, 2/2, 0/4) | 6 |
| U-Net | pools 1, 3, 6 | pools 14, 16, 18 (4/0, 2/2, 0/4) | 6 |

CosmoFlow is collected first when both are selected. Original performance
results remain separate; a profiled invocation is not another independent
performance repetition. Tracing can affect memory usage and runtime.

## Runtime requirements

On `anjuna2`, use the existing DLIO 2.0.0/MPICH/mpi4py environment, existing
datasets, and passwordless SSH/noninteractive sudo used by the current cache
and pool scripts. The passive inventory probe additionally requires
`sudo -n python3` on colva1–4 to read storage-directory/target mappings.
Use an exclusive reservation: do not drop caches or move targets alongside
another application experiment.

Required tools: `darshan-parser`, `darshan-dxt-parser`, `ip`, `findmnt`,
`beegfs-ctl`, Python 3.10+, PyYAML and matplotlib. HTML detail is standalone
SVG/JavaScript; no Plotly, dashboard or web server is needed.

**Use the existing server Darshan installation and utilities.** No rebuild,
upgrade, PATH replacement or system configuration edit is required or done.
Runtime selection uses, in order, `--darshan-library`, `PFS_DARSHAN_LIBRARY`,
the older scripts' `DARSHAN_RUNTIME_LIB`, then known installed paths including
`~/darshan-nompi/lib/libdarshan.so`, `/opt/darshan-nompi/lib/libdarshan.so`
and `/usr/local/lib/libdarshan.so`. The selected path is printed.

An existing non-MPI runtime is preferred for reader-child coverage. An
MPI-linked runtime is not rejected just for its build type, but its MPI must
match the library actually used by `mpi4py`; DLIO retains its existing MPICH.
That match is necessary, not sufficient: some MPI builds cannot capture
forked readers even if earlier MPI-only jobs worked. The tiny preflight must
prove per-process/rank/child capture and clock coverage. A failure stops
before full workloads or pool changes; select another already installed
compatible runtime rather than changing software.

`DARSHAN_ENABLE_NONMPI=1` requests per-process capture so reader children
are not silently omitted. MPI-IO tracing is not claimed. All tracing flags,
memory quotas and path filters apply only to this job's processes/config file;
they do not edit the installed Darshan configuration.

If automatic selection does not match your previous job's library, select
that **already installed** file explicitly:

```bash
export DARSHAN_RUNTIME_LIB=/path/used/by/your/previous/job/libdarshan.so
```

Do not export `LD_PRELOAD` globally: the launcher injects it only into rank
executables. It chains an existing Python `sitecustomize`, adds process/clock
markers, and preserves the original application entry point.

## Commands (repository root)

Print the matrix, with no writes, dependency checks, SSH or workloads:

```bash
python3 scripts/workloads/dlio/profiling/run_profiles.py --plan
```

Optional instrumentation preflight, no DLIO or pool changes:

```bash
python3 scripts/workloads/dlio/profiling/run_profiles.py --preflight
```

The preflight uses eight MPI ranks and one tiny forked reader per rank,
checks MPI rank identity, and validates real POSIX/DXT counts, bytes and
clock markers. It is not a training benchmark. The same preflight runs
automatically before a full study, before any pool moves.

Start the full 12-invocation study, from original pool 6 with no caps:

```bash
python3 scripts/workloads/dlio/profiling/run_profiles.py \
  --workload both --deadline 5h
```

Or use `--workload cosmoflow` / `--workload unet` for six invocations.
`--darshan-library /absolute/path/libdarshan.so` overrides the environment.
The launcher prints the new `~/tejas/dlio/profiling/profile-...` directory.

Continue that exact study during the next reservation:

```bash
python3 scripts/workloads/dlio/profiling/run_profiles.py \
  --continue "$HOME/tejas/dlio/profiling/profile-TIMESTAMP" --deadline 5h
```

Completed cases are skipped; partial attempts are retained and restarted
from the beginning. The saved matrix wins over `--workload` on continuation.
Source/runtime fingerprints and memory settings must still match. Changing
instrumentation memory/source settings requires a new study, not silently
mixing different profiling conditions. Existing result directories are never
adopted or overwritten.

The deadline includes staging, preflight and coverage parsing; up to the
last minute is reserved for stopping owned processes and restoring pool 6.
Ctrl-C/SIGTERM/SIGHUP and failures take the cleanup path. Owned processes
are checked by the study's environment tag even if the MPI launcher died.
Targets are not moved while tagged readers remain alive. A failed restoration
is recorded/reported, not assumed successful. SIGKILL cannot run cleanup;
continuation first checks for leftover owned processes. Keep the state/logs.

## What is added around the existing workload

```text
original preparation/placement -> sync + server/client cache drops -> 30s wait
                                    |
                             passive host samplers
                                    |
                        same mpirun -np 8 DLIO settings
                                    + Darshan POSIX/DXT per process
                                    + PID-to-rank/worker and clock markers
                                    |
                    native DLIO output + binary traces + counters
                                    |
                      coverage validation -> next configuration
                                    |
                         stop samplers + restore pool 6
```

No cache drops between epochs. Data validation/layout queries precede the
last cache drop. Instrumentation logs go outside the BeeGFS dataset and
checkpoint paths. Application checkpoint contents are validated and their
sizes/layouts saved before successful-run cleanup, as in the original suite.

DXT defaults to 64 MiB per process and aggregate modules to 32 MiB;
`--dxt-mib`/`--module-mib` record and control those budgets. No truncation is
accepted: matching per-file aggregate/DXT counts and bytes are required.
Per-process clocks use bracketed traced writes at start/end, not coarse log
start times. Source-host timezone offset is recorded for native DLIO clocks.

Passive samplers use one persistent Python process per involved host, with
live target-to-device mapping and routed network interfaces. They record
cumulative bytes, memory and client process context every second by default
(`--sample-seconds`). Devices are deduplicated, not added to their partitions.
Clock calibration/uncertainty and sampler failures are retained. These are
host-wide observations; interface routes are not proof of active BeeGFS
transport or exclusive application traffic.

## Offline analysis and diagrams

Copy the complete profile directory to the local machine. The analysis
requires the Darshan utilities locally but no DLIO, MPI launch, SSH or sudo:

```bash
python3 scripts/workloads/dlio/profiling/analyze_profiles.py \
  results/workloads/runs/profile-TIMESTAMP
```

It revalidates completed cases, then produces within each case's `analysis/`:

- `coverage.json`: coverage result, per-process clock error and native full-run metrics.
- `events.sqlite`: exact file/PID/rank/thread, offset, length and start/end data.
- `phases.csv`: non-overlapping complete epochs/gaps with actual API counts,
  bytes, p50/p95/modal/max sizes, file/process counts, crossing calls and activity unions.
- `files.csv`: per-process file counts/bytes and whole-process metadata counters.
- `file-sharing.csv`, `streams.csv`: rank/process sharing and observed per-stream
  offset transitions; missing thread identity is explicitly flagged.
- `phase_access.png` / `.svg`, `overview.html`: approved A/B/C views.
- `size-bins.csv`, `rendering.json`: exact bin/count/area definitions.
- `telemetry-phases.csv`: device/network deltas from wholly contained sampling
  intervals, with observed duration, phase coverage and clock uncertainty.
  Boundary-straddling intervals are excluded, not interpolated into fake bytes.

The study-level `analysis/` holds the two absolute-time configuration comparisons.
Native checkpoint markers are **inside** complete epochs 2/4 and labelled
rank 0. All-rank write activity comes from DXT. Static rank lanes merge
overlapping same-operation calls; exact request intervals remain queryable.
Size glyphs use one shared call-count scale across the study, with a disclosed
minimum visible area. Axes start at zero; main comparisons are not normalized.

Exact-event detail for one case attempt, with request hover and browser zoom:

```bash
python3 scripts/workloads/dlio/profiling/analyze_profiles.py \
  results/workloads/runs/profile-TIMESTAMP/cosmoflow-throttled-pool18/attempt-TIMESTAMP \
  --detail --rank 0 --start 100 --end 110
```

Add `--file /exact/dataset/file` if useful. Open `analysis/detail.html`.
Queries above 50,000 events are rejected until narrowed, not silently sampled.

## Interpretation and limits

Use the [reviewed study plan](../../../../docs/specs/DLIO_DXT_Phase_Study.md)
for the observation questions and matching existing microbenchmark evidence.
There is no automatic causal classifier: file count does not prove a metadata
bottleneck, white timeline gaps are not compute/metadata measurements, and
DXT bytes are not physical-device bytes. Native checkpoint timers do not
measure durable all-rank completion. No fsync is added.

The historical U-Net normal results have a cache-protocol provenance caveat.
Do not label every difference from old timings as Darshan overhead. Keep
one-run characterization distinct from the old repeated performance results.

## Local verification

```bash
python3 -m unittest discover -s scripts/workloads/dlio/profiling/tests -v
```

Tests cover settings equivalence against saved U-Net/CosmoFlow configs,
phase/checkpoint accounting, coverage and clock failures, restoration/resume,
telemetry boundary accounting, and diagrams. When local Darshan is available,
a tiny stdlib Python/fork test uses real binary logs; it does not run MPI,
DLIO or any cluster command. Full MPICH/DLIO compatibility is checked by the
explicit cluster preflight and first-case coverage gate, not claimed by the
  local tests.
