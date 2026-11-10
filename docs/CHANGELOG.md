# Changelog

Per the [Working Rules](research/CONTEXT.md#working-rules), every change to the codebase is logged here.

---

## 2026-11-10 — Proposed DLIO DXT phase study

- Removed repeated per-process Darshan configuration dumps (the configuration
  is already saved per attempt) and unused job-ID metadata parsing.
- Fixed a reproduced trace-loss bug: utility subprocesses inherited a rank's
  fixed `DARSHAN_LOGFILE`, wrote an empty log first, and prevented the rank
  from saving its real trace. Use native PID-specific filenames instead,
  handle flat/date-hierarchical logs, and validate helpers before excluding
  their non-workload logs. The preflight/regression now launches utility
  subprocesses before application I/O as well as forked readers. Workload
  settings and the installed Darshan versions are unchanged.
- Added measured POSIX single-write timestamp alignment when only a DXT clock
  marker is absent, with source/diagnostic reporting and unchanged strict
  workload coverage checks. Added `--recover --validate-only` to revalidate
  finished failed attempts in place without rerunning DLIO, moving pools or
  changing the saved workload fingerprint.
- Fixed preflight provenance collection for installed Darshan utilities that
  reject `--version`: preserve stdout/stderr/exit status and executable hash
  without failing an already validated rank/reader capture. Actual trace
  parsing and coverage checks remain strict; no Darshan installation changes.
- Implemented the isolated `scripts/workloads/dlio/profiling/` pipeline without
  modifying existing runners or DLIO workload settings: 12 one-invocation
  cases, per-process POSIX/DXT/reader coverage, traced clock anchors, passive
  host telemetry, ownership-aware cleanup, reservation continuation and
  fail-closed coverage checks. Added SQLite/CSV extraction, approved A/B/C
  charts and exact-event standalone HTML drill-down. No cluster workload
  execution during development; installed Darshan selection and explicit
  eight-rank/fork preflight are documented.
- Updated profiling to reuse existing server Darshan versions and earlier
  `DARSHAN_RUNTIME_LIB`/installation paths without rebuilds or PATH changes.
  MPI-linked runtimes are permitted when their MPI matches mpi4py; the real
  rank/fork/clock coverage preflight decides compatibility before full runs.
- Documented an approval-stage, instrumentation-only U-Net/CosmoFlow phase
  study: one invocation for each of three normal and three throttled
  configurations per workload (12 total), with all five epochs and eight
  ranks retained. No instrumentation or orchestration implementation yet.
- Added a clearly schematic chart mockup: semantic phase band, rank-level
  read/write timeline and request-size view on a shared time axis. Included
  DXT Explorer/PDSW references, trace-coverage requirements, metadata/cache
  attribution limits and microbenchmark-correlation methodology.
- Recorded user approval of all three proposed panels, with A (phase/activity)
  and C (request sizes) as primary explanatory views and B as rank-level
  supporting detail. Instrumentation implementation and cluster execution
  have not started.
- Critically reviewed the phase-study methodology without implementing it:
  corrected nested epoch/checkpoint accounting and rank-0 timer labelling,
  proposed passive phase-aligned traffic context, added historical cache
  comparability and tracing-coverage checks, and defined six falsifiable
  observation questions with a presentation/evidence strategy. Updated the
  schematic to show checkpoint timers inside the complete epoch spans.

## 2026-11-03 — Distinguish reported and inferred DLIO throughput

- Fixed managed command failure cleanup to terminate the local process group
  after a nonzero launcher exit, not only on timeout/interruption. A cluster
  MPICH Hydra proxy segfault left DLIO ranks alive after exit 144; the fix
  cleans up those descendants on future failures without changing workload
  parameters or claiming to fix the underlying MPI crash.
- Added reservation-aware ResNet `--all-pools --deadline 6h` and
  `--continue --deadline 6h` launches. Managed commands have bounded
  process-group termination; progress is saved per repetition, partial
  staging is reusable, interrupted attempts are preserved, and completed
  evidence is journalled for safe CSV reconstruction. A reserved cleanup
  interval restores pool 6 on a deadline or catchable interruption; failures
  are reported for inspection. Continuation restarts the partial repetition
  and skips completed ones, including across pool-6 restoration.
- Restricted managed ResNet to pools 14, 16 and 18 (4/0, 2/2, 0/4), nine
  full runs total. Continuation normalizes leftover target membership/caps
  back to original pool 6 before moving to the unfinished condition.
- Added legacy ResNet result adoption for `--continue` when no unfinished
  managed journal exists. Completed repetitions from the latest pool-14-led
  sweep are retained after evidence checks; partial repetitions restart and
  fully completed pool 14 is skipped. Already-running old launchers must be
  stopped before adoption and retain their original schedule until then.
- Replaced pool/per-run arithmetic mean epoch rates with end-to-end
  throughput for every workload: validated total processed sample count /
  full-run DLIO timestamp duration, including checkpoints. Removed the
  redundant whole-run-time panel and inferred pool-summary figure. Native
  epoch and dotted inferred byte-rate plots remain diagnostic views; CSVs
  retain sample counts and full durations for reproducing completion rates.
- Switched primary training plots in both DLIO analyzers to reported
  samples/s, averaging positive reported epochs per run and omitting
  zero-reported epoch points without interpolation. U-Net's available
  epochs are 2 and 4; these averages are not its zero-inclusive summary mean.
- Kept calculated MiB/s in separate dotted-line inferred figures and CSV
  fields. Regenerated normal and throttled U-Net galleries from copied runs.
- Set all DLIO throughput/loading-time plots to zero baselines with upper
  headroom and labelled U-Net's native pool comparison as epochs 2 and 4.
  The recorded first-epoch duration differs (129.25 s for all-fast versus
  386.52 s for all-slow, repetition 1), while reported epoch-2/4 durations
  remain about 82.36 s, explaining the different comparison outcomes.
- Added separate `pool_performance_inferred.png` figures for normal and
  throttled U-Net. Epoch CSVs now record configured sample bytes, processed
  sample count, nominal byte volume and the timing denominator/source, so
  inferred throughput can be reproduced without confusing it with physical
  file-read bandwidth or DLIO's reported samples/s.

## 2026-11-02 — Separate DLIO normal and throttled-SSD suites

- Collected the older POSIX synthetic workload, IOR wrapper, profiles, runners,
  compiled binary, and Darshan analysis under `scripts/workloads/legacy/`.
  Updated their repository-root path calculations and active entrypoint links;
  DLIO and LLM workload code remain separate.
- Moved the existing DLIO generator, workload runners, analyzer and shared
  profiles unchanged into `scripts/workloads/dlio/normal/`; updated their
  documented invocations. Specified a new four-target, four-stripe SSD
  speed-mix experiment in `docs/specs/DL_Throttled_SSD.md`.
- Added separate ResNet-50 and U-Net 3D runners in
  `scripts/workloads/dlio/throttled_ssd/`, using one shared helper to validate
  pool membership and configured device caps, stage identical inputs outside
  timing, and retain the same native DLIO evidence. Backend and client-network
  telemetry are required separately before claiming a device bottleneck.
- Recorded the pool 14–18 target moves and per-device cap commands in the
  throttled-SSD folder. Added `configure_pool.py` to perform one verified
  condition transition or restore pool 6 and clear the experiment's runtime
  limits; workload execution remains a separate command.
- Added an offline `plot_results.py` for completed throttled-SSD U-Net and
  ResNet sweeps. It accepts available pool conditions incrementally and plots
  training input, checkpoint writes, whole-run time, repetitions, and rank
  loading from three-run native and runner evidence without creating results
  READMEs.
- Corrected the throttled-SSD pool membership check so an empty pool's line
  cannot consume the following pool ID as a target. A pool-17 U-Net input copy
  failed with BeeGFS `Remote I/O error`; a later three-run sweep completed,
  but the earlier error still requires storage-side diagnosis.
- Added `--all-pools` to the throttled U-Net runner. From original pool 6 it
  now verifies the uncapped state, runs the five pool configurations with
  three repetitions apiece, and restores pool 6 only after full success;
  failures leave the current condition in place for inspection.
- Added a standalone `check_throughput.py` in the throttled-SSD code folder.
  It takes one bounded simultaneous sample of the four physical SSD read/write
  counters and displays their configured versus effective cgroup limits as
  four terminal rows; workload execution is unchanged.
- Extended the throughput checker with `--native --pool N`: a temporary
  four-stripe direct-I/O fio write/read through BeeGFS, reporting application
  and physical-device rates separately. The probe cleans up after success
  and runs only on a configured condition pool between DLIO measurements.
- Added pre-invocation `sync` and Linux page-cache drops over SSH on
  `colva1–4` to both normal and throttled-SSD DLIO runners, followed by the
  existing `anjuna2` cache drop. This applies to future runs, not previously
  collected evidence; failure on a storage host stops the run before DLIO.
- Found that colva1's target-104 partition path appeared in systemd but
  installed no `io.max` rule. Set its cap on the whole NVMe device instead,
  clear stale limit entries before each condition, and require all four
  physical-device read/write limits in runner preflight. The standalone
  probe now accepts the physical-device path as its configured limit.
- Added a 30-second wait after both server and client cache drops before
  each future DLIO invocation in the normal and throttled suites. Full-run
  checkpoint cleanup remains after evidence capture; pilots retain their
  checkpoint files for inspection.
- Added per-repetition snapshots of physical SSD read/write counters around
  each new throttled-SSD DLIO invocation. `device_io.csv` records bytes and
  whole-run rates per target after input staging and before checkpoint-file
  cleanup; delayed writeback can overlap. Older runs have no retrospective
  backend counts.
- Added a workload-specific throttled CosmoFlow runner using the same
  14–18 transitions, three repetitions per pool, cache preparation and
  pool-6 restoration as the U-Net runner. CosmoFlow dataset placement
  inspection samples 65 of its 8,192 files per condition; the offline
  throttled plotter accepts completed CosmoFlow sweeps.
- Added flushed per-run start/completion messages in both DLIO suites after
  successful evidence capture and checkpoint cleanup. The throttled suite
  also announces the start and end of first-time dataset copies.

---

## 2026-10-26 — Prepare narrow DLIO pool comparison

- Added an eight-rank ResNet-50 runner for three serial repetitions on each of
  the existing Default/HDD/SSD pools using the installed DLIO 2.0.0. It reads
  pre-generated datasets, records actual dataset/checkpoint OST placement,
  and collects native epoch timings and checkpoint durations with checkpoint
  rank-0 save rates derived from rank-0 output-file sizes (the only checkpoint
  rank timed by DLIO 2.0.0); all native outputs are retained.
- Checkpoint saves are explicitly labeled application-observed, not fsync or
  durable-throughput measurements: DLIO 2.0.0 has no working TensorFlow fsync
  setting. The runner does not pass unsupported configuration overrides and
  its folder-local usage notes distinguish this pilot from the global matrix.
- Aligned each measured invocation with the prior checkpoint-run preparation:
  `sync` and drop client caches before DLIO starts; remove only the newly
  created checkpoint directory after metrics and OST evidence are recorded.
- A cluster pilot showed stock DLIO 2.0.0's TensorFlow saver wrote only
  132-byte data files despite a 100-MB configured state. Added a local
  checkpoint hook that makes state trackable without editing DLIO; reject
  undersized files until a fresh cluster pilot verifies full payloads.
- The trackable-state pilot produced eight ~100-MB checkpoint files. Normalize
  DLIO 2.0.0's file-shuffle string to its enum in the local hook so seeded file
  order is actually seeded across comparable runs.
- Added workload-selectable DLIO data preparation and pilot/full-run commands
  for ResNet-50, U-Net 3D, and CosmoFlow. New inputs are generated in pool 1
  and copied to the other pools to keep dataset bytes identical; complete
  existing datasets are reused and partial ones stop preparation. The shared
  internal mechanics handle TensorFlow/PyTorch checkpoint files and bound CosmoFlow
  placement inspection to a recorded sample instead of 1.5 million queries.
  Kept all DLIO-owned scripts under `scripts/workloads/dlio/`. ResNet, U-Net,
  and CosmoFlow have separate small runners; the DLIO-imported checkpoint
  class lives in each runner rather than in a separate file. Added a folder-local
  README with the generate → pilot → full-run commands and metric definitions.
- Added color-coded Mermaid diagrams for the refactored ResNet runner call chain,
  the DLIO 2.0.0 TensorFlow TFRecord ingestion/checkpoint path, and the distinct
  owners of native DLIO JSON versus the runner's combined CSV. The diagrams
  distinguish the local refactor from a sweep already running from older code.
- Added an offline parser for a copied nine-run ResNet DLIO sweep. It validates
  native output against the runner CSV, extracts all rank/epoch load and process
  timings and checkpoint placement, and reports pool summaries without calling
  nominal training rate a measured POSIX/backend read rate. Flagged one native
  epoch with a 60-second pre-checkpoint timing gap; no benchmark was rerun.
- Added an offline DLIO plotter for per-epoch modeled training and wall time,
  rank-0 checkpoint saves, repetition trends, per-rank loading tails, and
  dataset OST file assignments. Rendered five figures from the copied sweep,
  with the anomalous epoch marked and per-run values separated from run-level
  mean/error bars.
- A five-epoch U-Net run exposed missing `block1.duration` and zero native
  throughput in odd epochs despite completed load steps on all ranks. Mark
  those training rates unavailable instead of zero I/O; accept the native
  epoch wall time separately. Kept the runner simple: there is no resume mode;
  a broken run can be repeated as a new sweep.
- Fixed pool-copy preparation after U-Net failed to initialize on pool 3:
  DLIO 2.0.0 lists `dataset/valid/` even with evaluation disabled, but the
  generator previously copied only `train/`. Create the empty directory in
  every pool and check for it before starting measured MPI runs; completed
  datasets remain reusable.
- Replaced the separate ResNet-only result parser and plotter entrypoints with
  one workload-selectable offline `analyze.py` command. It validates native
  output and runner evidence and writes tables and five figures in
  one invocation for ResNet, U-Net, or a completed CosmoFlow suite. U-Net's
  missing native odd-epoch rates remain unavailable instead of becoming zeros.
- Added a run-level three-metric pool comparison (nominal training input,
  rank-0 checkpoint save completion, and whole-run time), with three-run error
  bars. Published five curated figures and the
  comparison tables for each completed workload under
  `results/workloads/plots/dlio/{unet,resnet}/`.
- Simplified the generated plot sets to training and checkpoint throughput,
  pool performance, run-to-run throughput, and data loading time. Removed
  generated results READMEs and narrative reports and named each PNG for its
  measurement. U-Net's missing native rates are plotted only as explicitly
  estimated sample-size/time throughput, not as measured filesystem traffic.
- Reduced the CosmoFlow variant to 8,192 files per pool (1/64 of its
  DLIO preset). Both dataset generation and the pool runner now explicitly
  override DLIO's file count so generated inputs match measured runs; this is
  a reduced-scale variant, not the native full-size CosmoFlow workload.

---

## 2026-09-25 — Restrict microbenchmarks to layer effects

- Removed the standalone whole-system experiment and its plan, design,
  visualizer, inventory template, and synthetic tests. Retained the historical
  pool-management helpers as provenance, not live authority.
- Aligned the repository and Obsidian `MicroBenchmarks.md` specifications on five
  layer-focused domains. D/S/H whole-workload comparisons remain in the separate
  DLIO specification and global factor rules.

---

## 2026-09-25 — Visualize microbenchmarks directly from native evidence

- FIO and iperf3 visualizers now validate completed raw runs and create figures
  directly; removed their standalone intermediate-result parser commands without
  changing either raw benchmark runner. The available full runs produced ten FIO
  figures (separated by installed FIO version) and three iperf3 figures.
- Added folder-local raw-to-plot visualizers, offline plans, and synthetic
  evidence tests for cache, BeeGFS communication, and metadata.
  These three domains **still lack live cluster runners and reviewed pilots**.
- Plots are published as complete generations under each run's `plots/` directory,
  with the current paths in `plot_manifest.json`. Updated the guides and backlog
  to distinguish completed evidence from planned experiments.

---

## 2026-09-23 - Develop layer-effect microbenchmark designs

- Added designs for BeeGFS NetBench communication and controlled cache effects.
- Fixed each domain's workload geometry, matrix, state transitions, native
  evidence, telemetry, validation, durable resume, cleanup, and pilot gates.
- Kept capacity provisioning as a separate DLIO gate until site bands are
  approved.
- Changes in this entry are documentation only; no pool, cache, NetBench, IOR,
  FIO, iperf3, or mdtest operation ran.

---

## 2026-09-21 — Organize analysis by owning domain

- Moved Darshan parsing and workload-result analysis to
  `scripts/workloads/analysis/` and renamed `analysis.py` to
  `analyze_darshan.py`.
- Renamed `scripts/trace_analysis/` to `scripts/trace_characterization/` to
  distinguish external trace studies from benchmark-result analysis.
- Kept historical trace evidence under `results/trace_analysis/legacy/` and
  updated executable paths, defaults, and current documentation.
- Consolidated each analysis folder's guide as `README.md` and removed empty
  output/cache directories.

---

## 2026-09-21 — Layer-effects suite and execution requirements

- Revised experiment specifications and repository summaries for explicit
  communication/cache experiments, operation paths, and separate DLIO validation.
- Specified Darshan coverage/raw evidence, durable progress across allocations,
  independent visualization recovery and pilot-based duration planning.
- Added a work-tracking document at the time, separate from mechanical path
  repairs. Capacity meaning and numerical bands remained unresolved.
- Changes in this pass are documentation only.

---

## 2026-09-21 — Repair paths after repository reorganization

- Updated relocated scripts to find companion files from their installed
  locations instead of relying on the caller's working directory.
- Directed new benchmark, workload, analysis, checkpoint, and trace outputs to
  run-specific directories while keeping historical results read-only by default.
- Updated help text and documentation. Benchmark settings, pool IDs, formulas,
  filters, thresholds, cache behavior, and measurement logic were not changed.
- Verified with syntax checks, alternate-working-directory help commands, mocked
  Darshan traces, a workload dry run, and historical analysis inputs. No benchmark,
  pool-management, cache-drop, or cluster workload command ran.

---

## 2026-09-20 — Consolidate trace analysis and research documentation

- Moved contiguity/frequency tools under `scripts/trace_analysis/`, with Python
  source preserved byte-for-byte. Moved their historical CSVs/plots into
  `results/trace_analysis/legacy/`; merged an identical interrupted-session copy.
- Consolidated research notes under `docs/research/`, PDFs under
  `docs/references/`, and this changelog under `docs/`.
- Removed the empty root `r.txt` and generated Python bytecode caches; retained
  the separate nonempty FIO transcript and all source files.
- Updated documentation links and recorded trace-analysis path follow-ups in the
  path-repair checklist. No script logic or output defaults were edited.

---

## 2026-09-20 — Documentation and byte-preserving repository refactor

- Recovered missing tracked files from Git's index after interrupted OpenCode
  undo; retained existing staged renames and completed file organization.
- Consolidated FIO, pool helpers and placement/capacity analysis beneath
  `scripts/microbenchmarks/`; grouped workload orchestration and Darshan analysis.
- Moved historical measurements, plots, logs and checkpoints into `results/`;
  restored Apr-2 text summaries from `edc3d69` without regenerating derived data.
- Added migration/provenance documentation and a checklist for deferred
  input/output, script-lookup and working-directory repairs.
- Documented D/S/H, randomized/roundrobin, proposed capacity conditions, cache
  evidence, blocked repetitions and NetBench interpretation.
- Script/configuration logic is unchanged. No benchmark or pool commands ran.
  See [migration record](MIGRATION.md) for recovery gaps and validation.

---

## 2026-08-03 — Fix `nd_strided` access pattern collapsing into sequential I/O

**File:** [`scripts/workloads/legacy/posix_synthetic_workload.c`](../scripts/workloads/legacy/posix_synthetic_workload.c)

### What was wrong

The `PATTERN_ND_STRIDED` offset generator derived its row width from the row
pitch:

```c
long cols = row_stride / p->op_size;      /* row width taken FROM the pitch */
offset = row * row_stride + col * op_size;
```

Because `cols` was defined as `row_stride / op_size`, the row-major traversal
collapses algebraically to `offset = idx * op_size` — i.e. **literally
sequential I/O**, not merely something resembling it. No hole ever existed
between one row and the next, so no profiler (Darshan or otherwise) could
distinguish the pattern from a `contiguous` profile.

The old code only escaped this via an alternating row-major/column-major flip
that triggered once per full `rows × cols` grid pass. Since `num_ops` is fixed
(50,000) while the grid scales with file size, the column-major regime was
**never reached at 1 GB or 10 GB** — those variants profiled as 100%
sequential. Only the 0.1 GB variant produced any stride signal at all.

### What changed

- **Decoupled block length from row pitch.** New optional `block_size`
  parameter = contiguous bytes touched per innermost run, required to be
  `< stride_size` so a genuine `stride_size - block_size` byte hole remains
  between rows. This is the actual fix.
- **Generalized to N dimensions.** New optional `nd_dims` parameter (2–5,
  default 2), replacing the hardcoded 2D row/col grid. Modeled on HDF5
  hyperslab / PnetCDF `start/count/stride` selections: dim 0 is a contiguous
  run, each outer dim jumps past the span below it scaled by the same sparsity
  ratio. `ND_MAX_DIMS = 5` matches Darshan's own `H5D_MAX_NDIMS` /
  `PNETCDF_VAR_MAX_NDIMS` ceiling.
- **Removed the row-major/column-major alternation entirely** — it was a
  workaround for the missing hole, and it made the pattern's Darshan signature
  depend on file size. Replaced with a plain odometer over the dimensions
  (`nd_offset()`), which yields a stable signature at any file size.
- **Geometry precomputed once per file** (`nd_geometry_init()`) rather than
  recomputed per op; threaded through `do_write_phase` / `do_read_phase` /
  `calculate_offset` as a new `const NdGeometry *nd` parameter.
- **Added validation** rejecting `block_size >= stride_size` (the old
  degenerate case), `stride_size <= op_size`, and out-of-range `nd_dims`.
- `int_nth_root()` used instead of `pow()` so the existing `mpicc -O2` compile
  line still works without `-lm`.

Both new CLI args are **optional and backward compatible** (`argv[12]`,
`argv[13]`); existing 11-arg callers get `block_size = stride_size / 2` and
`nd_dims = 2`.

### Verified

Compiles clean under `mpicc -O2 -Wall -Wextra`. Offset deltas measured against
the real (not copied) static functions, `op_size=4096, stride_size=262144,
block_size=131072, 50,000 ops`:

| Config | Darshan-visible deltas | Consecutive |
|---|---|---|
| 2D @ 0.1 GB | 4096 (96.9%), 135168 (3.1%) | 96.9% |
| 2D @ 1 GB | 4096 (96.9%), 135168 (3.1%) | 96.9% |
| 2D @ 10 GB | 4096 (96.9%), 135168 (3.1%) | 96.9% |
| 3D @ 10 GB | + 75108352 | 96.9% |
| 4D @ 10 GB | + 11407360 | 96.9% |

The row-boundary delta matches theory exactly:
`stride_size - block_size + op_size = 262144 - 131072 + 4096 = 135168`.
The signature is now **identical across all three file sizes** (previously
1 GB and 10 GB showed no stride signal at all), and distinct-delta count stays
within Darshan's four `POSIX_STRIDE*_STRIDE` slots through 4D.

### Unresolved state recorded with this change

The change left the following items unresolved because they alter the scientific
meaning of `nd_strided`. Their historical discussion is recorded at:
[`CONTEXT.md` § Open Decisions](research/CONTEXT.md#open-decisions--nd_strided-workload-generator-as-of-2026-08-03)
(project-wide framing) and
[`Task 1 - Profiles Setup.md` § Open decisions](research/Task%201%20-%20Profiles%20Setup.md#open-decisions--nd_strided-profile-parameters-as-of-2026-08-03)
(profile-parameter framing).

- **`block_size` value is a research-semantics call.** `profiles_backup.json`
  does not set it, so it inherits `stride_size / 2` → **96.9% consecutive**,
  which a contiguity-first classifier using the ≥95% threshold from
  [`LitReview_task1.md`](research/LitReview_task1.md) Part D would still label
  `contiguous`. Either lower it (e.g. `16384` → 75%) or classify stride-first.
- **`nd_dims` (2–5) is configurable but unused** — every profile is 2D. Open
  question whether the top-20 set should carry distinct 3D/4D variants.
- **`run_workloads.py` does not pass the new args.** `block_size` and `nd_dims`
  are therefore not configurable from the JSON.
- **Pre-existing, unrelated:** `O_DIRECT` requires `_GNU_SOURCE` on current
  glibc — the documented compile lines
  (`mpicc -O2 -o ... -ldarshan -lpthread -lrt -lz` in `run_workloads.py:250`,
  `gcc -O3 ...` in [CONTEXT.md](research/CONTEXT.md)) fail to compile on this machine
  without `-D_GNU_SOURCE`. This was not introduced or fixed by that change.

These notes record the implementation state at the time.
