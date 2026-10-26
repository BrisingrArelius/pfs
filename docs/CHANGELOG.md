# Changelog

Per the [Working Rules](research/CONTEXT.md#working-rules), every change to the codebase is logged here.

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

**File:** [`scripts/workloads/posix_synthetic_workload.c`](../scripts/workloads/posix_synthetic_workload.c)

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
