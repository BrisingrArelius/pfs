# Changelog

Per the [Working Rules](research/CONTEXT.md#working-rules), every change to the codebase is logged here.

---

## 2026-09-21 — Six-domain benchmark suite and execution requirements

- Revised experiment specifications and repository summaries: six plainly named
  benchmark domains, explicit communication/cache experiments,
  operation paths, separate application validation and full placement matrix.
- Specified Darshan coverage/raw evidence, durable progress across allocations,
  independent parser recovery and pilot-based full-matrix duration planning.
- Added [implementation backlog](IMPLEMENTATION_BACKLOG.md), separate from
  mechanical path repairs. Capacity meaning and numerical bands remain unresolved.
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

### Known follow-ups (not addressed here)

Two of these are open **decisions**, not just pending work — they change what
the `nd_strided` profile means scientifically, so they were deliberately left
to be made explicitly. Written up in full at:
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
- **`run_workloads.py` does not pass the new args.** The C binary is only
  invoked for nd_strided profiles (per its own warning message); the new
  parameters need plumbing through `build_workload_cmd()` to be settable from
  the JSON. This blocks both decisions above from taking effect.
- **Pre-existing, unrelated:** `O_DIRECT` requires `_GNU_SOURCE` on current
  glibc — the documented compile lines
  (`mpicc -O2 -o ... -ldarshan -lpthread -lrt -lz` in `run_workloads.py:250`,
  `gcc -O3 ...` in [CONTEXT.md](research/CONTEXT.md)) fail to compile on this machine
  without `-D_GNU_SOURCE`. Not introduced by this change and not fixed here.
