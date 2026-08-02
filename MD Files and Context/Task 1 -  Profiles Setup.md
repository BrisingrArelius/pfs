# Task 1 — Profiles Setup

## Goal

Come up with the **top 20 most common/prevalent file I/O profiles in HPC**, stylistically similar to the existing [`scripts/workloads/profiles_backup.json`](../scripts/workloads/profiles_backup.json), but this time with each parameter/threshold backed by literature or prior experimental results — not chosen arbitrarily.

## Problem with the current `profiles_backup.json`

The 20 profiles currently in that file were defined **without a documented research basis**. Sizes, op counts, and thresholds were picked ad hoc. The task now is to go back and determine, with literature backing, what actually defines/classifies each I/O behavior (e.g., where the "small vs. large" boundary really sits, what counts as "frequent" vs. "seldom", etc.).

## Parameter dimensions used in the existing profiles

| Dimension | Values Used | HPC Examples |
|---|---|---|
| Size | Small: 4KB, Large: 4MB+ | PFS stripe threshold (prior discussion) |
| Frequency | Seldom: <5K ops, Freq: >20K ops | Darshan histograms |
| Direction | Read (1.0), Write (0.0), Mixed (0.5) | POSIX_BYTES_READ/WRITTEN ratios |
| Spatial | contiguous, random, strided (±), nd_strided | HDF5 hyperslabs, VPIC metadata |
| Phases | 1–4 | Checkpoint periodicity |

## Parameters of interest for the re-derivation (this task)

1. **I/O size:** small vs. large
2. **I/O access pattern:** contiguous vs. random vs. strided vs. nd_strided
3. **I/O type:** read-heavy vs. write-heavy vs. mixed
4. **I/O access temporal frequency:** seldom vs. frequent

Open question raised to Claude: whether there's an important parameter missing from this list (per literature) — still to be resolved. Candidates worth checking against the literature review (not yet confirmed): number of files/processes sharing a file (SSF vs. FPP vs. partial-shared, per the [Summer Plan](CONTEXT.md#overall-plan--summer-plan) notes), request/operation alignment to filesystem block or stripe boundaries, burst vs. sustained I/O phases, and metadata-op-heavy vs. data-op-heavy workloads.

## Constraint: thresholds must be derivable from Darshan POSIX/MPI-IO counters

All thresholds must be groundable in **Darshan POSIX/MPI-IO counters**, since that's what this project's pipeline actually parses from real logs — a literature threshold that can't be computed from a Darshan counter is unusable downstream. Rough mapping:

- **Size:** `POSIX_BYTES_READ/WRITTEN` ÷ `POSIX_READS/WRITES` (mean access size), or Darshan's `POSIX_SIZE_READ_*`/`POSIX_SIZE_WRITE_*` histogram buckets.
- **Frequency:** total `POSIX_READS`/`POSIX_WRITES` op counts.
- **Access pattern:** `POSIX_SEQ_*`, `POSIX_CONSEC_*`, `POSIX_STRIDE1_STRIDE4`+ counters.
- **Type (read/write/mixed):** `POSIX_BYTES_READ` vs. `POSIX_BYTES_WRITTEN` ratio.
- **Sharing pattern (SSF/FPP)**, if added: rank-level access counts (MPI-IO module).

If a literature threshold relies on data Darshan doesn't expose (e.g. raw latency), flag it as non-actionable rather than adopting it silently.

## HDF5/PnetCDF-derivable classification heuristic for `nd_strided`

The Constraint above notes `POSIX_STRIDE1_STRIDE4`+ as the access-pattern
grounding, but those POSIX counters are flat (one stride value per record) and
can't actually distinguish `nd_strided` from `strided` — see
[`LitReview_task1.md`](LitReview_task1.md) Part C's "genuine gap" finding.
Darshan's HDF5/PnetCDF modules close part of that gap: `H5D_COUNTERS`
(per-dataset) and `PNETCDF_VAR_COUNTERS` (per-variable) both carry real
per-dimension stride/length data across up to 5 dimensions
(`H5D_DATASPACE_NDIMS`, `H5D_ACCESS{1-4}_STRIDE_D{1-5}`/`_LENGTH_D{1-5}`,
`H5D_REGULAR_HYPERSLAB_SELECTS`, and the PnetCDF equivalents) — verified
directly against the Darshan source headers. Full derivation and the
counter-availability table (including why raw MPI-IO and raw POSIX are *not*
covered) are in
[`CONTEXT.md`](CONTEXT.md#darshans-dimensional-visibility-for-nd_strided-is-narrower-than-hdf5-or-pnetcdf).

Concrete heuristic:

```
is_multidim    = H5D_DATASPACE_NDIMS >= 2
uses_hyperslab = H5D_REGULAR_HYPERSLAB_SELECTS > 0
strided_dims   = count of Di in 1..NDIMS where STRIDE_Di > LENGTH_Di

nd_strided  ⇐  is_multidim AND uses_hyperslab AND strided_dims >= 2
strided     ⇐  strided_dims == 1
contiguous  ⇐  strided_dims == 0
```

**Caveat:** only actionable for files whose I/O went through Darshan's HDF5 or
PnetCDF instrumentation — raw MPI-IO (derived datatypes) and raw POSIX I/O
still fall back to the flat `POSIX_STRIDE1_STRIDE4`+ proxy, with no way to
confirm genuine dimensional structure.

## Open decisions — `nd_strided` profile parameters (as of 2026-08-03)

These block the `nd_strided` entries in the top-20 profile set specifically.
Background: the `nd_strided` generator in `posix_synthetic_workload.c` was
found to be emitting *literally sequential* I/O at the 1 GB and 10 GB variants
and was fixed on 2026-08-03 (details in [`CHANGELOG.md`](CHANGELOG.md), broader
framing in [`CONTEXT.md`](CONTEXT.md#open-decisions--nd_strided-workload-generator-as-of-2026-08-03)).
The fix added two new profile parameters that the JSON does not yet set.

### 1. New parameter: `block_size` — and where to put its threshold

`block_size` = contiguous bytes touched per innermost run, required to be
`< stride_size` (that gap is what makes the access strided at all). It is a
**new parameter dimension not present in the table above**, and like every other
threshold in this task, its value should be chosen deliberately rather than
inherited from a default.

It directly sets how contiguous the profile looks to Darshan:

| `block_size` | ops/row | Consecutive ratio | Survives the Part D ≥95% contiguity test as non-contiguous? |
|---|---|---|---|
| 131072 *(current default = `stride_size/2`)* | 32 | 96.9% | **No** — classified `contiguous` |
| 65536 | 16 | 93.8% | Yes |
| 16384 | 4 | 75.0% | Yes |
| 8192 | 2 | 50.0% | Yes |

**The tension:** [`LitReview_task1.md`](LitReview_task1.md) Part D established
empirically (12,301 real ALCF Polaris logs) that files with `total_ops ≥ 20`
are 98.11% contiguous, motivating a ≥95% "sequential" threshold. But a genuine
HDF5 hyperslab read *is* mostly contiguous within each row — high contiguity
and nd_strided structure are not mutually exclusive in reality. So either:

- **(a)** pick a smaller `block_size` so the profile is non-contiguous under any
  rule, or
- **(b)** keep it and make the classifier check for a **secondary stride before**
  applying the contiguity test.

Note this is the same "no validated threshold exists" problem Part C documented
for strided/nd_strided generally — only now it surfaces as a *workload
generation* parameter rather than a *classification* one. Whichever is chosen
should be documented as this project's own construction, consistent with how
Part C's verdict says the `POSIX_STRIDE*` heuristic must be labeled.

### 2. New parameter: `nd_dims` — should the profile set include 3D/4D variants?

`nd_dims` (2–5, default 2) now controls dimensionality. Per Part C's literature
(Kang et al. 2021 on E3SM; HDF5/PnetCDF hyperslab APIs), real HPC nd_strided
data is overwhelmingly **2D–4D** — 3D volumetric simulation (combustion, CFD,
cosmology), 4D time-resolved climate/weather. 5D is the practical ceiling, and
also happens to be Darshan's own hard limit (`H5D_MAX_NDIMS` /
`PNETCDF_VAR_MAX_NDIMS` = 5).

Open question for the top-20 set: does `nd_strided` stay a single 2D profile, or
split into distinct 2D/3D/4D variants? Each added dimension consumes one more of
Darshan's four `POSIX_STRIDE*_STRIDE` slots, so a 4D profile leaves no headroom
for anything else in the stride counters.

### 3. Blocker: parameters are not yet wired through

`run_workloads.py`'s `build_workload_cmd()` does not pass `block_size`
(argv[12]) or `nd_dims` (argv[13]) to the C binary, so both currently fall back
to defaults regardless of what the JSON specifies. Decisions 1 and 2 cannot take
effect until this is plumbed.

## Next steps

- User will provide a literature review covering this topic in a follow-up prompt.
- That literature will be used to set actual numeric thresholds (size cutoffs, ops/sec or ops-count cutoffs for frequency, etc.) for each dimension, replacing the ad hoc values in `profiles_backup.json`.
- Per the [Working Rules](CONTEXT.md#working-rules) in `CONTEXT.md`, any resulting change to `profiles_backup.json` (or a successor file) must be logged in `CHANGELOG.md`.
