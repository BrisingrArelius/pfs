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

## Next steps

- User will provide a literature review covering this topic in a follow-up prompt.
- That literature will be used to set actual numeric thresholds (size cutoffs, ops/sec or ops-count cutoffs for frequency, etc.) for each dimension, replacing the ad hoc values in `profiles_backup.json`.
- Per the [Working Rules](CONTEXT.md#working-rules) in `CONTEXT.md`, any resulting change to `profiles_backup.json` (or a successor file) must be logged in `CHANGELOG.md`.
