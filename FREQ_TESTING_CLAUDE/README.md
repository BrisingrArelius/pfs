# I/O rate distribution — real ALCF Polaris Darshan logs

## How to run this pipeline

Two steps, run from inside this directory (`FREQ_TESTING_CLAUDE/`).

**Prerequisites:**
- `darshan-parser` binary — defaults to `/home/advay/darshan/bin/darshan-parser`;
  override with `--parser <path>` if it lives elsewhere.
- `python3` with `matplotlib` installed (standard library covers everything else:
  `csv`, `argparse`, `multiprocessing`, `subprocess`, `math`).
- Read access to a directory tree of `.darshan` log files. The analysis in this
  README used `/home/advay/darshan_logs_analysis/2024-20260128T084726Z-3-001/2024/`
  (14,900 logs) — point at a different tree to run against different data.

**Step 1 — parse the logs into a per-file CSV** (this is the slow step; it shells
out to `darshan-parser` once per log file, parallelized across all CPU cores by
default):

```bash
cd FREQ_TESTING_CLAUDE
python3 parse_logs.py /home/advay/darshan_logs_analysis/2024-20260128T084726Z-3-001/2024/ output/per_file_rate.csv
```

Optional flags: `--parser /path/to/darshan-parser` (custom binary path), `--jobs N`
(worker count, default = all cores). Progress is printed every 1,000 logs; on the
14,900-log corpus this took a few minutes and produced 162,722 rows.

**Step 2 — bin the rates and compute the seldom/frequent splits.** Run twice: once
unfiltered, once with the `total_ops ≥ 20` floor (same stability floor Part D
established for `contig_ratio`) — both are referenced in the findings below, so both
are needed to reproduce this README's numbers:

```bash
python3 analyze_distribution.py output/per_file_rate.csv output
python3 analyze_distribution.py output/per_file_rate.csv output --min-ops 20
```

This prints the distribution tables and split tables to stdout, and writes to
`output/`: `rate_active_distribution[_minops20].png`,
`rate_wallclock_distribution[_minops20].png`, one overlay PNG per metric × split
method (`<metric>_<median|tercile|quartile>_split[_minops20].png`, 12 total), and
`rate_splits[_minops20].csv` (the combined numeric split summary).

To reproduce this README's numbers exactly, use the same log corpus path above —
results (especially the Finding 3 peak) are Polaris/corpus-specific and will differ
on a different log tree.

## Overview

Part E of the Task 1 threshold work (see
[`../MD Files and Context/LitReview_task1.md`](../MD%20Files%20and%20Context/LitReview_task1.md)),
parallel in method to Part D's contiguous-ratio analysis in
[`../CONTIG_TESTING_CLAUDE/`](../CONTIG_TESTING_CLAUDE/README.md). This directory is
fully standalone — it does not import, edit, or depend on anything in
`CONTIG_TESTING_CLAUDE/`.

The lit review flags **frequency** as the weakest-supported profile dimension: no
paper reviewed gives a flat op-count cutoff for "seldom" vs. "frequent" like the
current `profiles_backup.json` (`<5K ops seldom / >20K ops frequent`); the papers that
touch frequency at all measure it as a *rate* (ops/sec, burstiness ratio, Hurst
parameter), and even those give no numeric split point. Since Darshan's standard
POSIX summary counters can't reconstruct burstiness/Hurst (those need per-request
timestamp series, i.e. DXT trace data), this analysis computes the two rate
formulas that *are* derivable from standard counters, then derives a seldom/frequent
split empirically — the same approach Part D used when literature came up empty for
the contiguity threshold.

## Data source

Same corpus as Part D: `/home/advay/darshan_logs_analysis/2024-20260128T084726Z-3-001/2024/`
— **14,900** real ALCF Polaris `.darshan` logs, covering the same 9-day partial slice
(2024-04-24–30, 2024-05-11/12). Reusing this corpus keeps Part D and Part E directly
comparable (same jobs, same filesystems). **162,722** file records extracted — matches
Part D's count exactly, as expected (same per-record grain, same corpus).

## Method

Per file record (aggregated across MPI ranks, same `record_id`-keyed aggregation as
Part D — safe because Darshan never keeps both per-rank and shared-aggregate records
for the same file):

```
rate_active    = (POSIX_READS + POSIX_WRITES) / (POSIX_F_READ_TIME + POSIX_F_WRITE_TIME)
rate_wallclock = (POSIX_READS + POSIX_WRITES) / (POSIX_F_CLOSE_END_TIMESTAMP - POSIX_F_OPEN_START_TIMESTAMP)
```

`POSIX_READS`/`POSIX_WRITES`/`POSIX_F_READ_TIME`/`POSIX_F_WRITE_TIME` are summed
across ranks (consistent with how op counts are aggregated in Part D).
`POSIX_F_OPEN_START_TIMESTAMP` takes the **min** across ranks (earliest open) and
`POSIX_F_CLOSE_END_TIMESTAMP` the **max** (latest close) — summing timestamps, unlike
summing counts/cumulative-time, would be meaningless. Records with an undefined
denominator (zero active time, or zero/negative wallclock span) are excluded from
that metric rather than emitting `inf`/negative rates. Manually verified against raw
`darshan-parser` output on sample records before the full run (see `parse_logs.py`'s
docstring for the aggregation-strategy table).

## Files

- `parse_logs.py` — walks a log directory, runs `darshan-parser` on every `.darshan`
  file, computes `rate_active`/`rate_wallclock` per file record, writes one row per
  (log, file) to a CSV. Parallelized across CPU cores. Standalone, new script.
- `analyze_distribution.py` — bins each rate metric into log10-scale buckets
  (`<1, 1-10, 10-100, ..., >1M` ops/sec — rate is unbounded/heavy-tailed, unlike
  Part D's bounded `[0,1]` ratio, so linear 5%-wide bins don't apply here), prints the
  distribution table, and computes **three candidate seldom/frequent splits per
  metric** — median (50th pct), tercile (33rd/67th pct), quartile (25th/75th pct) —
  each with its own overlay plot showing the cut points on the histogram.
  `--min-ops N` filters low-op-count files first (same quantization concern Part D
  documented: a rate computed from 1-2 ops is not a stable estimate).
- `output/` — generated CSVs and PNGs (run output, not committed logic).

## Finding 1: `rate_wallclock` is unusable for most of the corpus

Of 162,722 file records, **141,879 (87.2%) have an undefined `rate_wallclock`**
(zero or negative open-to-close span) — vs. **0 undefined** for `rate_active`. This
means the wallclock metric silently drops the overwhelming majority of files before
any distribution/split analysis even starts (unfiltered n=20,843 vs. 162,722; at
`total_ops≥20`, n=2,703 vs. 61,395). At Polaris's timestamp resolution, most files are
opened and closed too quickly relative to Darshan's timestamp precision to yield a
positive span — this is a real measurement-floor artifact, not a data quality bug.
**Practical implication: `rate_active` is the metric with actual coverage; treat
`rate_wallclock` numbers below as a much smaller, likely biased subsample** (files
open long enough to have a measurable span skew toward longer-running/slower jobs),
not a like-for-like alternative.

## Finding 2: raw distributions (log10-scale, unfiltered, n=162,722 / n=20,843)

**`rate_active`** — right-skewed but tightly concentrated: 53.04% of files fall in the
10K–100K ops/sec bucket, 27.96% in 1K–10K. Mean 24,645 ops/sec, median 25,258 ops/sec.

| Bucket (ops/sec) | % of files |
|---|---|
| <1 | 0.16% |
| 1–10 | 1.89% |
| 10–100 | 7.57% |
| 100–1K | 7.77% |
| 1K–10K | 27.96% |
| 10K–100K | 53.04% |
| 100K–1M | 1.61% |
| >1M | 0.00% |

**`rate_wallclock`** (n=20,843, the reduced subsample per Finding 1) — much more
spread out: 37.05% in 10–100, 28.81% in 100–1K. Mean 2,965 ops/sec, median 60.65
ops/sec — an order of magnitude lower than `rate_active`'s median, consistent with it
including idle/gap time that `rate_active` excludes.

## Finding 3: filtering to `total_ops ≥ 20` collapses `rate_active` into a sharp peak — investigated, looks like a real signal

At `total_ops ≥ 20` (n=61,395, same floor Part D established), **94.88% of files land
in the 10K–100K bucket** for `rate_active` — even tighter than the unfiltered
distribution. Zooming into that bucket at finer (2K-wide) resolution shows a sharp
mode around **30K–36K ops/sec** specifically: 42,722 of 61,395 filtered files
(69.6%) fall in the 28K–40K ops/sec range alone.

Traced before trusting it (same approach Part D used for its 80–85% contiguity
cluster): this peak is **not** a single-job or single-day artifact —

- **Filesystem-concentrated**: 42,712 of 42,722 peak files (99.98%) are on
  `/lus/eagle`, vs. 60,430/61,395 (98.4%) for the filtered set overall — slightly
  higher concentration but not exclusively eagle-driven in a way that would flag it
  as a fluke.
- **Not dominated by a handful of jobs**: the top 5 log files (jobs) contributing to
  the peak account for only ~27K of the 42.7K peak records combined — spread across
  many distinct jobs on both 5/11 and 5/12, not one benchmark run.
- **Write-heavy**: mean 1,735 writes vs. 3.6 reads per file in the peak — this is a
  write-dominated cluster, median 648 total ops.

**Reading:** this looks like a real platform signature — many independent
write-heavy jobs on `/lus/eagle` converging on a similar effective write throughput
(~30–36K ops/sec by this metric), plausibly reflecting a practical per-client
small-write IOPS ceiling on that Lustre target, not measurement noise. Unlike Part
D's cluster (which traced to one day's jobs), this one is broad-based — which makes
it a stronger candidate for "this is what frequent/typical write-heavy I/O looks like
on this platform" rather than an artifact to discard.

**`rate_wallclock` at `total_ops ≥ 20`** (n=2,703, further reduced by Finding 1's
issue) is dominated by the 100–1K bucket (64.96%), mean 10,019 ops/sec, median 598.81
ops/sec.

## Candidate seldom/frequent splits

All three methods, both metrics, both floors. Full detail in
`output/rate_splits.csv` / `output/rate_splits_minops20.csv` and the per-split overlay
PNGs (`output/<metric>_<method>_split[_minops20].png`).

### `rate_active`, unfiltered (n=162,722)

| Method | Seldom | Moderate/Unclassified | Frequent | Cut points (ops/sec) |
|---|---|---|---|---|
| Median | ≤25,257.50 (50.00%) | — | >25,257.50 (50.00%) | 25,257.50 |
| Tercile | ≤4,694.84 (33.47%) | 4,694.84–33,573.14 (33.23%) | >33,573.14 (33.31%) | 4,694.84 / 33,573.14 |
| Quartile | ≤3,937.01 (25.12%) | 3,937.01–35,350.44 (49.88%) | >35,350.44 (25.00%) | 3,937.01 / 35,350.44 |

### `rate_active`, `total_ops ≥ 20` (n=61,395)

| Method | Seldom | Moderate/Unclassified | Frequent | Cut points (ops/sec) |
|---|---|---|---|---|
| Median | ≤34,414.08 (50.01%) | — | >34,414.08 (49.99%) | 34,414.08 |
| Tercile | ≤32,605.00 (33.33%) | 32,605.00–35,864.02 (33.34%) | >35,864.02 (33.33%) | 32,605.00 / 35,864.02 |
| Quartile | ≤31,273.75 (25.00%) | 31,273.75–37,412.31 (50.00%) | >37,412.31 (24.99%) | 31,273.75 / 37,412.31 |

Note how tight the tercile/quartile bands are here (32.6K–37.4K spans the middle two
quartiles) — direct numeric evidence of Finding 3's sharp peak. **A percentile split
on this filtered distribution barely separates anything** — most files sit within a
narrow band of each other, so the "seldom" and "frequent" tails here are really just
"below/above a tight central cluster," not two visibly distinct populations.

### `rate_wallclock`, unfiltered (n=20,843)

| Method | Seldom | Moderate/Unclassified | Frequent | Cut points (ops/sec) |
|---|---|---|---|---|
| Median | ≤60.65 (50.00%) | — | >60.65 (50.00%) | 60.65 |
| Tercile | ≤37.16 (33.33%) | 37.16–319.18 (33.34%) | >319.18 (33.32%) | 37.16 / 319.18 |
| Quartile | ≤26.20 (25.00%) | 26.20–536.26 (50.00%) | >536.26 (25.00%) | 26.20 / 536.26 |

### `rate_wallclock`, `total_ops ≥ 20` (n=2,703)

| Method | Seldom | Moderate/Unclassified | Frequent | Cut points (ops/sec) |
|---|---|---|---|---|
| Median | ≤598.81 (50.02%) | — | >598.81 (49.98%) | 598.81 |
| Tercile | ≤460.85 (33.37%) | 460.85–880.44 (33.30%) | >880.44 (33.33%) | 460.85 / 880.44 |
| Quartile | ≤418.52 (25.05%) | 418.52–1,731.70 (49.94%) | >1,731.70 (25.01%) | 418.52 / 1,731.70 |

## Implication for Task 1 threshold work

- **`rate_active` is the usable metric; `rate_wallclock` is not, for most of this
  corpus** — 87.2% of records have an undefined wallclock span at Polaris's timestamp
  resolution. Any threshold adopted from this analysis should be based on
  `rate_active`, with `rate_wallclock` noted as a supplementary, small-subsample view
  only (skews toward longer-running jobs).
- **At `total_ops ≥ 20`, `rate_active` is dominated by one tight cluster (~30–36K
  ops/sec)**, traced to broad-based write-heavy activity on `/lus/eagle`, not a
  single-job artifact — likely a genuine platform signature (a practical write-IOPS
  ceiling), but it means percentile splits on the filtered distribution don't cleanly
  separate "seldom" from "frequent" the way Part D's contiguity splits separated
  "sequential" from "not" — the middle band is where almost everything lives.
- **The unfiltered `rate_active` distribution is more usable for a three-way split**
  (tercile cut points 4,695 / 33,573 ops/sec spread across a much wider range) than
  the `total_ops ≥ 20` version, but unfiltered includes low-op-count files whose rate
  estimate is unstable (same quantization concern flagged for contig_ratio in Part
  D) — there's a real tension between "floor for a stable rate estimate" and "floor
  that doesn't collapse the distribution into a single platform-specific cluster."
- This is HPC-native, Darshan-native, and **Polaris-specific** (single-platform,
  9-day sample, same caveat Part D carries) — worth re-running against a fuller/
  multi-platform collection before treating any specific cut point here as final.
  No numeric threshold from this analysis has been written into `profiles_backup.json`
  or any other existing file — this directory is a standalone empirical input for that
  decision, left for a follow-up choice among the candidates above.
