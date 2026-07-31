# Contiguous ratio distribution — real ALCF Polaris Darshan logs

Computes, per file record, `contig_ratio = (POSIX_CONSEC_READS + POSIX_CONSEC_WRITES) / (POSIX_READS + POSIX_WRITES)`
across a corpus of real Darshan logs, and shows how that ratio is distributed across files.

## Data source used (important)

The path originally given, `/home/advay/Documents/Code/dASHLAB/2024`, only contains the
Polaris collection's directory *skeleton* — every `logs.tar.gz` under it is a **0-byte
placeholder** (221 tarballs, all empty). No actual log data has been downloaded there.

Real `.darshan` files were found instead at:

```
/home/advay/darshan_logs_analysis/2024-20260128T084726Z-3-001/2024/
```

This covers **9 days** (2024-04-24 through 2024-04-30, and 2024-05-11/12) — a partial
slice of the full year-long collection, not all of it. Re-run `unpack-darshan-logs.sh`
against real (non-empty) tarballs if you want the full 2024 collection instead.

Two of those 9 days (**4/30** and **5/11**) initially showed 0 logs — not because the
data was missing, but because, unlike every other populated day, they only had the
`logs.tar.gz` archive on disk with no extracted `logs/` folder next to it (confirmed by
comparing directory contents: populated days have both `logs.tar.gz` *and* `logs/`; these
two only had the tarball). Both tarballs were legitimate, non-empty archives
(~11 MB and ~12 MB) and were extracted (`tar -xzf`), adding **2,342** and **257** logs
respectively (2,599 new logs total).

**Current totals (all 9 days, fully extracted): 14,900 logs, 162,722 file records.**
(Earlier numbers below, from before the 4/30 + 5/11 extraction, were 12,301 logs /
89,389 file records — kept in the history here since the two runs are still worth
comparing.)

## Files

- `parse_logs.py` — walks a log directory, runs `darshan-parser` (at
  `/home/advay/darshan/bin/darshan-parser`) on every `.darshan` file, sums
  `POSIX_READS/WRITES/CONSEC_READS/CONSEC_WRITES` **per record_id** (i.e. per file,
  aggregated across all MPI ranks — safe because Darshan never keeps both per-rank
  and shared-aggregate records for the same file), and writes one row per
  (log, file) to a CSV. Parallelized across CPU cores.
- `analyze_distribution.py` — bins `contig_ratio` into 5%-wide buckets, prints the
  table, and plots a bar chart (`--min-ops N` filters out low-op-count files first).
- `output/` — generated CSVs and PNGs (not committed logic, just run output).

## Rerunning

```bash
python3 parse_logs.py <log_root_dir> output/per_file_contig_ratio.csv
python3 analyze_distribution.py output/per_file_contig_ratio.csv output
python3 analyze_distribution.py output/per_file_contig_ratio.csv output --min-ops 20
```

## Key finding: the raw distribution is dominated by a quantization artifact, not real signal

The unfiltered distribution (`contig_ratio_distribution.png`, now n=162,722) still looks
trimodal: **44.0%** of files at 0-5%, **9.4%** at 50-55%, **36.5%** at 95-100%. That shape
is **mostly an artifact of low operation counts**, not evidence about real access patterns:

- `POSIX_CONSEC_*` only counts an op as "consecutive" relative to the *previous* op on
  that file — the first op can never count. So a file with **1 total op always has
  contig_ratio = 0%**, regardless of what that single op actually did.
- A file with **2 total ops** can only land at 0% or 50% (0 or 1 of the 2 ops can be
  consecutive) — nothing in between is even possible.
- Verified directly (recomputed on the full 162,722-record set): the 0-5% bucket has a
  **median total_ops of 1** (98.2% have ≤4 ops); the 50-55% bucket has a **median
  total_ops of 2** (85.0% have ≤4 ops). Both spikes are essentially "files that were
  barely touched," not "files with random/half-and-half access patterns."
- The 95-100% bucket is the only population with real weight behind it: **median
  total_ops = 432**, 0% of it has ≤4 ops.

Filtering to files with **`total_ops ≥ 20`** (`contig_ratio_distribution_minops20.png`,
**61,395 files**, up from 32,852) collapses almost entirely into the top bucket:
**96.73% of files land at 95-100% contiguous**, mean ratio 98.31%, median 99.77% — close
to, but slightly softer than, the pre-extraction numbers (98.11% / 98.70% / 99.88%).

**New in the fuller dataset — an 80-85% cluster:** a bucket that was negligible before
(0.20%, 67 files) is now **2.2% of the `total_ops ≥ 20` subset (1,359 files)**. Traced
before trusting it: **95% of this bucket (1,290 of 1,359 files) comes from a single day,
2024-05-11** — one of the two newly-extracted days — almost entirely on `/lus/eagle`,
with a median of only 35 total ops (much lower than the 95-100% bucket's median of 432).
This looks like a real, distinct behavior from specific jobs that ran that day
(consistently ~80-85% contiguous, not fully sequential) rather than noise, given how
tightly it clusters in one bucket and one filesystem — but it's one day's worth of jobs,
not a general pattern confirmed across the dataset.

**Implication for Task 1 threshold work:** any file-level "contiguous ratio" threshold
needs a minimum-op-count floor to be meaningful — below that floor the ratio is
quantized and doesn't reflect real access-pattern behavior. This is exactly the kind of
empirically-derived, project-specific finding flagged as missing from literature in
`LitReview_task1.md`. The 80-85% cluster is also a reminder that single-day artifacts can
swing the aggregate distribution meaningfully — worth re-checking once more of the full
year-long collection is available, rather than treating this 9-day (now fully-extracted)
slice as final.
