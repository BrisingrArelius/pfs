"""
Walk a directory tree of .darshan logs, run darshan-parser on each one, and compute
two per-file I/O rate metrics -- for every file record (one row per (log, record_id),
aggregated across MPI ranks):

    rate_active    = (POSIX_READS + POSIX_WRITES) / (POSIX_F_READ_TIME + POSIX_F_WRITE_TIME)
    rate_wallclock = (POSIX_READS + POSIX_WRITES) / (POSIX_F_CLOSE_END_TIMESTAMP - POSIX_F_OPEN_START_TIMESTAMP)

Same methodology as CONTIG_TESTING_CLAUDE/parse_logs.py (darshan-parser CLI via
subprocess, not pydarshan) applied to the frequency/rate dimension instead of the
contiguous-ratio dimension. This is a standalone script -- it does not import or
modify anything in CONTIG_TESTING_CLAUDE/.

Usage:
    python3 parse_logs.py <log_root_dir> <out_csv> [--parser /path/to/darshan-parser] [--jobs N]
        [--chunksize N] [--timeout SECONDS]

Notes on scaling to large corpora (500K+ logs):
  - Rows are written to <out_csv> incrementally as each log finishes, not buffered
    in memory until the end -- keeps memory flat and means a kill/crash partway
    through still leaves a usable partial CSV.
  - --chunksize controls how many logs each worker pulls off the queue at once
    before reporting back. The default (32, inherited from CONTIG_TESTING_CLAUDE)
    is fine for small/uniform corpora, but on large corpora with a few unusually
    slow/huge logs it causes a "straggler" effect: one worker can get stuck
    serially churning through a chunk containing several slow logs (each up to
    --timeout seconds) while the rest of the pool sits idle, waiting -- this shows
    up as the progress counter appearing to freeze near the end of a run even
    though nothing has actually hung. Lower --chunksize (e.g. 1-4) spreads slow
    logs across workers instead of concentrating them, at the cost of slightly
    more IPC overhead -- worth it once the corpus is large enough that a handful
    of pathological logs is likely.
  - Progress lines now include elapsed time and a rolling logs/sec rate, so a
    genuine stall (rate dropping toward 0) is distinguishable from "just slow".
"""
import argparse
import csv
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

# counter_name -> (python_type, aggregation_strategy)
# - POSIX_READS/WRITES: op counts, summed across ranks.
# - POSIX_F_READ_TIME/WRITE_TIME: cumulative time-in-syscall per rank, summed across
#   ranks (same logic as summing op counts -- total active I/O time across all ranks
#   touching this file).
# - POSIX_F_OPEN_START_TIMESTAMP: point-in-time, take the earliest (min) across ranks.
# - POSIX_F_CLOSE_END_TIMESTAMP: point-in-time, take the latest (max) across ranks.
# (Summing timestamps, as the count/time counters are summed, would be meaningless.)
COUNTER_SPEC = {
    "POSIX_READS": (int, "sum"),
    "POSIX_WRITES": (int, "sum"),
    "POSIX_F_READ_TIME": (float, "sum"),
    "POSIX_F_WRITE_TIME": (float, "sum"),
    "POSIX_F_OPEN_START_TIMESTAMP": (float, "min"),
    "POSIX_F_CLOSE_END_TIMESTAMP": (float, "max"),
}
NEEDED_COUNTERS = set(COUNTER_SPEC)


def parse_one_log(args):
    log_path, darshan_parser, timeout = args
    try:
        proc = subprocess.run(
            [darshan_parser, str(log_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        return [], f"{log_path}: subprocess failed ({e})"

    per_record = {}  # record_id -> {counter_name: aggregated_value}
    file_names = {}
    mount_info = {}

    for line in proc.stdout.splitlines():
        if not line.startswith("POSIX\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        _module, _rank, record_id, counter_name, counter_value, file_name, mount_pt, fs_type = parts[:8]
        if counter_name not in NEEDED_COUNTERS:
            continue
        py_type, strategy = COUNTER_SPEC[counter_name]
        try:
            val = py_type(counter_value)
        except ValueError:
            continue
        if val < 0:  # -1 (or -1.0) means "Darshan could not monitor this counter"
            continue

        rec = per_record.setdefault(record_id, {})
        if counter_name not in rec:
            rec[counter_name] = val
        elif strategy == "sum":
            rec[counter_name] += val
        elif strategy == "min":
            rec[counter_name] = min(rec[counter_name], val)
        elif strategy == "max":
            rec[counter_name] = max(rec[counter_name], val)

        file_names[record_id] = file_name
        mount_info[record_id] = (mount_pt, fs_type)

    rows = []
    for record_id, counters in per_record.items():
        reads = counters.get("POSIX_READS", 0)
        writes = counters.get("POSIX_WRITES", 0)
        total_ops = reads + writes
        if total_ops == 0:
            continue  # opened/stat'd but never read/written -- no rate to compute

        f_read_time = counters.get("POSIX_F_READ_TIME", 0.0)
        f_write_time = counters.get("POSIX_F_WRITE_TIME", 0.0)
        open_start_ts = counters.get("POSIX_F_OPEN_START_TIMESTAMP")
        close_end_ts = counters.get("POSIX_F_CLOSE_END_TIMESTAMP")

        active_time = f_read_time + f_write_time
        rate_active = (total_ops / active_time) if active_time > 0 else ""

        rate_wallclock = ""
        if open_start_ts is not None and close_end_ts is not None:
            wallclock_span = close_end_ts - open_start_ts
            if wallclock_span > 0:
                rate_wallclock = total_ops / wallclock_span

        mount_pt, fs_type = mount_info[record_id]
        rows.append({
            "log_file": str(log_path),
            "record_id": record_id,
            "file_name": file_names[record_id],
            "mount_pt": mount_pt,
            "fs_type": fs_type,
            "posix_reads": reads,
            "posix_writes": writes,
            "total_ops": total_ops,
            "f_read_time": f_read_time,
            "f_write_time": f_write_time,
            "open_start_ts": open_start_ts if open_start_ts is not None else "",
            "close_end_ts": close_end_ts if close_end_ts is not None else "",
            "rate_active": rate_active,
            "rate_wallclock": rate_wallclock,
        })
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_root", help="directory to recursively search for *.darshan files")
    ap.add_argument("out_csv", help="output CSV path (one row per file record)")
    ap.add_argument("--parser", default="/home/advay/darshan/bin/darshan-parser")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("--chunksize", type=int, default=4,
                     help="logs per worker batch (default 4; lower spreads slow/large "
                          "logs across workers instead of stalling one worker on a "
                          "batch full of them -- see module docstring)")
    ap.add_argument("--timeout", type=float, default=60,
                     help="per-log darshan-parser timeout in seconds (default 60)")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    logs = sorted(log_root.rglob("*.darshan"))
    if not logs:
        print(f"ERROR: no .darshan files found under {log_root}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(logs)} .darshan logs under {log_root}")

    tasks = [(p, args.parser, args.timeout) for p in logs]
    errors = []
    n_rows = 0
    n_active_undefined = 0
    n_wallclock_undefined = 0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["log_file", "record_id", "file_name", "mount_pt", "fs_type",
                  "posix_reads", "posix_writes", "total_ops",
                  "f_read_time", "f_write_time", "open_start_ts", "close_end_ts",
                  "rate_active", "rate_wallclock"]

    start = time.monotonic()
    last_report_t = start
    last_report_i = 0
    last_flush_t = start

    with open(out_path, "w", newline="") as f, mp.Pool(args.jobs) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()  # header should be visible to external readers immediately, not
                   # held in the buffer until the first periodic flush below

        for i, (rows, err) in enumerate(
            pool.imap_unordered(parse_one_log, tasks, chunksize=args.chunksize), 1
        ):
            if err:
                errors.append(err)
            if rows:
                writer.writerows(rows)
                n_rows += len(rows)
                n_active_undefined += sum(1 for r in rows if r["rate_active"] == "")
                n_wallclock_undefined += sum(1 for r in rows if r["rate_wallclock"] == "")

            if i % 1000 == 0 or i == len(logs):
                now = time.monotonic()
                elapsed = now - start
                interval_rate = (i - last_report_i) / max(now - last_report_t, 1e-9)
                overall_rate = i / max(elapsed, 1e-9)
                print(f"  processed {i}/{len(logs)} logs, {n_rows} file records so far  "
                      f"[{interval_rate:.1f} logs/sec last batch, {overall_rate:.1f} logs/sec overall, "
                      f"{elapsed:.0f}s elapsed]")
                last_report_t = now
                last_report_i = i

            # Flush on a short time-based cadence (independent of the 1000-log print
            # interval above) so the CSV is genuinely tail-able / externally readable
            # while a large run is still in progress, instead of only becoming visible
            # once every 1000 logs -- important on slow/large runs where 1000 logs can
            # take a long time.
            now = time.monotonic()
            if now - last_flush_t > 5:
                f.flush()
                last_flush_t = now

    print(f"\nWrote {n_rows} file records to {out_path}")
    print(f"  rate_active undefined (zero active I/O time): {n_active_undefined}")
    print(f"  rate_wallclock undefined (zero/negative wallclock span): {n_wallclock_undefined}")
    print(f"Logs with errors: {len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")
    if len(errors) > 10:
        errors_path = out_path.with_suffix(".errors.txt")
        errors_path.write_text("\n".join(errors))
        print(f"  ... ({len(errors) - 10} more; full list written to {errors_path})")


if __name__ == "__main__":
    main()
