"""
Walk a directory tree of .darshan logs, run darshan-parser on each one, and
compute the "contiguous ratio" -- (POSIX_CONSEC_READS + POSIX_CONSEC_WRITES) / total_ops
-- for every file record (one row per (log, record_id), aggregated across MPI ranks).

Usage:
    python3 parse_logs.py <log_root_dir> <out_csv> [--parser /path/to/darshan-parser] [--jobs N]
"""
import argparse
import csv
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

NEEDED_COUNTERS = {"POSIX_READS", "POSIX_WRITES", "POSIX_CONSEC_READS", "POSIX_CONSEC_WRITES"}


def parse_one_log(args):
    log_path, darshan_parser = args
    try:
        proc = subprocess.run(
            [darshan_parser, str(log_path)],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        return [], f"{log_path}: subprocess failed ({e})"

    # per record_id: counter_name -> summed value (across all rank rows)
    per_record = {}
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
        try:
            val = int(counter_value)
        except ValueError:
            continue
        if val < 0:  # -1 means "Darshan could not monitor this counter"
            continue

        rec = per_record.setdefault(record_id, {})
        rec[counter_name] = rec.get(counter_name, 0) + val
        file_names[record_id] = file_name
        mount_info[record_id] = (mount_pt, fs_type)

    rows = []
    for record_id, counters in per_record.items():
        reads = counters.get("POSIX_READS", 0)
        writes = counters.get("POSIX_WRITES", 0)
        consec_reads = counters.get("POSIX_CONSEC_READS", 0)
        consec_writes = counters.get("POSIX_CONSEC_WRITES", 0)
        total_ops = reads + writes
        if total_ops == 0:
            continue  # ratio undefined; file was opened/stat'd but never read/written
        contig_ratio = (consec_reads + consec_writes) / total_ops
        mount_pt, fs_type = mount_info[record_id]
        rows.append({
            "log_file": str(log_path),
            "record_id": record_id,
            "file_name": file_names[record_id],
            "mount_pt": mount_pt,
            "fs_type": fs_type,
            "posix_reads": reads,
            "posix_writes": writes,
            "consec_reads": consec_reads,
            "consec_writes": consec_writes,
            "total_ops": total_ops,
            "contig_ratio": contig_ratio,
        })
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_root", help="directory to recursively search for *.darshan files")
    ap.add_argument("out_csv", help="output CSV path (one row per file record)")
    ap.add_argument("--parser", default="/home/advay/darshan/bin/darshan-parser")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    log_root = Path(args.log_root)
    logs = sorted(log_root.rglob("*.darshan"))
    if not logs:
        print(f"ERROR: no .darshan files found under {log_root}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(logs)} .darshan logs under {log_root}")

    tasks = [(p, args.parser) for p in logs]
    all_rows = []
    errors = []

    with mp.Pool(args.jobs) as pool:
        for i, (rows, err) in enumerate(pool.imap_unordered(parse_one_log, tasks, chunksize=32), 1):
            if err:
                errors.append(err)
            all_rows.extend(rows)
            if i % 1000 == 0 or i == len(logs):
                print(f"  processed {i}/{len(logs)} logs, {len(all_rows)} file records so far")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["log_file", "record_id", "file_name", "mount_pt", "fs_type",
                  "posix_reads", "posix_writes", "consec_reads", "consec_writes",
                  "total_ops", "contig_ratio"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} file records to {out_path}")
    print(f"Logs with errors: {len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
