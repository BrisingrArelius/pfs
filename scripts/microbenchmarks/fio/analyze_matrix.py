#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

def fmt_bw(mib):
    if mib < 1:
        return f"{mib*1024:.1f} KiB/s"
    if mib > 1024:
        return f"{mib/1024:.2f} GiB/s"
    return f"{mib:.1f} MiB/s"

def fmt_iops(iops):
    if iops > 1000:
        return f"{iops/1000:.1f}k"
    return f"{iops:.0f}"

def fmt_lat(ms):
    if ms < 1:
        return f"{ms*1000:.0f} us"
    return f"{ms:.2f} ms"

def analyze_json(json_path):
    print(f"Analyzing: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Group by (pool, num_files, fsize, mode)
    # Each group will hold a list of results from the N runs
    grouped = defaultdict(list)
    for row in data:
        key = (row["pool"], row.get("num_files", 1), row["fsize"], row["mode"])
        grouped[key].append(row)

    # We want to print tables per (pool, num_files, fsize)
    tables = defaultdict(dict)
    for (pool, num_files, fsize, mode), runs in grouped.items():
        # Calculate averages safely
        avg_read_bw = statistics.mean([r.get("read_bw_mib", 0) for r in runs])
        avg_write_bw = statistics.mean([r.get("write_bw_mib", 0) for r in runs])
        avg_read_iops = statistics.mean([r.get("read_iops", 0) for r in runs])
        avg_write_iops = statistics.mean([r.get("write_iops", 0) for r in runs])
        avg_read_lat = statistics.mean([r.get("read_lat_ms", 0) for r in runs])
        avg_write_lat = statistics.mean([r.get("write_lat_ms", 0) for r in runs])

        # Track OST hits by flattening all runs
        all_hits = []
        for r in runs:
            all_hits.extend(r.get("ost_hits", []))
        unique_hits = len(set(all_hits))

        tables[(pool, num_files, fsize)][mode] = {
            "runs": len(runs),
            "read_bw": avg_read_bw,
            "write_bw": avg_write_bw,
            "read_iops": avg_read_iops,
            "write_iops": avg_write_iops,
            "read_lat": avg_read_lat,
            "write_lat": avg_write_lat,
            "ost_hits_count": unique_hits
        }

    # Print results
    for (pool, num_files, fsize), modes_dict in sorted(tables.items()):
        print(f"\n==================================================================================")
        print(f" Target: {pool} | Files: {num_files} | Total Size: {fsize}")
        print(f"==================================================================================")
        print(f" {'Mode':<12} | {'Read BW':<12} | {'Write BW':<12} | {'Read IOPS':<10} | {'Write IOPS':<10} | {'Read Lat':<10} | {'Write Lat':<10} | {'Runs':<4}")
        print("-" * 110)

        # Sort mode to be consistent
        for mode in ["seq_read", "seq_write", "rand_read", "rand_write", "seq_rw", "rand_rw"]:
            if mode not in modes_dict:
                continue
            
            d = modes_dict[mode]
            rbw = fmt_bw(d["read_bw"]) if d["read_bw"] > 0 else "-"
            wbw = fmt_bw(d["write_bw"]) if d["write_bw"] > 0 else "-"
            riops = fmt_iops(d["read_iops"]) if d["read_iops"] > 0 else "-"
            wiops = fmt_iops(d["write_iops"]) if d["write_iops"] > 0 else "-"
            rlat = fmt_lat(d["read_lat"]) if d["read_lat"] > 0 else "-"
            wlat = fmt_lat(d["write_lat"]) if d["write_lat"] > 0 else "-"
            
            print(f" {mode:<12} | {rbw:<12} | {wbw:<12} | {riops:<10} | {wiops:<10} | {rlat:<10} | {wlat:<10} | {d['runs']:<4}")

        print("")

def main():
    parser = argparse.ArgumentParser(description="Analyze matrix benchmark JSON output.")
    parser.add_argument("file", nargs="?", help="Path to matrix_results_*.json file.")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        # Try to find the newest matrix_results in results/ or ost_results/
        files = []
        for d in ["results", "ost_results"]:
            p = Path(__file__).parent / d
            if p.exists():
                files.extend(list(p.glob("matrix_results_*.json")))
        
        if not files:
            print("No matrix_results_*.json files found in results/ or ost_results/. Please specify a file.")
            return
            
        files.sort(key=lambda x: x.stat().st_mtime)
        files = [files[-1]] # just take the single newest file

    for f in files:
        analyze_json(f)

if __name__ == "__main__":
    main()
