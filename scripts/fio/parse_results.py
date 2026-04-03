#!/usr/bin/env python3
"""
parse_results.py — summarise fio result files from scripts/fio/results/

Usage:
    python3 parse_results.py --beegfs                # latest hdd + ssd txt files in results/
    python3 parse_results.py --ost                   # latest hdd + ssd txt files in ost_results/
    python3 parse_results.py results/hdd_*.txt       # specific file(s)
"""

import re
import sys
import glob
import os
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_txt(path: Path) -> dict:
    """
    Extract per-job final stats from a fio --output-format=normal txt file.
    We want the LAST occurrence of each job block (the clean 60s window),
    not the intermediate cumulative dumps printed by --status-interval.
    """
    text = path.read_text()

    # Split into per-job blocks by the "jobname: (groupid=..." header
    job_blocks = re.split(r'\n(?=\w[\w_]+: \(groupid=)', text)

    # Collect all blocks per job name, keep only the last one
    last_blocks = {}
    for block in job_blocks:
        m = re.match(r'^(\w[\w_]+): \(groupid=(\d+)', block)
        if not m:
            continue
        name = m.group(1)
        last_blocks[name] = block   # later blocks overwrite earlier ones

    results = {}
    for job, block in last_blocks.items():
        entry = {"job": job}

        # bandwidth
        bw_w = re.search(r'write:.*?BW=([\d.]+)(MiB|KiB|GiB)/s', block)
        bw_r = re.search(r'read:.*?BW=([\d.]+)(MiB|KiB|GiB)/s', block)
        iops_w = re.search(r'write:.*?IOPS=([\d.]+k?)', block)
        iops_r = re.search(r'read:.*?IOPS=([\d.]+k?)', block)

        # avg latency (clat)
        lat_r = re.search(r'read:.*?clat.*?avg=([\d.]+)', block, re.DOTALL)
        lat_w = re.search(r'write:.*?clat.*?avg=([\d.]+)', block, re.DOTALL)
        lat_r_unit = re.search(r'read:.*?clat \((\w+)\)', block)
        lat_w_unit = re.search(r'write:.*?clat \((\w+)\)', block)

        def to_mib(val, unit):
            val = float(val)
            if unit == "KiB":
                return val / 1024
            if unit == "GiB":
                return val * 1024
            return val  # MiB

        def to_iops(val):
            val = str(val)
            if val.endswith("k"):
                return float(val[:-1]) * 1000
            return float(val)

        def to_ms(val, unit):
            val = float(val)
            if unit == "usec":
                return val / 1000
            if unit == "nsec":
                return val / 1_000_000
            return val  # msec

        if bw_w:
            entry["write_bw_mib"]  = to_mib(bw_w.group(1), bw_w.group(2))
        if bw_r:
            entry["read_bw_mib"]   = to_mib(bw_r.group(1), bw_r.group(2))
        if iops_w:
            entry["write_iops"]    = to_iops(iops_w.group(1))
        if iops_r:
            entry["read_iops"]     = to_iops(iops_r.group(1))
        if lat_w and lat_w_unit:
            entry["write_lat_ms"]  = to_ms(lat_w.group(1), lat_w_unit.group(1))
        if lat_r and lat_r_unit:
            entry["read_lat_ms"]   = to_ms(lat_r.group(1), lat_r_unit.group(1))

        results[job] = entry

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

JOB_ORDER = ["seq_write", "seq_read", "rand_write_4k", "rand_read_4k"]

def fmt_bw(mib):
    if mib is None:
        return "—"
    if mib >= 1024:
        return f"{mib/1024:.2f} GiB/s"
    if mib < 1:
        return f"{mib*1024:.1f} KiB/s"
    return f"{mib:.1f} MiB/s"

def fmt_iops(v):
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v/1000:.1f}k"
    return f"{v:.0f}"

def fmt_lat(ms):
    if ms is None:
        return "—"
    if ms < 1:
        return f"{ms*1000:.0f} µs"
    return f"{ms:.1f} ms"


def print_summary(label: str, jobs: dict, file=sys.stdout):
    print(f"\n{'='*62}", file=file)
    print(f"  {label}", file=file)
    print(f"{'='*62}", file=file)
    print(f"  {'Job':<18}  {'Direction':<6}  {'Bandwidth':>12}  {'IOPS':>8}  {'Avg Lat':>10}", file=file)
    print(f"  {'-'*18}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*10}", file=file)

    for jname in JOB_ORDER:
        j = jobs.get(jname)
        if not j:
            continue
        if "write_bw_mib" in j:
            print(f"  {jname:<18}  {'WRITE':<6}  {fmt_bw(j.get('write_bw_mib')):>12}  "
                  f"{fmt_iops(j.get('write_iops')):>8}  {fmt_lat(j.get('write_lat_ms')):>10}", file=file)
        if "read_bw_mib" in j:
            print(f"  {jname:<18}  {'READ':<6}  {fmt_bw(j.get('read_bw_mib')):>12}  "
                  f"{fmt_iops(j.get('read_iops')):>8}  {fmt_lat(j.get('read_lat_ms')):>10}", file=file)


def print_comparison(title: str, base: dict, target: dict, base_lbl: str, target_lbl: str, file=sys.stdout):
    print(f"\n{'='*82}", file=file)
    print(f"  {title}", file=file)
    print(f"{'='*82}", file=file)
    print(f"  {'Job':<18}  {'Dir':<5}  "
          f"{base_lbl+' BW':>12}  {target_lbl+' BW':>12}  {'Speedup':>8}  "
          f"{base_lbl+' IOPS':>9}  {target_lbl+' IOPS':>9}", file=file)
    print(f"  {'-'*18}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*9}", file=file)

    for jname in JOB_ORDER:
        h = base.get(jname, {})
        s = target.get(jname, {})

        for direction, bw_key, iops_key in [
            ("WRITE", "write_bw_mib", "write_iops"),
            ("READ",  "read_bw_mib",  "read_iops"),
        ]:
            hbw = h.get(bw_key)
            sbw = s.get(bw_key)
            if hbw is None and sbw is None:
                continue
            speedup = f"{sbw/hbw:.2f}x" if hbw and sbw else "—"
            print(f"  {jname:<18}  {direction:<5}  "
                  f"{fmt_bw(hbw):>12}  {fmt_bw(sbw):>12}  {speedup:>8}  "
                  f"{fmt_iops(h.get(iops_key)):>9}  {fmt_iops(s.get(iops_key)):>9}", file=file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def latest_file(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    return Path(files[-1]) if files else None


def main():
    args = sys.argv[1:]
    
    target_dir = RESULTS_DIR
    if "--ost" in args:
        target_dir = RESULTS_DIR.parent / "ost_results"
        args.remove("--ost")
    elif "--beegfs" in args:
        target_dir = RESULTS_DIR
        args.remove("--beegfs")

    compare_mode = "--compare" in args
    if compare_mode:
        args.remove("--compare")

    if args:
        files = [Path(a) for a in args]
    else:
        # Auto-detect latest hdd and ssd txt files from standard and ost dirs
        files = []
        d = target_dir
        if d.exists():
            txt_files = list(d.glob("*.txt"))
            txt_files.sort(key=lambda x: x.stat().st_mtime)
            if txt_files:
                newest = txt_files[-1]
                try:
                    ts = newest.stem.split("_")[-2:]
                    if len(ts) == 2 and ts[0].isdigit() and ts[1].isdigit():
                        ts_str = f"_{ts[0]}_{ts[1]}"
                        files.extend(list(d.glob(f"*{ts_str}.txt")))
                except Exception:
                    pass
                
        # Always try to match up with beegfs baseline if we are in ost
        # or grab everything from both just in case
        if d != RESULTS_DIR and RESULTS_DIR.exists():
            txt_files = list(RESULTS_DIR.glob("*.txt"))
            txt_files.sort(key=lambda x: x.stat().st_mtime)
            if txt_files:
                newest = txt_files[-1]
                try:
                    ts = newest.stem.split("_")[-2:]
                    if len(ts) == 2 and ts[0].isdigit() and ts[1].isdigit():
                        ts_str = f"_{ts[0]}_{ts[1]}"
                        files.extend(list(RESULTS_DIR.glob(f"*{ts_str}.txt")))
                except Exception:
                    pass

    if not files:
        print(f"No result files found in {target_dir}")
        sys.exit(1)

    parsed = {}
    for f in files:
        if "summary_comparison" in f.name:
            continue
            
        label = f.stem          # e.g. hdd_ost1_20260402_191721
        pool = label.split("_20")[0].upper()
        print(f"Parsing: {f}")
        parsed[pool] = (label, parse_txt(f))

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = target_dir / f"summary_comparison_{ts}.txt"
    target_dir.mkdir(exist_ok=True)
    
    def pool_sort_key(p):
        cat = 0 if p.startswith("HDD") else (1 if p.startswith("SSD") else 2)
        m = re.search(r'\d+$', p)
        return (cat, int(m.group()) if m else 0, p)
        
    sorted_pools = sorted(parsed.keys(), key=pool_sort_key)

    with open(out_file, "w") as fout:
        for pool in sorted_pools:
            label, jobs = parsed[pool]
            print_summary(f"{pool}  ({label})", jobs, file=fout)
            print_summary(f"{pool}  ({label})", jobs, file=sys.stdout)

        for p in sorted_pools:
            if p.startswith("HDD_OST") and "HDD" in parsed:
                _, ost = parsed[p]
                _, hdd = parsed["HDD"]
                print_comparison(f"HDD vs {p} Comparison", hdd, ost, "HDD", p, file=fout)
                print_comparison(f"HDD vs {p} Comparison", hdd, ost, "HDD", p, file=sys.stdout)
                
            if p.startswith("SSD_OST") and "SSD" in parsed:
                _, ost = parsed[p]
                _, ssd = parsed["SSD"]
                print_comparison(f"SSD vs {p} Comparison", ssd, ost, "SSD", p, file=fout)
                print_comparison(f"SSD vs {p} Comparison", ssd, ost, "SSD", p, file=sys.stdout)

    print(f"\nSaved summary and comparisons to {out_file}")


if __name__ == "__main__":
    main()
