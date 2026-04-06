#!/usr/bin/env python3
import argparse
import json
import statistics
import os
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required for plotting.")
    print("Please install them using: pip3 install matplotlib numpy")
    exit(1)

def visualize_json(json_path):
    print(f"Processing data from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Group data by (num_files, fsize) -> pool -> mode -> list of runs
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in data:
        key = (row.get("num_files", 1), row["fsize"])
        pool = row["pool"]
        mode = row["mode"]
        grouped[key][pool][mode].append(row)

    modes_order = ["seq_read", "seq_write", "rand_read", "rand_write", "seq_rw", "rand_rw"]
    
    # Create a plots subdirectory next to the json file
    output_dir = Path(json_path).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    for (num_files, fsize), pool_dict in grouped.items():
        all_pools = sorted(pool_dict.keys())
        if not all_pools:
            continue
            
        plot_groups = {
            "combined": all_pools,
            "hdd_only": [p for p in all_pools if "HDD" in p.upper()],
            "ssd_only": [p for p in all_pools if "SSD" in p.upper()]
        }
        
        for group_name, pools in plot_groups.items():
            if not pools:
                continue

            # Determine all metrics (Read vs Write) present
            metrics_present = []
            for m in modes_order:
                if any(m in pool_dict[p] for p in pools):
                    if "read" in m and "rw" not in m:
                        metrics_present.append((m, "read"))
                    elif "write" in m:
                        metrics_present.append((m, "write"))
                    elif "rw" in m:
                        metrics_present.append((m, "read"))
                        metrics_present.append((m, "write"))
            
            if not metrics_present:
                continue

            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(pools))
            
            # Calculate width of individual bars
            total_group_width = 0.8
            bar_width = total_group_width / len(metrics_present)

            for i, (m, rw_type) in enumerate(metrics_present):
                means = []
                errs = []
                
                for p in pools:
                    runs = pool_dict[p].get(m, [])
                    if not runs:
                        means.append(0)
                        errs.append(0)
                        continue
                        
                    if rw_type == "read":
                        bws = [r.get("read_bw_mib", 0) for r in runs]
                    else:
                        bws = [r.get("write_bw_mib", 0) for r in runs]
                    
                    means.append(statistics.mean(bws))
                    errs.append(statistics.stdev(bws) if len(bws) > 1 else 0)
                    
                offsets = x - (total_group_width / 2) + (i * bar_width) + (bar_width / 2)
                
                label = f"{m} ({rw_type.upper()})"
                ax.bar(offsets, means, bar_width, yerr=errs, label=label, capsize=4, alpha=0.85, edgecolor='black')

            group_title_suffix = ""
            if group_name == "hdd_only":
                group_title_suffix = " (HDD Targets only)"
            elif group_name == "ssd_only":
                group_title_suffix = " (SSD Targets only)"

            ax.set_ylabel('Bandwidth [MiB/s]', fontsize=12)
            ax.set_title(f'FIO Bandwidth over Targets{group_title_suffix} | {num_files} File(s) | {fsize}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pools, fontsize=11, rotation=45, ha='right')
            ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')
            
            ax.set_axisbelow(True)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            fig.tight_layout()
            
            # Output chart
            out_file = output_dir / f"{group_name}_plot_{num_files}f_{fsize}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate bar charts from matrix benchmark JSON.")
    parser.add_argument("file", nargs="?", help="Path to matrix_results_*.json file. Defaults to newest in results/ or ost_results/.")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = []
        for d in ["results", "ost_results"]:
            p = Path(__file__).parent / d
            if p.exists():
                files.extend(list(p.glob("matrix_results_*.json")))
        
        if not files:
            print("No matrix_results JSON found. Please run the benchmark first.")
            return
            
        files.sort(key=lambda x: x.stat().st_mtime)
        files = [files[-1]] # analyze the newest file by default

    for f in files:
        visualize_json(f)

if __name__ == "__main__":
    main()
