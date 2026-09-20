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
    
    # Create output directory structure outside scripts
    repo_root = Path(__file__).resolve().parent.parent.parent
    base_out_dir = repo_root / "results" / "ost_plots"
    bar_dir = base_out_dir / "bar_graphs"
    box_dir = base_out_dir / "box_plots"
    heatmap_dir = base_out_dir / "heatmaps"
    
    base_out_dir.mkdir(parents=True, exist_ok=True)
    bar_dir.mkdir(exist_ok=True)
    box_dir.mkdir(exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)

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

            fig_bar, ax_bar = plt.subplots(figsize=(16, 8))
            fig_box, ax_box = plt.subplots(figsize=(16, 8))
            x = np.arange(len(pools))
            
            # Calculate width of individual bars
            total_group_width = 0.8
            bar_width = total_group_width / len(metrics_present)
            
            heatmap_data = np.zeros((len(pools), len(metrics_present)))

            for i, (m, rw_type) in enumerate(metrics_present):
                means = []
                errs = []
                box_data = []
                
                for p_idx, p in enumerate(pools):
                    runs = pool_dict[p].get(m, [])
                    if not runs:
                        means.append(0)
                        errs.append(0)
                        box_data.append([0])
                        heatmap_data[p_idx, i] = 0
                        continue
                        
                    if rw_type == "read":
                        bws = [r.get("read_bw_mib", 0) for r in runs]
                    else:
                        bws = [r.get("write_bw_mib", 0) for r in runs]
                    
                    mean_val = statistics.mean(bws)
                    means.append(mean_val)
                    errs.append(statistics.stdev(bws) if len(bws) > 1 else 0)
                    box_data.append(bws)
                    heatmap_data[p_idx, i] = mean_val
                    
                offsets = x - (total_group_width / 2) + (i * bar_width) + (bar_width / 2)
                label = f"{m} ({rw_type.upper()})"
                
                # Plot Bar Chart
                ax_bar.bar(offsets, means, bar_width, yerr=errs, label=label, capsize=4, alpha=0.85, edgecolor='black')
                
                # Plot Box Plot (using the same positions)
                bp = ax_box.boxplot(box_data, positions=offsets, widths=bar_width*0.8, patch_artist=True, manage_ticks=False)
                for patch in bp['boxes']:
                    patch.set_facecolor(plt.cm.tab10(i % 10))
                    patch.set_alpha(0.7)

            group_title_suffix = ""
            if group_name == "hdd_only":
                group_title_suffix = " (HDD Targets only)"
            elif group_name == "ssd_only":
                group_title_suffix = " (SSD Targets only)"

            def format_axes(ax, title):
                ax.set_ylabel('Bandwidth [MiB/s]', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(pools, fontsize=11, rotation=45, ha='right')
                ax.set_axisbelow(True)
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Export Bar Graph
            format_axes(ax_bar, f'FIO Bandwidth (Bar) over Targets{group_title_suffix} | {num_files} File(s) | {fsize}')
            ax_bar.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')
            fig_bar.tight_layout()
            bar_out = bar_dir / f"{group_name}_bar_{num_files}f_{fsize}.png"
            fig_bar.savefig(bar_out, dpi=300, bbox_inches='tight')
            plt.close(fig_bar)

            # Export Box Plot
            format_axes(ax_box, f'FIO Bandwidth (Box) over Targets{group_title_suffix} | {num_files} File(s) | {fsize}')
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(color=plt.cm.tab10(i % 10), alpha=0.7, label=f"{m} ({rw.upper()})") for i, (m, rw) in enumerate(metrics_present)]
            ax_box.legend(handles=handles, fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')
            fig_box.tight_layout()
            box_out = box_dir / f"{group_name}_box_{num_files}f_{fsize}.png"
            fig_box.savefig(box_out, dpi=300, bbox_inches='tight')
            plt.close(fig_box)

            # Export Heatmap
            fig_hm, ax_hm = plt.subplots(figsize=(10, len(pools)*0.6 + 3))
            im = ax_hm.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax_hm.set_xticks(np.arange(len(metrics_present)))
            ax_hm.set_yticks(np.arange(len(pools)))
            metric_labels = [f"{m}\n({rw.upper()})" for m, rw in metrics_present]
            ax_hm.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=10)
            ax_hm.set_yticklabels(pools, fontsize=11)
            
            for i_p in range(len(pools)):
                for j_m in range(len(metrics_present)):
                    val = heatmap_data[i_p, j_m]
                    color = "black" if val < np.max(heatmap_data)*0.6 else "white"
                    ax_hm.text(j_m, i_p, f"{val:.0f}", ha="center", va="center", color=color)

            ax_hm.set_title(f'FIO Bandwidth Heatmap{group_title_suffix}\n{num_files} File(s) | {fsize}', fontsize=14, fontweight='bold')
            fig_hm.tight_layout()
            hm_out = heatmap_dir / f"{group_name}_heatmap_{num_files}f_{fsize}.png"
            fig_hm.savefig(hm_out, dpi=300, bbox_inches='tight')
            plt.close(fig_hm)

            print(f"Saved bar, box, and heatmap for {group_name}, {num_files}f, {fsize}")

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
