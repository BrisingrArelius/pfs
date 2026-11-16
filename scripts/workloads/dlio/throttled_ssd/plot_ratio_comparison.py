#!/usr/bin/env python3
"""Compare all-fast/all-slow end-to-end throughput across completed studies."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT/"results/workloads/runs"
PLOTS = ROOT/"results/workloads/plots/dlio"
WORKLOADS = {"cosmoflow":"cosmoflow", "unet":"unet3d"}


def collect():
    results=[]
    fast_rates=set()
    for root in sorted(RUNS.glob("throttled_ssd_*")):
        journals=sorted(root.glob("study-*.json"))
        if len(journals)!=1:
            if journals:
                raise ValueError(f"Expected one study journal, found {len(journals)}: {root}")
            continue
        study=json.loads(journals[0].read_text())
        if study.get("status")!="complete":
            continue
        identity=study["identity"]
        fast,slow=identity["fast_MiB_s"],identity["slow_MiB_s"]
        ratio=fast/slow
        if not math.isclose(ratio,float(root.name.removeprefix("throttled_ssd_")),rel_tol=1e-9):
            raise ValueError(f"Directory does not match recorded fast/slow rates: {root}")
        fast_rates.add(fast)
        for workload, folder in WORKLOADS.items():
            summary=PLOTS/root.name/workload/"comparison_runs.csv"
            if not summary.is_file():
                raise ValueError(f"Generate standard performance plots first: {summary}")
            with summary.open(newline="") as file:
                records=list(csv.DictReader(file))
            pool_stats={}
            for pool in (14,18):
                progress=study["workloads"][workload]["pools"][str(pool)]
                if progress["completed"]!=[1,2,3]:
                    raise ValueError(f"Incomplete pool {pool}: {root}")
                case=root/folder/Path(progress["output"]).name
                condition=json.loads((case/"condition.json").read_text())
                if (condition["pool"]!=pool or condition["fast_MiB_s"]!=fast
                        or condition["slow_MiB_s"]!=slow or len(condition["fast"])!=(4 if pool==14 else 0)):
                    raise ValueError(f"Incorrect pool or rates: {case}")
                selected=[item for item in records if int(item["pool"])==pool]
                if sorted(int(item["repetition"]) for item in selected)!=[1,2,3]:
                    raise ValueError(f"Expected three runs for pool {pool}: {summary}")
                speeds=[]
                for row in selected:
                    run=case/f"rep{row['repetition']}-pool{pool}"
                    config=yaml.safe_load((run/".hydra/config.yaml").read_text())["workload"]
                    if not config["workflow"].get("profiling") or config["profiling"]["profiler"]!="darshan":
                        raise ValueError(f"Darshan profiling not enabled: {run}")
                    recorded_source=Path(row["source"])
                    if not recorded_source.is_absolute():
                        recorded_source=ROOT/recorded_source
                    if recorded_source.resolve()!=run.resolve():
                        raise ValueError(f"Performance data belongs to another run: {row['source']}")
                    samples=int(row["total_processed_samples"])
                    seconds=float(row["whole_run_seconds"])
                    speed=float(row["end_to_end_samples_s"])
                    expected=40960 if workload=="cosmoflow" else 800
                    if samples!=expected or seconds<=0 or not math.isclose(speed,samples/seconds,rel_tol=1e-9):
                        raise ValueError(f"Unexpected processed samples or throughput: {run}")
                    speeds.append(speed)
                pool_stats[pool]=(statistics.mean(speeds),statistics.stdev(speeds))
            high, high_sd=pool_stats[14]
            low, low_sd=pool_stats[18]
            results.append({"device_speed_ratio":ratio,"device_speed_advantage_pct":(ratio-1)*100,
                            "fast_MiB_s":fast,"slow_MiB_s":slow,
                            "workload":workload,"fast_pool":14,"slow_pool":18,"repetitions_per_pool":3,
                            "fast_samples_s_mean":high,"fast_samples_s_sd":high_sd,
                            "slow_samples_s_mean":low,"slow_samples_s_sd":low_sd,
                            "speedup_fast_over_slow":high/low,"throughput_gain_pct":(high/low-1)*100,
                            "study":str(journals[0].relative_to(ROOT))})
    if len(fast_rates)!=1:
        raise ValueError(f"Studies have different fast caps: {fast_rates}")
    ratios=[row["device_speed_ratio"] for row in results if row["workload"]=="unet"]
    if len(ratios)!=len(set(ratios)):
        raise ValueError("More than one completed study has the same device-speed ratio")
    if not results:
        raise ValueError("No completed studies with standard performance plots found")
    return sorted(results,key=lambda row:(row["device_speed_ratio"],row["workload"]))


def render(rows, output):
    output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10.5,6))
    for name,label,color,offset in (("cosmoflow","CosmoFlow","#2878B5",-18),
                                    ("unet","3D U-Net","#D08734",10)):
        points=[row for row in rows if row["workload"]==name]
        ax.plot([row["device_speed_advantage_pct"] for row in points],
                [row["throughput_gain_pct"] for row in points],
                "o-",color=color,linewidth=2.5,markersize=7,label=label)
        for row in points:
            ax.annotate(f"{row['throughput_gain_pct']:.1f}%",
                        (row["device_speed_advantage_pct"],row["throughput_gain_pct"]),
                        xytext=(0,offset),textcoords="offset points",ha="center",fontsize=9,color=color)
    ax.axhline(0,color=".5",linestyle="--",linewidth=1,label="Equal performance")
    advantages=sorted({row["device_speed_advantage_pct"] for row in rows})
    ax.set_xticks(advantages,[f"{value:g}%" for value in advantages])
    ax.set_ylim(min(0,min(row["throughput_gain_pct"] for row in rows))-4,
                max(row["throughput_gain_pct"] for row in rows)*1.15)
    ax.set_xlabel("Configured fast target speed advantage over slow target (%)")
    ax.set_ylabel("All-fast vs all-slow end-to-end throughput gain (%)")
    ax.set_title("Application throughput gain vs SSD speed advantage",fontweight="bold")
    ax.grid(alpha=.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper left")
    fig.text(.5,.025,"Gain = (mean of 3 all-fast runs ÷ mean of 3 all-slow runs − 1) × 100; full DLIO time includes checkpoints.\n"
             "Fast cap 45 MiB/s; slow cap varies. Darshan enabled. Points are independent studies, not paired runs.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.105,1,1))
    fig.savefig(output/"throughput-gain-percent.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    with (output/"throughput-gain-vs-speed-ratio.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PLOTS/"ratio_comparison")
    args=parser.parse_args()
    rows=collect()
    render(rows,args.output)
    print(f"Plotted {len(rows)//2} completed ratios and saved throughput-gain-percent.png and source CSV to {args.output}")


if __name__=="__main__":
    main()
