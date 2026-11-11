#!/usr/bin/env python3
"""Illustrate size / aggregate bandwidth; these are not measured file latencies."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


OUTPUT = Path(__file__).resolve().parents[3] / "results/workloads/plots/dlio/transfer-estimates"
CASES = (
    ("SSD · 4 fast / 0 slow", 180, "#4b84ba"),
    ("SSD · 2 fast / 2 slow", 120, "#4b84ba"),
    ("SSD · 0 fast / 4 slow", 60, "#4b84ba"),
    ("OST/server cache", 263, "#4d9d7c"),
    ("Client cache", 9532, "#4d9d7c"),
)


def draw(ax, name, size_mib, subtitle, max_ms):
    rates = [rate for _, rate, _ in CASES]
    durations = [1000 * size_mib / rate for rate in rates]
    ax.set_facecolor("white")
    positions = list(range(len(CASES)))
    bars = ax.barh(positions, durations, color=[c for _, _, c in CASES], height=.58)
    ax.invert_yaxis()
    ax.set_yticks(positions, [label for label, _, _ in CASES], fontsize=12)
    ax.set_xlabel("Idealized bulk-transfer equivalent (milliseconds)", fontsize=11)
    ax.set_xlim(0, max_ms * 1.43)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for bar, duration, rate in zip(bars, durations, rates):
        ax.text(bar.get_width()+max_ms*.018, bar.get_y()+bar.get_height()/2,
                f"{duration:,.2f} ms  ({rate:,} MiB/s)", va="center", fontsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"{name}  ·  {subtitle}", fontsize=14, fontweight="bold", pad=10)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.patch.set_facecolor("#f5f8fc")
    fig.suptitle("CosmoFlow vs 3D U-Net: SSD mix and cache estimates", fontsize=19, fontweight="bold", y=.98)
    max_ms = 1000 * 146 / min(rate for _, rate, _ in CASES)
    draw(axes[0], "CosmoFlow", 2.70, "2.70 MiB (~2.83 MB) configured mean record", max_ms)
    draw(axes[1], "3D U-Net", 146, "146 MiB illustration; configured mean is 139.8 MiB (~146.6 MB)", max_ms)
    fig.text(.5,.075,"Estimate = file size ÷ rate. SSD rates sum configured target caps; cache rates are separate bulk microbenchmarks.",
             ha="center",fontsize=9.5,color="#364b63")
    fig.text(.5,.045,"Illustrative throughput equivalents—not measured per-file latency. Both panels use the same horizontal scale.",
             ha="center",fontsize=9.5,color="#8a4b16")
    fig.text(.5,.022,"Assumes perfect aggregate SSD capacity; excludes open/metadata, queues, contention and framework overhead; does not establish cache hits.",
             ha="center",fontsize=9.5,color="#8a4b16")
    fig.tight_layout(rect=(.01,.10,.99,.95),h_pad=2.5)
    fig.savefig(OUTPUT/"cosmoflow-unet-estimates.png",dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    for old in ("cosmoflow-2.70MiB.png", "unet-146MiB.png"):
        (OUTPUT/old).unlink(missing_ok=True)
    print(f"Saved combined diagram to {OUTPUT/'cosmoflow-unet-estimates.png'}")


if __name__ == "__main__":
    main()
