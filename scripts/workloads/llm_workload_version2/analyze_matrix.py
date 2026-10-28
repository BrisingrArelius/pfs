#!/usr/bin/env python3
"""
analyze_matrix.py — average the F/S rank-balance matrix over its repetitions
and plot the comparison.

    python3 analyze_matrix.py --results /mnt/beegfs/$USER/ckpt_experiment/llmv2_results

Reads, from --results:
    runs.tsv                 written by 05_run_matrix.sh (only status=ok rows used)
    configs.json             written by 04_make_configs.sh
    <tag>.ops.csv            per-operation Darshan DXT trace (parse_darshan.py)
    <tag>.summary.json       eval_load.py's own per-rank timers

Writes, to --out (default <results>/analysis):
    run_metrics.csv          one row per run
    matrix_summary.csv/.md   per config: mean, std, min, max over reps
    latency_by_medium.csv    per run x op class x medium
    file_times.csv           per run x rank x file: when it was read, how fast
    fig1..fig8 *.png

Definitions (all from the Darshan trace, on the common start_abs clock):
    load time        slowest rank's dcp.load wall time (eval_load.py io_s).
                     This is what the job sees.
    rank N time      first read of the run -> rank N's last read
    agg bandwidth    bytes read by both ranks / (first read -> last read)
    rank bandwidth   rank's bytes / (its first read -> its last read)
    latency          end - start of one blocking pread, split into
                       framing  <= one 512 KiB chunk, so a single target
                       payload  >  one chunk, striped over the file's targets
    medium           from the op's real targets: F only, S only, or F+S
                     (an op that spans both media; only D files do this)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MiB = 1048576
CHUNK = 512 * 1024

LABEL = {"allF": "All F", "allS": "All S", "unbalanced": "Unbalanced",
         "balanced": "Balanced", "onefast": "One file fast", "D": "Default pool D"}
# Estimated run time (minutes) from experiment_context.md, drawn as a band.
PREDICTED = {"allF": (2, 2.5), "allS": (8, 10), "unbalanced": (8, 10),
             "balanced": (5, 6.3), "onefast": (8, 10), "D": (7.5, 8.5)}

# Colours. Media carry meaning across every figure: F blue, S orange, ops that
# span both media aqua. Ranks and single-series totals use neutral greys so
# they never read as a medium. The six-config CDF uses the categorical order.
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e5e4e0", "#fcfcfb"
MEDIUM_C = {"F": "#2a78d6", "S": "#eb6834", "F+S": "#1baf7a"}
RANK_C = {0: "#52514e", 1: "#a8a7a0"}
TOTAL_C = "#52514e"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def rank_pattern(cfgs: dict, cfg: str) -> str:
    """'FFFF | FFSS' — each rank's files in read order."""
    p = cfgs["placement"][cfg]
    return " | ".join("".join(p[f"__{i}_0.distcp"] for i in cfgs["rank_files"][r])
                      for r in ("0", "1"))


def op_medium(targets, F: set, S: set) -> str:
    if not isinstance(targets, str) or not targets:
        return "?"
    t = {int(x) for x in targets.split(",")}
    if t <= F:
        return "F"
    if t <= S:
        return "S"
    return "F+S"


def load_run(res: Path, row, cfgs: dict):
    ops = pd.read_csv(res / f"{row.tag}.ops.csv")
    ops = ops[ops.op == "read"]
    if ops["rank"].isna().all():
        # parse_darshan maps pid -> rank only when given --emit-reads
        raise ValueError("ops.csv has no rank column values; re-parse with "
                         "--emit-reads (see RUNBOOK troubleshooting)")
    ops = ops[ops["rank"].notna()].copy()
    ops["rank"] = ops["rank"].astype(int)
    aligned = "start_abs" in ops and ops["start_abs"].notna().all()
    ops["t0"] = ops["start_abs"] if aligned else ops["start_s"]
    ops["t1"] = ops["end_abs"] if aligned else ops["end_s"]
    origin = ops["t0"].min()
    ops["t0"] -= origin
    ops["t1"] -= origin
    ops["lat_ms"] = (ops["end_s"] - ops["start_s"]) * 1e3
    ops["cls"] = np.where(ops["length"] <= CHUNK, "framing", "payload")
    F, S = set(cfgs["targets"]["F"]), set(cfgs["targets"]["S"])
    if "targets" in ops:
        ops["medium"] = [op_medium(t, F, S) for t in ops["targets"]]
    else:  # pre-target-map trace: fall back to the file's assigned copy
        pl = cfgs["placement"][row.config]
        ops["medium"] = ops["file"].map(lambda f: pl.get(f, pl["__0_0.distcp"]))
    summ = json.load(open(res / f"{row.tag}.summary.json"))
    return ops, summ, aligned


def run_metrics(row, ops: pd.DataFrame, summ: dict, aligned: bool) -> dict:
    m = {"config": row.config, "rep": int(row.rep), "tag": row.tag,
         "aligned_clock": aligned,
         "load_s": max(r["io_s"] for r in summ["per_rank"]),
         "wall_s": ops["t1"].max(), "bytes": int(ops["length"].sum())}
    m["agg_MiBs"] = m["bytes"] / MiB / m["wall_s"]
    for r, g in ops.groupby("rank"):
        m[f"rank{r}_s"] = g["t1"].max()
        m[f"rank{r}_MiBs"] = g["length"].sum() / MiB / (g["t1"].max() - g["t0"].min())
    m["imbalance_s"] = abs(m.get("rank0_s", 0) - m.get("rank1_s", 0))
    for cls, g in ops.groupby("cls"):
        lat = g["lat_ms"]
        m[f"{cls}_n"] = len(lat)
        m[f"{cls}_mean_ms"] = lat.mean()
        for q in (50, 95, 99):
            m[f"{cls}_p{q}_ms"] = lat.quantile(q / 100)
    return m


def file_times(row, ops: pd.DataFrame) -> pd.DataFrame:
    """Per rank x file the rank really reads (not the ~6 MB of cross-reads)."""
    g = ops.groupby(["rank", "file"]).agg(start=("t0", "min"), end=("t1", "max"),
                                          bytes=("length", "sum"),
                                          medium=("medium", lambda s: "/".join(sorted(set(s)))))
    g = g[g["bytes"] > 100e6].reset_index()
    g["dur_s"] = g["end"] - g["start"]
    g["MiBs"] = g["bytes"] / MiB / g["dur_s"]
    g.insert(0, "config", row.config)
    g.insert(1, "rep", int(row.rep))
    return g


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------


def xlabels(ax, order, cfgs):
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{LABEL.get(c, c)}\n{rank_pattern(cfgs, c)}" for c in order],
                       fontsize=8.5)
    ax.tick_params(axis="x", length=0)


def bars_with_reps(ax, x, per_rep: list[np.ndarray], color, width=0.6, label=None,
                   fmt="{:.1f}", text_color=INK):
    """Mean bar, +-1 std whisker, one dot per rep, value label on top."""
    means = np.array([v.mean() if len(v) else np.nan for v in per_rep])
    stds = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in per_rep])
    ax.bar(x, means, width, color=color, label=label, edgecolor=SURF, linewidth=2,
           zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=1.2,
                capsize=3, zorder=3)
    for xi, v in zip(x, per_rep):
        ax.scatter(np.full(len(v), xi) + np.linspace(-width / 6, width / 6, len(v)),
                   v, s=14, color=SURF, edgecolor=INK, linewidth=0.8, zorder=4)
    top = ax.get_ylim()[1]
    for xi, mu, sd in zip(x, means, stds):
        if np.isfinite(mu):
            ax.text(xi, mu + sd + top * 0.015, fmt.format(mu), ha="center",
                    va="bottom", fontsize=8, color=text_color)
    return means


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"  {out / name}")


def reps_of(df, cfg, col):
    return df.loc[df.config == cfg, col].dropna().to_numpy()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_load_time(df, order, cfgs, out, n_reps):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(order))
    for xi, c in zip(x, order):
        lo, hi = PREDICTED.get(c, (np.nan, np.nan))
        ax.fill_between([xi - 0.42, xi + 0.42], lo, hi, color=GRID, zorder=1,
                        label="estimated range" if xi == 0 else None)
    bars_with_reps(ax, x, [reps_of(df, c, "load_s") / 60 for c in order], TOTAL_C,
                   label=f"measured, mean ± 1 sd (dots: {n_reps} reps)")
    xlabels(ax, order, cfgs)
    ax.set_ylabel("load time (min)")
    ax.set_title("Eval checkpoint load time: slowest rank's dcp.load")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fig1_load_time.png")


def fig_rank_time(df, order, cfgs, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(order))
    w = 0.38
    for r, dx in ((0, -w / 2), (1, w / 2)):
        bars_with_reps(ax, x + dx, [reps_of(df, c, f"rank{r}_s") / 60 for c in order],
                       RANK_C[r], width=w, label=f"rank {r}")
    xlabels(ax, order, cfgs)
    ax.set_ylabel("time to rank's last read (min)")
    ax.set_title("Per-rank completion: the job waits for the slower rank")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fig2_rank_completion.png")


def fig_bandwidth(df, order, cfgs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(order))
    ax = axes[0]
    bars_with_reps(ax, x, [reps_of(df, c, "agg_MiBs") for c in order], TOTAL_C,
                   fmt="{:.0f}")
    nF, nS = len(cfgs["targets"]["F"]), len(cfgs["targets"]["S"])
    for v, txt, ls in ((nF * cfgs["rbps"]["F"] / MiB, f"F pool ceiling ({nF} × 20 MiB/s)", "--"),
                       (nS * cfgs["rbps"]["S"] / MiB, f"S pool ceiling ({nS} × 5 MiB/s)", ":")):
        ax.axhline(v, color=INK2, linestyle=ls, linewidth=1.2, zorder=5, label=txt)
    ax.legend(loc="upper right", fontsize=8.5)
    xlabels(ax, order, cfgs)
    ax.set_ylabel("MiB/s")
    ax.set_title("Aggregate client read bandwidth (both ranks)")
    ax = axes[1]
    w = 0.38
    for r, dx in ((0, -w / 2), (1, w / 2)):
        bars_with_reps(ax, x + dx, [reps_of(df, c, f"rank{r}_MiBs") for c in order],
                       RANK_C[r], width=w, label=f"rank {r}", fmt="{:.0f}")
    xlabels(ax, order, cfgs)
    ax.set_ylabel("MiB/s")
    ax.set_title("Per-rank read bandwidth")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, out, "fig3_bandwidth.png")


def fig_latency(df, order, cfgs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(order))
    for ax, cls, desc in ((axes[0], "framing", "framing reads (≤ 512 KiB, one target)"),
                          (axes[1], "payload", "payload reads (> 512 KiB, striped)")):
        bars_with_reps(ax, x, [reps_of(df, c, f"{cls}_mean_ms") for c in order],
                       TOTAL_C, fmt="{:.0f}", label="mean (± 1 sd over reps)")
        p95 = [reps_of(df, c, f"{cls}_p95_ms").mean() for c in order]
        ax.scatter(x, p95, marker="D", s=30, color=SURF, edgecolor=INK, linewidth=1.2,
                   zorder=5, label="p95 (mean over reps)")
        xlabels(ax, order, cfgs)
        ax.set_ylabel("latency per read (ms)")
        ax.set_title(f"Read latency: {desc}")
        ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fig4_latency.png")


def fig_latency_medium(lat, order, cfgs, out):
    meds = [m for m in ("F", "S", "F+S") if m in set(lat["medium"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(order))
    w = 0.8 / max(len(meds), 1)
    for ax, cls in zip(axes, ("framing", "payload")):
        sub = lat[lat.cls == cls]
        for k, m in enumerate(meds):
            dx = (k - (len(meds) - 1) / 2) * w
            per = [sub[(sub.config == c) & (sub.medium == m)]["mean_ms"].to_numpy()
                   for c in order]
            bars_with_reps(ax, x + dx, per, MEDIUM_C[m], width=w, fmt="{:.0f}",
                           label={"F": "F targets only", "S": "S targets only",
                                  "F+S": "spans F and S"}[m])
        xlabels(ax, order, cfgs)
        ax.set_ylabel("mean latency per read (ms)")
        ax.set_title(f"{cls.capitalize()} read latency by medium touched")
        ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fig5_latency_by_medium.png")


def fig_latency_cdf(all_ops, order, cfgs, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, c in enumerate(order):
        v = np.sort(all_ops.loc[(all_ops.config == c) & (all_ops.cls == "payload"),
                                "lat_ms"].to_numpy())
        if not len(v):
            continue
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=CATEGORICAL[k % 6],
                linewidth=2, label=f"{LABEL.get(c, c)}  ({rank_pattern(cfgs, c)})")
    ax.set_xscale("log")
    ax.set_xlabel("payload read latency (ms, log scale)")
    ax.set_ylabel("fraction of reads")
    ax.set_title("Payload read latency CDF (all reps pooled)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, out, "fig6_latency_cdf.png")


def fig_timeline(all_ops, df, order, cfgs, out, bin_s=5.0):
    n = len(order)
    cols = 3
    rows = int(np.ceil(n / cols))
    tmax = all_ops["t1"].max()
    edges = np.arange(0, tmax + bin_s, bin_s)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), sharex=True,
                             sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, order):
        curves = []
        for tag, g in all_ops[all_ops.config == c].groupby("tag"):
            # credit each op's bytes to the bin its midpoint falls in
            mid = (g["t0"] + g["t1"]) / 2
            h, _ = np.histogram(mid, edges, weights=g["length"])
            curves.append(h / MiB / bin_s)
            ax.plot(edges[:-1] / 60, curves[-1], color=GRID, linewidth=1)
        if curves:
            ax.plot(edges[:-1] / 60, np.mean(curves, axis=0), color=TOTAL_C, linewidth=2)
        ax.set_title(f"{LABEL.get(c, c)}  {rank_pattern(cfgs, c)}", fontsize=10)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("MiB/s")
    for ax in axes[-1]:
        ax.set_xlabel("time since first read (min)")
    fig.suptitle(f"Client read throughput over time ({bin_s:.0f} s bins; "
                 "thin: each rep, thick: mean)", fontweight="bold")
    save(fig, out, "fig7_throughput_timeline.png")


def fig_gantt(files, df, order, cfgs, out):
    """Which file each rank was reading when, for the rep nearest the median."""
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.25 * n + 0.8), sharex=True,
                             squeeze=False)
    for ax, c in zip(axes[:, 0], order):
        runs = df[df.config == c]
        if runs.empty:
            ax.set_visible(False)
            continue
        med = runs.iloc[(runs["load_s"] - runs["load_s"].median()).abs().argmin()]
        f = files[(files.config == c) & (files.rep == med.rep)]
        pl = cfgs["placement"][c]
        for _, r in f.iterrows():
            src = pl.get(r.file, "?")
            colour = MEDIUM_C.get(src, MEDIUM_C["F+S"])  # D copy -> mixed colour
            ax.barh(r["rank"], r.dur_s / 60, left=r.start / 60, height=0.62,
                    color=colour, edgecolor=SURF, linewidth=2)
            ax.text((r.start + r.dur_s / 2) / 60, r["rank"],
                    f"{r.file.split('_')[2] if r.file.startswith('__') else r.file}"
                    f" {src}", ha="center", va="center", fontsize=7.5, color=INK)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["rank 0", "rank 1"])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{LABEL.get(c, c)}  (rep {int(med.rep)}, load {med.load_s / 60:.1f} min)",
                     fontsize=9.5, loc="left")
    axes[-1, 0].set_xlabel("time since first read (min)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MEDIUM_C[k]) for k in ("F", "S", "F+S")]
    fig.legend(handles, ["F copy", "S copy", "D copy (mixed pool)"], loc="upper right",
               ncol=3, fontsize=8.5)
    fig.suptitle("Per-file read spans (label: file number and copy)", fontweight="bold",
                 x=0.01, ha="left")
    save(fig, out, "fig8_file_gantt.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def summary_table(df, order, cfgs) -> tuple[pd.DataFrame, str]:
    cols = ["load_s", "wall_s", "rank0_s", "rank1_s", "imbalance_s", "agg_MiBs",
            "rank0_MiBs", "rank1_MiBs", "framing_mean_ms", "framing_p95_ms",
            "payload_mean_ms", "payload_p95_ms"]
    cols = [c for c in cols if c in df]
    g = df.groupby("config")[cols].agg(["mean", "std", "min", "max", "count"])
    g = g.reindex([c for c in order if c in g.index])
    g.columns = [f"{a}_{b}" for a, b in g.columns]

    def ms(c, col, scale=1, nd=1):
        mu, sd = g.loc[c, f"{col}_mean"] / scale, g.loc[c, f"{col}_std"] / scale
        return f"{mu:.{nd}f} ± {0 if np.isnan(sd) else sd:.{nd}f}"

    lines = ["| Config | Rank 0 \\| Rank 1 | Reps | Load time (min) | Estimate (min) | "
             "Rank 0 (min) | Rank 1 (min) | Agg BW (MiB/s) | Framing mean / p95 (ms) | "
             "Payload mean / p95 (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in g.index:
        lo, hi = PREDICTED.get(c, (np.nan, np.nan))
        lines.append(
            f"| {LABEL.get(c, c)} | {rank_pattern(cfgs, c).replace('|', chr(92) + '|')} | "
            f"{int(g.loc[c, 'load_s_count'])} | "
            f"{ms(c, 'load_s', 60, 2)} | {lo}–{hi} | {ms(c, 'rank0_s', 60, 2)} | "
            f"{ms(c, 'rank1_s', 60, 2)} | {ms(c, 'agg_MiBs')} | "
            f"{g.loc[c, 'framing_mean_ms_mean']:.0f} / {g.loc[c, 'framing_p95_ms_mean']:.0f} | "
            f"{g.loc[c, 'payload_mean_ms_mean']:.0f} / {g.loc[c, 'payload_p95_ms_mean']:.0f} |")
    lines.append("")
    lines.append("± is one standard deviation across reps. Latency columns are the mean "
                 "over reps of each run's per-read mean and p95.")
    return g, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset / order of configs to plot (default: configs.json order)")
    args = ap.parse_args()
    res = args.results
    out = args.out or res / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    cfgs = json.load(open(res / "configs.json"))
    runs = pd.read_csv(res / "runs.tsv", sep="\t", dtype={"rep": int})
    runs = runs[runs.status == "ok"]
    # a (config, rep) re-run after a failure counts once: keep the latest
    runs = runs.sort_values("start_epoch").drop_duplicates(["config", "rep"], keep="last")
    if runs.empty:
        sys.exit(f"no successful runs in {res / 'runs.tsv'}")

    metrics, lat_rows, file_rows, op_frames = [], [], [], []
    for row in runs.itertuples():
        try:
            ops, summ, aligned = load_run(res, row, cfgs)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {row.tag}: {e}", file=sys.stderr)
            continue
        if not aligned:
            print(f"  WARNING {row.tag}: no start_abs column; ranks on separate clocks",
                  file=sys.stderr)
        metrics.append(run_metrics(row, ops, summ, aligned))
        file_rows.append(file_times(row, ops))
        for (cls, med), g in ops.groupby(["cls", "medium"]):
            lat_rows.append({"config": row.config, "rep": int(row.rep), "cls": cls,
                             "medium": med, "n": len(g), "mean_ms": g.lat_ms.mean(),
                             "p50_ms": g.lat_ms.median(), "p95_ms": g.lat_ms.quantile(.95)})
        ops = ops[["rank", "file", "length", "t0", "t1", "lat_ms", "cls", "medium"]].copy()
        ops["config"], ops["tag"] = row.config, row.tag
        op_frames.append(ops)

    df = pd.DataFrame(metrics)
    lat = pd.DataFrame(lat_rows)
    files = pd.concat(file_rows, ignore_index=True)
    all_ops = pd.concat(op_frames, ignore_index=True)
    order = [c for c in (args.configs or cfgs["order"]) if c in set(df.config)]
    n_reps = int(df.groupby("config").size().max())

    df.sort_values(["config", "rep"]).to_csv(out / "run_metrics.csv", index=False)
    lat.to_csv(out / "latency_by_medium.csv", index=False)
    files.to_csv(out / "file_times.csv", index=False)
    g, md = summary_table(df, order, cfgs)
    g.to_csv(out / "matrix_summary.csv")
    (out / "matrix_summary.md").write_text(md + "\n")
    print(md)
    print()

    fig_load_time(df, order, cfgs, out, n_reps)
    fig_rank_time(df, order, cfgs, out)
    fig_bandwidth(df, order, cfgs, out)
    fig_latency(df, order, cfgs, out)
    fig_latency_medium(lat, order, cfgs, out)
    fig_latency_cdf(all_ops, order, cfgs, out)
    fig_timeline(all_ops, df, order, cfgs, out)
    fig_gantt(files, df, order, cfgs, out)


if __name__ == "__main__":
    main()
