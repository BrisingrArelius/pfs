#!/usr/bin/env python3
"""
compare_pools.py — cross-run comparison of the same eval read on different
BeeGFS storage pools (Default / HDD / SSD).

Where plot_io.py describes *one* run in depth, this answers the only question
the pool experiment exists to ask: does the media under the checkpoint change
what the reader sees? Three figures, three runs each.

    C1  read bandwidth        MB/s, one bar per run
    C2  read latency          per-operation service time, ms, log y
    C3  open latency          per-file open cost, ms, log y  (see caveat below)

Usage
-----
    python3 compare_pools.py \\
        --run "Default (pool 1)=/mnt/beegfs/pfs/ckpt_experiment/results/eval_..._183817" \\
        --run "HDD (pool 3)=/mnt/beegfs/pfs/ckpt_experiment/results_pool3/eval_..." \\
        --run "SSD (pool 6)=/mnt/beegfs/pfs/ckpt_experiment/results_pool6/eval_..." \\
        --out-dir plots/pool_compare

Each PREFIX is what `parse_darshan.py --out PREFIX` was given, which is also
what `02_eval_read.sh` used for `--summary`. So one prefix names all of:

    PREFIX.summary.json     reader's own view: total_bytes, per-rank io_s
    PREFIX.ops.csv          per-operation offsets and start/end times (DXT)
    PREFIX.files.csv        per-file rollup, including open counts
    PREFIX.darshanlogs/     the raw logs, needed for C3

Two honesty notes
-----------------
* **Bandwidth** is `total_bytes / max(per-rank io_s)`, i.e. gated by the
  slowest rank, because that is what a real job waits for. The mean across
  ranks would flatter every run equally and hide stragglers, which on a
  shared PFS are the interesting failure.

* **Open latency is not directly traced.** DXT records read and write
  segments only -- there is no open segment -- so C3 falls back to Darshan's
  aggregate POSIX counters. Two sources, in order of preference:

    1. `POSIX_F_OPEN_END_TIMESTAMP - POSIX_F_OPEN_START_TIMESTAMP` when the
       file was opened exactly once. That interval brackets one open() and is
       a true open latency.
    2. `POSIX_F_META_TIME / POSIX_OPENS` otherwise. This is an **upper
       bound**: Darshan's META_TIME lumps open, stat, seek and close
       together, so attributing all of it to opens overstates them.

  Which source was used is recorded per run in the figure and in stats.json.
  If neither is available the panel is drawn empty rather than omitted, so a
  missing measurement stays visible instead of silently vanishing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour per medium, kept stable across all three figures so a reader can
# follow one pool by eye without re-reading the legend each time.
PALETTE = ["#4a5568", "#c05621", "#2b6cb0", "#2f855a", "#6b46c1"]

# 2.5 GbE on anjuna2. Drawn on C1 because a run sitting on this line is
# network-bound, and a media comparison at that point is measuring the NIC.
NIC_CEILING_MBPS = 312.0

# Reads faster than this are almost certainly page-cache hits, not media.
# plot_io.py uses the same threshold for its "drop_caches did not take" warning.
CACHE_HIT_S = 10e-6


def parse_run(spec: str) -> tuple[str, Path]:
    """Split a `LABEL=PREFIX` argument. The label is free text, so split once
    from the left only -- a Windows-style path would otherwise lose its drive,
    and more realistically a label containing '=' would silently truncate."""
    if "=" not in spec:
        raise SystemExit(f"--run needs LABEL=PREFIX, got {spec!r}")
    label, prefix = spec.split("=", 1)
    return label.strip(), Path(prefix.strip())


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_summary(prefix: Path) -> dict | None:
    p = Path(f"{prefix}.summary.json")
    if not p.exists():
        print(f"  ! {p.name} missing -- bandwidth unavailable for this run")
        return None
    return json.loads(p.read_text())


def load_ops(prefix: Path) -> pd.DataFrame | None:
    p = Path(f"{prefix}.ops.csv")
    if not p.exists():
        print(f"  ! {p.name} missing -- latency unavailable for this run")
        return None
    df = pd.read_csv(p)
    if "op" in df.columns:
        df = df[df["op"].astype(str).str.lower().str.startswith("read")]
    if df.empty:
        print(f"  ! {p.name} holds no read operations")
        return None
    return df


def find_logs(prefix: Path) -> Path | None:
    """02_eval_read.sh harvests the run's logs into PREFIX.darshanlogs."""
    d = Path(f"{prefix}.darshanlogs")
    return d if d.is_dir() and any(d.iterdir()) else None


# ---------------------------------------------------------------------------
# C3's data source: aggregate POSIX counters, via darshan-parser
# ---------------------------------------------------------------------------

_WANT = {"POSIX_OPENS", "POSIX_F_META_TIME",
         "POSIX_F_OPEN_START_TIMESTAMP", "POSIX_F_OPEN_END_TIMESTAMP"}

# Files worth timing: the checkpoint's own. Darshan also records the python
# binary, shared libraries and the log itself, whose open costs are not the
# measurement and would swamp eight .distcp opens by sheer count.
DEFAULT_FILE_MATCH = r"\.distcp$|\.metadata$"


def _counter_fields(line: str) -> tuple[str, str, str] | None:
    """(counter, value, file) from one darshan-parser row, or None.

    darshan-parser prints tab-separated
    `<module> <rank> <record id> <counter> <value> <file> <mount> <fs>`, but
    the column count has moved between releases. Locating the field by its
    `POSIX_` prefix rather than by index survives that; the module column is
    bare `POSIX` with no underscore, so it cannot be mistaken for a counter.
    """
    parts = line.split()
    for i, p in enumerate(parts):
        if i >= 1 and p.startswith("POSIX_") and len(parts) > i + 2:
            return p, parts[i + 1], parts[i + 2]
    return None


def open_latencies_ms(logdir: Path,
                      file_match: str = DEFAULT_FILE_MATCH
                      ) -> tuple[np.ndarray, str]:
    """Per-file open cost in ms, plus a label saying how it was derived.

    Returns an empty array and a reason string when nothing usable is found,
    rather than raising -- one run lacking logs should not kill the figure for
    the other two.
    """
    if not shutil.which("darshan-parser"):
        return np.array([]), "darshan-parser not on PATH"
    keep = re.compile(file_match)

    # (log, file) -> {counter: value}. Keyed per log because each process
    # writes its own, and the same .distcp appears in several of them.
    per: dict[tuple[str, str], dict[str, float]] = {}
    for log in sorted(logdir.glob("*.darshan")):
        try:
            out = subprocess.run(["darshan-parser", str(log)],
                                 capture_output=True, text=True, timeout=300)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"    ! darshan-parser failed on {log.name}: {e}")
            continue
        if out.returncode != 0:
            print(f"    ! darshan-parser returned {out.returncode} on {log.name}")
            continue
        for line in out.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            got = _counter_fields(line)
            if not got:
                continue
            name, raw, fname = got
            if name not in _WANT or not keep.search(fname):
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            per.setdefault((log.name, fname), {})[name] = v

    exact, bounded = [], []
    for (_, fname), c in per.items():
        opens = c.get("POSIX_OPENS", 0)
        if opens < 1:
            continue
        s = c.get("POSIX_F_OPEN_START_TIMESTAMP")
        e = c.get("POSIX_F_OPEN_END_TIMESTAMP")
        # Only trustworthy at one open: with several, start is the first open's
        # start and end the last one's end, so the span covers the reads between.
        if opens == 1 and s is not None and e is not None and e >= s:
            exact.append((e - s) * 1e3)
        elif "POSIX_F_META_TIME" in c:
            bounded.append(c["POSIX_F_META_TIME"] / opens * 1e3)

    # Never mix the two: an exact open() interval and a META_TIME upper bound
    # differ by whatever seek and close cost, so plotting them in one box would
    # make the spread an artefact of which files happened to be opened twice.
    # Preferring `exact` also means the caption stays truthful for every point.
    if exact:
        note = f"open() interval, {len(exact)} files opened once"
        if bounded:
            note += f" ({len(bounded)} multi-open files excluded)"
        return np.array(exact), note
    if bounded:
        return np.array(bounded), f"META_TIME/opens, n={len(bounded)} — UPPER BOUND"
    return np.array([]), "no open timing in logs"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def c1_bandwidth(runs, out: Path, stats: dict) -> None:
    labels, bw, detail = [], [], {}
    for r in runs:
        s = r["summary"]
        if not s:
            continue
        io = max((x["io_s"] for x in s["per_rank"]), default=0)
        if io <= 0:
            continue
        mbps = s["total_bytes"] / io / 1e6
        labels.append(r["label"])
        bw.append(mbps)
        detail[r["label"]] = {
            "MB_per_s": round(mbps, 1),
            "total_bytes": s["total_bytes"],
            "slowest_rank_io_s": io,
            "per_rank_io_s": [x["io_s"] for x in s["per_rank"]],
        }

    fig, ax = plt.subplots(figsize=(7, 5))
    if labels:
        bars = ax.bar(labels, bw, color=PALETTE[:len(labels)], width=.6)
        for b, v in zip(bars, bw):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(NIC_CEILING_MBPS, ls="--", lw=1.2, color="#c53030")
        # Left-aligned: a run at the ceiling puts its value label at the right
        # of the plot, exactly where a right-aligned annotation would land.
        ax.text(.01, NIC_CEILING_MBPS, f"2.5 GbE ceiling ~{NIC_CEILING_MBPS:.0f} MB/s",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8, color="#c53030")
        ax.set_ylim(0, max(max(bw), NIC_CEILING_MBPS) * 1.18)
    else:
        ax.text(.5, .5, "no summary.json found for any run",
                ha="center", va="center")
    ax.set_ylabel("MB/s")
    ax.set_title("C1  eval-read bandwidth by storage pool\n"
                 "total bytes / slowest-rank I/O time", fontsize=11)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "C1_bandwidth.png", dpi=140)
    plt.close(fig)
    stats["C1_bandwidth"] = detail
    print("  C1_bandwidth.png")


def _latency_panel(ax, series, labels, title, ylabel):
    """Shared box-plot body for C2 and C3.

    Box plot rather than bars: a single mean hides the bimodality that matters
    here (cache hits at microseconds, media at milliseconds), and the whole
    point of comparing media is the shape of the tail.
    """
    # Title and label go on even when there is nothing to draw, so a panel that
    # measured nothing still states what it was meant to measure.
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if not series:
        ax.text(.5, .5, "no data", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        return {}
    # `labels=` was renamed to `tick_labels` in matplotlib 3.9; setting the
    # ticks directly works on every version the cluster and laptop might have.
    bp = ax.boxplot(series, showfliers=False, widths=.5, patch_artist=True,
                    whis=(5, 95), medianprops=dict(color="black", lw=1.6))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(.65)
    detail = {}
    for i, (lab, v) in enumerate(zip(labels, series), start=1):
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        detail[lab] = {"n": int(v.size), "p50_ms": round(float(p50), 4),
                       "p95_ms": round(float(p95), 4),
                       "p99_ms": round(float(p99), 4),
                       "mean_ms": round(float(v.mean()), 4)}
        # Just clear of the box (width .5, so half-width .25), not centred on
        # it -- at these box heights a centred label sits on the median line.
        ax.annotate(f"p50 {p50:.3g}\np99 {p99:.3g}", (i + .28, p50),
                    textcoords="offset points", xytext=(2, -8),
                    fontsize=7.5, color="#2d3748")
    ax.set_yscale("log")
    ax.grid(alpha=.3, axis="y", which="both")
    return detail


def c2_read_latency(runs, out: Path, stats: dict) -> None:
    series, labels, cache_note = [], [], {}
    for r in runs:
        df = r["ops"]
        if df is None:
            continue
        lat = (df["end_s"] - df["start_s"]).to_numpy(dtype=float)
        lat = lat[np.isfinite(lat) & (lat > 0)]
        if lat.size == 0:
            continue
        cache_note[r["label"]] = round(float((lat < CACHE_HIT_S).mean()), 4)
        series.append(lat * 1e3)
        labels.append(r["label"])

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    detail = _latency_panel(
        ax, series, labels,
        "C2  per-read service time by storage pool\n"
        "box = IQR, whiskers = p5-p95, outliers hidden",
        "read latency (ms, log)")
    warn = [f"{k}: {v:.0%}" for k, v in cache_note.items() if v > .25]
    if warn:
        fig.text(.5, .005, "page-cache hits (<10 us) exceed 25% for "
                 + ", ".join(warn) + " — drop_caches did not take",
                 ha="center", fontsize=8, color="#c53030")
    fig.tight_layout()
    fig.savefig(out / "C2_read_latency.png", dpi=140)
    plt.close(fig)
    for k in detail:
        detail[k]["frac_below_10us_cache_hits"] = cache_note.get(k)
    stats["C2_read_latency"] = detail
    print("  C2_read_latency.png")


def c3_open_latency(runs, out: Path, stats: dict, file_match: str) -> None:
    series, labels, sources = [], [], {}
    for r in runs:
        logs = find_logs(r["prefix"])
        if logs is None:
            sources[r["label"]] = "no PREFIX.darshanlogs directory"
            continue
        vals, how = open_latencies_ms(logs, file_match)
        sources[r["label"]] = how
        if vals.size:
            series.append(vals)
            labels.append(r["label"])

    # Room at the bottom for one provenance line per run. Reserved before
    # tight_layout so the note cannot be pushed off the canvas, which is how
    # an "UPPER BOUND" caveat would otherwise get lost.
    fig, ax = plt.subplots(figsize=(7.5, 5.4 + .18 * len(sources)))
    detail = _latency_panel(
        ax, series, labels,
        "C3  per-file open cost by storage pool",
        "open latency (ms, log)")
    shaky = any("UPPER BOUND" in v or v.startswith("no ") for v in sources.values())
    fig.tight_layout(rect=(0, .04 + .028 * len(sources), 1, 1))
    for i, (k, v) in enumerate(reversed(list(sources.items()))):
        fig.text(.5, .012 + .028 * i, f"{k} — {v}", ha="center", fontsize=7.5,
                 color="#c53030" if ("UPPER BOUND" in v or v.startswith("no "))
                 else "#2d3748")
    stats.setdefault("_notes", {})["C3_source_trustworthy"] = not shaky
    fig.savefig(out / "C3_open_latency.png", dpi=140)
    plt.close(fig)
    stats["C3_open_latency"] = {"per_run": detail, "source": sources}
    print("  C3_open_latency.png")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PREFIX",
                    help="repeat once per run, in the order they should be plotted")
    ap.add_argument("--out-dir", type=Path, default=Path("plots/pool_compare"))
    ap.add_argument("--file-match", default=DEFAULT_FILE_MATCH,
                    help="regex selecting which files C3 times "
                         f"(default: {DEFAULT_FILE_MATCH})")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.run:
        label, prefix = parse_run(spec)
        print(f"{label}  <- {prefix}")
        runs.append({"label": label, "prefix": prefix,
                     "summary": load_summary(prefix), "ops": load_ops(prefix)})

    if not any(r["summary"] or r["ops"] is not None for r in runs):
        raise SystemExit("none of the given prefixes resolved to usable data")

    stats: dict = {"runs": {r["label"]: str(r["prefix"]) for r in runs}}
    c1_bandwidth(runs, out, stats)
    c2_read_latency(runs, out, stats)
    c3_open_latency(runs, out, stats, args.file_match)

    (out / "compare_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  compare_stats.json\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
