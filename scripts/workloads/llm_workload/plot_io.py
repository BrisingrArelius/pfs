#!/usr/bin/env python3
"""
plot_io.py — turn one run's parsed Darshan output into the figure set.

Consumes what `parse_darshan.py --out PREFIX` writes:

    PREFIX.ops.csv      one row per POSIX operation (the rich source)
    PREFIX.files.csv    one row per file
    PREFIX.darshan.json rollup, pid->rank map, clock skew, stripe geometry

and emits a PNG per figure plus `stats.json` / `stats.md` holding every number
the figures are drawn from, so a claim in the paper can be traced to a value
without re-reading a plot.

    python3 plot_io.py --prefix results/eval_llama31_8b_..._183817 \\
                       --out-dir plots/eval_tp2

Figure groups, and what each is for
-----------------------------------
size      P1  read-size histogram + CDF
          P2  count-weighted vs byte-weighted size distribution
spatial   P3  offset-vs-time scatter per file  (the access-pattern fingerprint)
          P4  per-stream transition classification
          P5  signed-gap histogram
sharing   P6  file x rank access matrix
          P7  per-file sharing classification (private / partial / full)
temporal  P8  IOPS and MB/s over time
          P9  per-file activity Gantt
          P10 inter-arrival time distribution
per-file  P11 per-file summary table
          P12 re-read map: how many times each block of each file was read
striping  P13 bytes per stripe target over time
latency   P14 per-operation service-time distribution
          P15 latency vs request size

On reading these honestly
-------------------------
* Latency (P14/P15) is `end_s - start_s` of a blocking POSIX read. DCP's
  reader is synchronous and single-threaded per rank (verified against the
  installed torch), so that interval genuinely brackets the read. It does not
  decompose into client cache / network / server queue / media, and page-cache
  hits appear as sub-microsecond outliers -- if you see a large population
  below ~10 us, `drop_caches` did not take and the run is warm.
* Non-MPI Darshan writes one log per process with its own time origin. The
  `start_abs`/`end_abs` columns put all ranks on a common clock; every
  cross-rank figure here uses them and says so if they are missing.
* Darshan sees syscalls. Reads served from Python's BufferedReader without
  entering the kernel are invisible, which is correct -- they cost no I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KIND_COLOR = {"model": "#2b6cb0", "optim": "#c05621", "mixed": "#6b46c1",
              "index": "#2f855a", "": "#718096"}
CLS_COLOR = {"contiguous": "#2f855a", "strided": "#3182ce",
             "forward-jump": "#d69e2e", "backward-seek": "#c53030"}
MAX_PANELS = 16          # cap on per-file small multiples
BLOCK = 524288           # re-read map resolution, matches BeeGFS chunksize


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load(prefix: Path | None, ops_p, files_p, json_p):
    if prefix:
        ops_p = ops_p or Path(f"{prefix}.ops.csv")
        files_p = files_p or Path(f"{prefix}.files.csv")
        json_p = json_p or Path(f"{prefix}.darshan.json")
    if not ops_p or not Path(ops_p).exists():
        raise SystemExit(f"need an ops CSV; {ops_p} not found")

    ops = pd.read_csv(ops_p)
    ops = ops[ops["op"] == "read"].copy()
    if ops.empty:
        raise SystemExit("no read operations in the ops CSV -- nothing to plot")

    # rank may be blank when pid->rank matching failed; fall back to pid
    ops["rank"] = ops["rank"].fillna(-1)
    if (ops["rank"] == -1).all():
        codes = {p: i for i, p in enumerate(sorted(ops["pid"].unique()))}
        ops["rank"] = ops["pid"].map(codes)
        print("  ! no pid->rank map; ranks are synthetic, ordered by pid")
    ops["rank"] = ops["rank"].astype(int)

    # common clock if the parser supplied one, else per-process clocks
    aligned = "start_abs" in ops.columns and ops["start_abs"].notna().all()
    ops["t0"] = ops["start_abs"] if aligned else ops["start_s"]
    ops["t1"] = ops["end_abs"] if aligned else ops["end_s"]
    ops["t0"] -= ops["t0"].min()
    ops["t1"] -= (ops["start_abs"] if aligned else ops["start_s"]).min()
    ops["dur"] = (ops["end_s"] - ops["start_s"]).clip(lower=0)

    files = pd.read_csv(files_p) if files_p and Path(files_p).exists() else None
    meta = json.loads(Path(json_p).read_text()) if json_p and Path(json_p).exists() else {}
    kind = {}
    if files is not None:
        kind = dict(zip(files["file"], files["kind"].fillna("")))
    ops["kind"] = ops["file"].map(kind).fillna("")
    return ops, files, meta, aligned


def _fmt_bytes(n):
    for u, d in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= d:
            return f"{n/d:.2f} {u}"
    return f"{n:.0f} B"


def _save(fig, out: Path, name: str, stats: dict, note: str = "", tight=True):
    p = out / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    stats.setdefault("_figures", {})[name] = note or name
    print(f"  {p.name}")


# ---------------------------------------------------------------------------
# spatial classification -- the definitions the paper has to state
# ---------------------------------------------------------------------------


def classify_streams(ops: pd.DataFrame):
    """Label each transition within a (rank, file) stream, in issue order.

    A *stream* is one rank's operations on one file, ordered by issue time.
    That ordering is meaningful only because DCP's reader is synchronous and
    single-threaded per rank; with a thread pool, concurrent threads would
    share a pid and a sequential stream would look random.

    For consecutive operations n, n+1:

        gap   = offset[n+1] - (offset[n] + length[n])
        delta = offset[n+1] -  offset[n]

        contiguous     gap == 0        the next read starts where this ended
        backward-seek  gap <  0        overlaps or revisits earlier bytes;
                                       this is where re-reads show up
        strided        gap >  0 and delta equals the previous delta
                                       (a repeated constant stride)
        forward-jump   gap >  0 otherwise

    `strided` is deliberately local -- two transitions with the same delta is
    the weakest claim that still means anything, and a stricter run-length rule
    would need a tolerance nobody can justify from first principles.
    """
    rows, gaps = [], []
    for (rank, fname), g in ops.groupby(["rank", "file"], sort=True):
        g = g.sort_values("start_s")
        off = g["offset"].to_numpy()
        ln = g["length"].to_numpy()
        if len(off) < 2:
            rows.append({"rank": rank, "file": fname, "ops": len(off),
                         "contiguous": 0, "strided": 0, "forward-jump": 0,
                         "backward-seek": 0})
            continue
        gap = off[1:] - (off[:-1] + ln[:-1])
        delta = np.diff(off)
        prev_same = np.zeros(len(delta), dtype=bool)
        prev_same[1:] = (delta[1:] == delta[:-1]) & (delta[1:] != 0)
        cls = np.where(gap == 0, "contiguous",
                       np.where(gap < 0, "backward-seek",
                                np.where(prev_same, "strided", "forward-jump")))
        c = Counter(cls.tolist())
        rows.append({"rank": rank, "file": fname, "ops": len(off),
                     **{k: c.get(k, 0) for k in CLS_COLOR}})
        gaps.append(pd.DataFrame({"rank": rank, "file": fname, "gap": gap,
                                  "cls": cls}))
    return pd.DataFrame(rows), (pd.concat(gaps, ignore_index=True)
                                if gaps else pd.DataFrame(columns=["gap", "cls"]))


def classify_sharing(ops: pd.DataFrame, nranks: int):
    """Per-file sharing class, and the workload-level FPP/SSF/PSF label.

    private  read by exactly one rank        -> file-per-process behaviour
    full     read by every rank              -> single-shared-file behaviour
    partial  read by a strict subset >1      -> partially-shared behaviour

    The workload label is the majority of files *by bytes*, because a
    thousand-byte index file shared by everyone should not outvote the shards.
    One run yields one label; the comparison that matters is across layouts and
    reader widths, which needs the per-shard and TP-sweep runs.
    """
    per = ops.groupby("file").agg(readers=("rank", "nunique"),
                                  bytes=("length", "sum")).reset_index()
    per["class"] = np.where(per["readers"] == 1, "private",
                            np.where(per["readers"] >= nranks, "full", "partial"))
    by_bytes = per.groupby("class")["bytes"].sum()
    label = {"private": "FPP", "full": "SSF", "partial": "PSF"}
    top = by_bytes.idxmax() if len(by_bytes) else ""
    return per, label.get(top, "?"), by_bytes.to_dict()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def p1_size_hist(ops, out, stats):
    sizes = ops["length"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.logspace(math.log10(max(sizes.min(), 1)),
                       math.log10(sizes.max() * 1.05), 60)
    bottom = np.zeros(len(bins) - 1)
    for k, g in ops.groupby("kind"):
        h, _ = np.histogram(g["length"], bins=bins)
        ax[0].bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
                  color=KIND_COLOR.get(k, "#718096"), label=k or "unknown")
        bottom += h
    ax[0].set_xscale("log")
    ax[0].set_xlabel("request size (bytes, log)")
    ax[0].set_ylabel("operations")
    ax[0].set_title("P1a  read-size distribution by file kind")
    ax[0].legend(fontsize=8)

    s = np.sort(sizes)
    ax[1].plot(s, np.arange(1, len(s) + 1) / len(s), color="#2b6cb0")
    qs = {}
    for q, c in ((50, "#2f855a"), (95, "#d69e2e"), (99, "#c53030")):
        v = float(np.percentile(s, q))
        qs[f"p{q}"] = v
        ax[1].axvline(v, color=c, ls="--", lw=1,
                      label=f"p{q} = {_fmt_bytes(v)}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("request size (bytes, log)")
    ax[1].set_ylabel("cumulative fraction of operations")
    ax[1].set_title("P1b  CDF")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    stats["size"] = {"ops": int(len(s)), "min": int(s.min()), "max": int(s.max()),
                     "mean": float(s.mean()), **qs}
    _save(fig, out, "P1_request_size", stats, "read-size histogram and CDF")


def p2_count_vs_bytes(ops, out, stats):
    sizes = ops["length"].to_numpy()
    bins = np.logspace(math.log10(max(sizes.min(), 1)),
                       math.log10(sizes.max() * 1.05), 40)
    cnt, _ = np.histogram(sizes, bins=bins)
    byt, _ = np.histogram(sizes, bins=bins, weights=sizes)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    ax[0].bar(bins[:-1], cnt / cnt.sum(), width=np.diff(bins), align="edge",
              color="#c53030")
    ax[0].set_title("P2a  weighted by OPERATION COUNT\n(what seek cost tracks)")
    ax[0].set_ylabel("fraction of operations")
    ax[1].bar(bins[:-1], byt / byt.sum(), width=np.diff(bins), align="edge",
              color="#2b6cb0")
    ax[1].set_title("P2b  weighted by BYTES\n(what bandwidth tracks)")
    ax[1].set_ylabel("fraction of bytes")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("request size (bytes, log)")
        a.grid(alpha=.3)

    # the headline contrast: small ops as a share of count vs of bytes
    small = sizes <= BLOCK
    stats["size_weighting"] = {
        "threshold_bytes": BLOCK,
        "ops_at_or_below": int(small.sum()),
        "frac_of_ops": float(small.mean()),
        "frac_of_bytes": float(sizes[small].sum() / sizes.sum()),
    }
    _save(fig, out, "P2_count_vs_byte_weighted", stats,
          "count- vs byte-weighted size distribution")


def p3_offset_time(ops, out, stats):
    order = (ops.groupby("file")["length"].sum().sort_values(ascending=False))
    sel = list(order.index[:MAX_PANELS])
    n = len(sel)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                             squeeze=False)
    ranks = sorted(ops["rank"].unique())
    cmap = plt.get_cmap("tab10")
    for i, fname in enumerate(sel):
        a = axes[i // cols][i % cols]
        g = ops[ops["file"] == fname]
        for r in ranks:
            gr = g[g["rank"] == r]
            if gr.empty:
                continue
            a.scatter(gr["t0"], gr["offset"] / 1e9, s=3, alpha=.55,
                      color=cmap(r % 10), label=f"rank {r}")
        a.set_title(Path(fname).name[:34], fontsize=8)
        a.set_xlabel("time (s)", fontsize=7)
        a.set_ylabel("offset (GB)", fontsize=7)
        a.tick_params(labelsize=6)
        a.grid(alpha=.25)
        if i == 0:
            a.legend(fontsize=6, markerscale=2)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("P3  offset vs time per file — diagonal = sequential, "
                 "parallel diagonals = strided, cloud = random", fontsize=10)
    stats["p3_files_shown"] = n
    stats["p3_files_total"] = int(ops["file"].nunique())
    _save(fig, out, "P3_offset_vs_time", stats, "access-pattern fingerprint")


def p4_stream_class(streams, out, stats):
    agg = streams.groupby("file")[list(CLS_COLOR)].sum()
    agg = agg.loc[agg.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(7, .6 * len(agg) + 4), 4.8))
    bottom = np.zeros(len(agg))
    idx = np.arange(len(agg))
    for c, col in CLS_COLOR.items():
        ax.bar(idx, agg[c], bottom=bottom, color=col, label=c)
        bottom += agg[c].to_numpy()
    ax.set_xticks(idx)
    ax.set_xticklabels([Path(f).name[:22] for f in agg.index], rotation=45,
                       ha="right", fontsize=7)
    ax.set_ylabel("transitions")
    ax.set_title("P4  access-pattern transitions per file "
                 "(within each rank's stream, issue order)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    tot = agg.sum()
    stats["spatial"] = {k: int(tot[k]) for k in CLS_COLOR}
    stats["spatial"]["contiguous_frac"] = (float(tot["contiguous"] / tot.sum())
                                           if tot.sum() else None)
    _save(fig, out, "P4_pattern_class", stats, "transition classification")


def p5_gap_hist(gaps, out, stats):
    if gaps.empty:
        return
    g = gaps["gap"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lim = max(abs(g).max(), 1)
    pos = np.logspace(0, math.log10(lim + 1), 40)
    bins = np.concatenate([-pos[::-1], [0], pos])
    ax.hist(g, bins=bins, color="#2b6cb0")
    ax.set_xscale("symlog", linthresh=1)
    ax.axvline(0, color="#2f855a", lw=1.5, label="gap = 0 (perfectly contiguous)")
    ax.set_xlabel("gap between end of op n and start of op n+1 (bytes, symlog)")
    ax.set_ylabel("transitions")
    ax.set_title("P5  seek-gap distribution — negative = revisiting bytes "
                 "already read (the over-fetch)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    stats["gaps"] = {
        "zero": int((g == 0).sum()), "negative": int((g < 0).sum()),
        "positive": int((g > 0).sum()),
        "negative_bytes": int(-g[g < 0].sum()) if (g < 0).any() else 0,
    }
    _save(fig, out, "P5_gap_distribution", stats, "signed seek gaps")


def p6_access_matrix(ops, out, stats):
    m = ops.pivot_table(index="file", columns="rank", values="length",
                        aggfunc="sum", fill_value=0)
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * m.shape[1] + 4),
                                    max(3.5, .35 * len(m) + 2)))
    im = ax.imshow(m.to_numpy() / 1e9, aspect="auto", cmap="viridis")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([f"rank {c}" for c in m.columns], fontsize=8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([Path(f).name[:30] for f in m.index], fontsize=7)
    for i in range(len(m)):
        for j in range(m.shape[1]):
            v = m.iat[i, j] / 1e9
            ax.text(j, i, f"{v:.2f}" if v else "—", ha="center", va="center",
                    fontsize=6, color="w" if v < m.to_numpy().max()/1e9*.6 else "k")
    fig.colorbar(im, ax=ax, label="GB read")
    ax.set_title("P6  file x rank access matrix")
    _save(fig, out, "P6_access_matrix", stats, "which rank read which file")


def p7_sharing(per, label, by_bytes, nranks, out, stats):
    counts = per["class"].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["private", "partial", "full"]
    cols = {"private": "#2f855a", "partial": "#d69e2e", "full": "#c53030"}
    ax[0].bar(order, [counts.get(o, 0) for o in order],
              color=[cols[o] for o in order])
    ax[0].set_ylabel("files")
    ax[0].set_title("P7a  files by sharing class")
    ax[1].bar(order, [by_bytes.get(o, 0) / 1e9 for o in order],
              color=[cols[o] for o in order])
    ax[1].set_ylabel("GB read")
    ax[1].set_title("P7b  bytes by sharing class")
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.suptitle(f"workload label: {label}   "
                 f"(private=FPP, full=SSF, partial=PSF; {nranks} ranks, "
                 f"{len(per)} files) — one run gives one point", fontsize=9)
    stats["sharing"] = {"label": label, "nranks": nranks,
                        "files_by_class": counts.to_dict(),
                        "bytes_by_class": {k: int(v) for k, v in by_bytes.items()}}
    _save(fig, out, "P7_sharing_pattern", stats, "FPP / SSF / PSF")


def p8_throughput(ops, out, stats, aligned):
    span = ops["t1"].max()
    nb = max(20, min(300, int(span * 10) or 20))
    edges = np.linspace(0, span, nb + 1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    cmap = plt.get_cmap("tab10")
    for r, g in ops.groupby("rank"):
        h, _ = np.histogram(g["t0"], bins=edges)
        b, _ = np.histogram(g["t0"], bins=edges, weights=g["length"])
        w = np.diff(edges)
        ax[0].plot(edges[:-1], h / w, lw=1, color=cmap(r % 10), label=f"rank {r}")
        ax[1].plot(edges[:-1], b / w / 1e6, lw=1, color=cmap(r % 10))
    ax[0].set_ylabel("read IOPS")
    ax[0].set_title("P8  I/O rate over time" +
                    ("" if aligned else "  — NOT clock-aligned across ranks"))
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("MB/s")
    ax[1].set_xlabel("time (s)")
    for a in ax:
        a.grid(alpha=.3)
    stats["throughput"] = {
        "wall_s": float(span),
        "mean_MBps": float(ops["length"].sum() / 1e6 / span) if span else None,
        "clock_aligned": bool(aligned),
    }
    _save(fig, out, "P8_throughput", stats, "IOPS and MB/s over time")


def p9_gantt(ops, out, stats):
    g = ops.groupby("file").agg(first=("t0", "min"), last=("t1", "max"),
                                nbytes=("length", "sum")).reset_index()
    g = g.sort_values("first")
    fig, ax = plt.subplots(figsize=(10, max(3, .32 * len(g) + 1.6)))
    for i, r in enumerate(g.itertuples()):
        ax.barh(i, r.last - r.first, left=r.first, height=.6,
                color=KIND_COLOR.get(
                    ops.loc[ops["file"] == r.file, "kind"].iloc[0], "#718096"))
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([Path(f).name[:30] for f in g["file"]], fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title("P9  per-file activity window — overlap means concurrent, "
                 "staircase means serialised")
    ax.grid(alpha=.3, axis="x")
    # how concurrent is it really: max files active at once
    events = sorted([(r.first, 1) for r in g.itertuples()] +
                    [(r.last, -1) for r in g.itertuples()])
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    stats["concurrency"] = {"files": int(len(g)), "peak_files_active": int(peak)}
    _save(fig, out, "P9_file_gantt", stats, "per-file activity windows")


def p10_interarrival(ops, out, stats):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    allg = []
    cmap = plt.get_cmap("tab10")
    for r, g in ops.groupby("rank"):
        d = np.diff(np.sort(g["start_s"].to_numpy()))
        d = d[d > 0]
        if len(d) == 0:
            continue
        allg.append(d)
        s = np.sort(d)
        ax.plot(s * 1e6, np.arange(1, len(s) + 1) / len(s), lw=1.2,
                color=cmap(r % 10), label=f"rank {r} (n={len(s):,})")
    ax.set_xscale("log")
    ax.set_xlabel("inter-arrival time between consecutive reads (us, log)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("P10  read inter-arrival distribution — burstiness a tiering "
                 "policy has to absorb")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    if allg:
        a = np.concatenate(allg)
        stats["interarrival_us"] = {"p50": float(np.percentile(a, 50) * 1e6),
                                    "p95": float(np.percentile(a, 95) * 1e6),
                                    "max": float(a.max() * 1e6)}
    _save(fig, out, "P10_interarrival", stats, "inter-arrival CDF")


def p11_file_table(ops, files, out, stats):
    t = ops.groupby("file").agg(
        reads=("length", "size"), bytes_read=("length", "sum"),
        readers=("rank", "nunique"), mean_req=("length", "mean"),
        p95_req=("length", lambda s: np.percentile(s, 95)),
    ).reset_index()
    if files is not None:
        have = [c for c in ("file", "kind", "bytes_on_disk", "read_frac",
                            "amplif") if c in files.columns]
        t = t.merge(files[have], on="file", how="outer")
    # the per-file CSV may be absent entirely; give every optional column a
    # real Series so the row builder below never indexes into a scalar
    for c, d in (("kind", ""), ("bytes_on_disk", 0), ("read_frac", 0),
                 ("amplif", 0)):
        if c not in t.columns:
            t[c] = d
    t = t.fillna({"kind": "", "bytes_on_disk": 0, "read_frac": 0, "amplif": 0,
                  "reads": 0, "bytes_read": 0, "readers": 0, "mean_req": 0,
                  "p95_req": 0})
    t = t.sort_values("bytes_read", ascending=False)
    disp = pd.DataFrame({
        "file": [Path(str(f)).name[:28] for f in t["file"]],
        "kind": t["kind"].astype(str).to_numpy(),
        "size": [_fmt_bytes(v) for v in t["bytes_on_disk"]],
        "readers": t["readers"].astype(int),
        "reads": t["reads"].astype(int),
        "bytes read": [_fmt_bytes(v) for v in t["bytes_read"]],
        "read frac": [f"{v:.3f}" for v in t["read_frac"]],
        "amplif": [f"{v:.2f}x" if v else "—" for v in t["amplif"]],
        "mean req": [_fmt_bytes(v) for v in t["mean_req"]],
    })
    fig, ax = plt.subplots(figsize=(13, max(2.2, .32 * len(disp) + 1.4)))
    ax.axis("off")
    tb = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center",
                  cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.3)
    for j, c in enumerate(disp.columns):
        tb[0, j].set_facecolor("#2d3748")
        tb[0, j].set_text_props(color="w", weight="bold")
    for i in range(len(disp)):
        col = KIND_COLOR.get(str(disp["kind"].iloc[i]), "#718096")
        tb[i + 1, 1].set_facecolor(col)
        tb[i + 1, 1].set_text_props(color="w")
    ax.set_title("P11  per-file I/O profile — the placement-policy input",
                 fontsize=11, pad=14)
    stats["per_file"] = t.to_dict(orient="records")
    _save(fig, out, "P11_file_table", stats, "per-file summary")


def reuse_distances(ops, block=BLOCK):
    """LRU stack distance for every repeated block access, in bytes.

    For each access to a block that was seen before, the stack distance is the
    number of *distinct* blocks touched since its previous access. A cache of
    `distance * block` bytes under LRU would have turned that access into a
    hit. This is the number a placement or caching policy actually needs;
    "total size of all blocks read more than once" is only an upper bound,
    because it assumes every re-read block must be resident simultaneously.

    Computed with a Fenwick tree over access indices: O(n log n) rather than
    the O(n^2) pairwise scan, which does not finish on a full-scale trace.
    """
    seq = []
    for f, o, ln in zip(ops["file"], ops["offset"], ops["length"]):
        for b in range(o // block, (o + ln + block - 1) // block):
            seq.append((f, b))
    n = len(seq)
    if n == 0:
        return np.array([]), 0
    tree = [0] * (n + 1)

    def add(i, v):
        i += 1
        while i <= n:
            tree[i] += v
            i += i & -i

    def pref(i):          # sum over [0, i]
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    last: dict = {}
    dists = []
    for i, key in enumerate(seq):
        if key in last:
            j = last[key]
            dists.append(pref(i - 1) - pref(j))   # distinct blocks in (j, i)
            add(j, -1)
        add(i, 1)
        last[key] = i
    return np.array(dists, dtype=np.int64), n


def p12_reread_map(ops, out, stats):
    """How many times each BLOCK-sized region of each file was read, and how
    far apart the repeats are.

    Panel (a) is the volume question, (c) is the policy question: a cache of
    `x` bytes captures `y` of the duplicate traffic.
    """
    per_file_counts, hist = {}, Counter()
    for fname, g in ops.groupby("file"):
        hi = int(((g["offset"] + g["length"]).max() + BLOCK - 1) // BLOCK)
        cov = np.zeros(hi, dtype=np.int32)
        for o, ln in zip(g["offset"], g["length"]):
            cov[o // BLOCK: (o + ln + BLOCK - 1) // BLOCK] += 1
        per_file_counts[fname] = cov
        for v, c in zip(*np.unique(cov, return_counts=True)):
            hist[int(v)] += int(c)

    sel = sorted(per_file_counts, key=lambda f: -per_file_counts[f].size)[:MAX_PANELS]
    dists, n_access = reuse_distances(ops)

    fig = plt.figure(figsize=(13, 6.8 + .42 * len(sel)), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, max(1, .5 * len(sel)), 1])
    ax0 = fig.add_subplot(gs[0])
    ks = sorted(k for k in hist if k > 0)
    ax0.bar([str(k) for k in ks], [hist[k] * BLOCK / 1e9 for k in ks],
            color=["#2f855a" if k == 1 else "#c53030" for k in ks])
    ax0.set_xlabel("times the block was covered by a read")
    ax0.set_ylabel("GB of file")
    ax0.set_title("P12a  how much of the checkpoint is read once vs repeatedly")
    ax0.grid(alpha=.3, axis="y")

    ax1 = fig.add_subplot(gs[1])
    width = max((c.size for c in per_file_counts.values()), default=1)
    img = np.full((len(sel), width), np.nan)
    for i, f in enumerate(sel):
        c = per_file_counts[f]
        img[i, :c.size] = c
    im = ax1.imshow(img, aspect="auto", cmap="inferno", interpolation="nearest")
    ax1.set_yticks(range(len(sel)))
    ax1.set_yticklabels([Path(f).name[:30] for f in sel], fontsize=7)
    ax1.set_xlabel(f"file offset (in {BLOCK//1024} KB blocks)")
    ax1.set_title("P12b  where in each file the re-reads land")
    fig.colorbar(im, ax=ax1, label="read count")

    # (c) the policy question: cache size vs duplicate traffic captured
    ax2 = fig.add_subplot(gs[2])
    cache_pts: dict = {}
    if len(dists):
        s = np.sort(dists)
        cap = np.arange(1, len(s) + 1) / len(s)
        ax2.plot(np.maximum(s, 1) * BLOCK / 1e9, cap, color="#c53030", lw=1.6)
        for frac in (.5, .9, .99):
            i = min(int(frac * (len(s) - 1)), len(s) - 1)
            gb = max(s[i], 1) * BLOCK / 1e9
            cache_pts[f"capture_{int(frac*100)}pct_GB"] = float(gb)
            ax2.axvline(gb, ls="--", lw=1, color="#2f855a",
                        label=f"{frac:.0%} of re-reads at {gb:.2f} GB")
        ax2.set_xscale("log")
        ax2.legend(fontsize=8)
    else:
        ax2.text(.5, .5, "no repeated block accesses in this trace",
                 ha="center", va="center", fontsize=10)
    ax2.set_xlabel("LRU cache size (GB, log)")
    ax2.set_ylabel("fraction of re-reads\nturned into hits")
    ax2.set_title("P12c  reuse-distance curve — how big a cache actually has "
                  "to be to absorb the duplicate traffic")
    ax2.grid(alpha=.3)

    dup_blocks = sum(c for k, c in hist.items() if k > 1)
    stats["reread"] = {
        "block_bytes": BLOCK,
        "blocks_read_once": int(hist.get(1, 0)),
        "blocks_read_multiple": int(dup_blocks),
        "bytes_in_multiread_blocks_UPPER_BOUND": int(dup_blocks * BLOCK),
        "max_read_count": int(max(hist) if hist else 0),
        "block_accesses": int(n_access),
        "repeat_accesses": int(len(dists)),
        "reuse_distance_blocks": {
            "p50": int(np.percentile(dists, 50)) if len(dists) else None,
            "p90": int(np.percentile(dists, 90)) if len(dists) else None,
            "max": int(dists.max()) if len(dists) else None,
        },
        **cache_pts,
    }
    _save(fig, out, "P12_reread_map", stats, "re-read localisation", tight=False)


def p13_targets(ops, out, stats, meta):
    if "targets" not in ops.columns or ops["targets"].isna().all():
        print("  P13 skipped: no target columns "
              "(re-run parse_darshan.py with --chunksize/--numtargets)")
        return
    recs = []
    for t0, tl, bl in zip(ops["t0"], ops["targets"], ops["target_bytes"]):
        if not isinstance(tl, str) or not isinstance(bl, str):
            continue
        for t, b in zip(tl.split(","), bl.split(",")):
            recs.append((t0, t, int(b)))
    if not recs:
        print("  P13 skipped: target columns present but empty")
        return
    d = pd.DataFrame(recs, columns=["t", "target", "bytes"])
    span = d["t"].max() or 1
    edges = np.linspace(0, span, max(20, min(200, int(span * 5) or 20)) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    cmap = plt.get_cmap("tab20")
    tot = d.groupby("target")["bytes"].sum().sort_values(ascending=False)
    for i, tg in enumerate(tot.index):
        g = d[d["target"] == tg]
        h, _ = np.histogram(g["t"], bins=edges, weights=g["bytes"])
        ax[0].plot(edges[:-1], h / np.diff(edges) / 1e6, lw=1,
                   color=cmap(i % 20), label=f"{tg}")
    ax[0].set_ylabel("MB/s")
    ax[0].set_xlabel("time (s)")
    ax[0].legend(fontsize=7, ncol=4)
    ax[0].grid(alpha=.3)
    lbl = "target" if meta.get("target_map") else "stripe slot"
    ax[0].set_title(f"P13a  bytes per {lbl} over time")
    ax[1].bar([str(x) for x in tot.index], tot.to_numpy() / 1e9,
              color=[cmap(i % 20) for i in range(len(tot))])
    ax[1].set_ylabel("GB")
    ax[1].set_xlabel(lbl)
    ax[1].grid(alpha=.3, axis="y")
    imb = tot.max() / tot.min() if tot.min() else float("inf")
    ax[1].set_title(f"P13b  total per {lbl} — max/min imbalance {imb:.2f}x")
    if not meta.get("target_map"):
        fig.text(.5, .005, "slots are positions within each file's own target "
                 "list, not disks; pass --target-map to resolve",
                 ha="center", fontsize=8, color="#c53030")
    stats["targets"] = {"label": lbl, "resolved": bool(meta.get("target_map")),
                        "bytes": {str(k): int(v) for k, v in tot.items()},
                        "imbalance": float(imb)}
    _save(fig, out, "P13_targets", stats, "per-target load")


def p14_latency(ops, out, stats):
    d = ops["dur"].to_numpy()
    d = d[d > 0]
    if len(d) == 0:
        print("  P14 skipped: all durations zero (timer resolution?)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.logspace(math.log10(d.min()), math.log10(d.max() * 1.05), 60)
    ax[0].hist(d * 1e6, bins=bins * 1e6, color="#2b6cb0")
    ax[0].axvline(10, color="#c53030", ls="--", lw=1,
                  label="10 us — below this is page cache, not media")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("service time (us, log)")
    ax[0].set_ylabel("operations")
    ax[0].set_title("P14a  per-operation service time")
    ax[0].legend(fontsize=8)
    s = np.sort(d)
    ax[1].plot(s * 1e6, np.arange(1, len(s) + 1) / len(s), color="#2b6cb0")
    qd = {}
    for q, c in ((50, "#2f855a"), (95, "#d69e2e"), (99, "#c53030")):
        v = float(np.percentile(s, q))
        qd[f"p{q}_us"] = v * 1e6
        ax[1].axvline(v * 1e6, color=c, ls="--", lw=1, label=f"p{q} = {v*1e6:,.0f} us")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("service time (us, log)")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].set_title("P14b  CDF")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    warm = float((d < 10e-6).mean())
    stats["latency"] = {**qd, "ops": int(len(d)),
                        "frac_under_10us": warm,
                        "total_service_s": float(d.sum())}
    if warm > .25:
        print(f"  ! {warm:.0%} of reads completed in under 10 us — the page "
              "cache was warm; drop_caches did not take")
    _save(fig, out, "P14_latency", stats, "service-time distribution")


def p15_latency_vs_size(ops, out, stats):
    g = ops[ops["dur"] > 0]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for r, gr in g.groupby("rank"):
        ax.scatter(gr["length"], gr["dur"] * 1e6, s=4, alpha=.35,
                   color=cmap(r % 10), label=f"rank {r}")
    # bandwidth reference lines
    x = np.array([g["length"].min(), g["length"].max()], dtype=float)
    for bw, c, nm in ((200e6, "#c53030", "200 MB/s (HDD-ish)"),
                      (2e9, "#2f855a", "2 GB/s (NVMe-ish)")):
        ax.plot(x, x / bw * 1e6, ls="--", lw=1, color=c, label=nm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("request size (bytes, log)")
    ax.set_ylabel("service time (us, log)")
    ax.set_title("P15  latency vs request size — slope gives effective "
                 "bandwidth, floor gives fixed per-op cost")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=.3)
    big = g[g["length"] >= 1e6]
    if len(big):
        stats["effective_bw_MBps_large_ops"] = float(
            (big["length"].sum() / big["dur"].sum()) / 1e6)
    _save(fig, out, "P15_latency_vs_size", stats, "latency vs size")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def write_report(stats, out: Path, tag: str, meta: dict, aligned: bool):
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    L = [f"# I/O access-pattern report — {tag}", ""]
    if meta.get("warnings"):
        L += ["## Warnings carried over from the parser", ""]
        L += [f"* {w}" for w in meta["warnings"]] + [""]

    sz, sw = stats.get("size", {}), stats.get("size_weighting", {})
    L += ["## Request size", "",
          f"* {sz.get('ops', 0):,} read operations, "
          f"{_fmt_bytes(sz.get('mean', 0))} mean",
          f"* p50 {_fmt_bytes(sz.get('p50', 0))}, p95 {_fmt_bytes(sz.get('p95', 0))}, "
          f"p99 {_fmt_bytes(sz.get('p99', 0))}, max {_fmt_bytes(sz.get('max', 0))}"]
    if sw:
        L += [f"* operations at or below {_fmt_bytes(sw['threshold_bytes'])}: "
              f"**{sw['frac_of_ops']:.1%} of operations** but "
              f"**{sw['frac_of_bytes']:.1%} of bytes** — the count/byte split "
              f"that decides whether seek cost or bandwidth dominates"]
    sp = stats.get("spatial", {})
    if sp:
        tot = sum(v for k, v in sp.items() if k in CLS_COLOR) or 1
        L += ["", "## Spatial pattern", ""]
        L += [f"* {k}: {sp[k]:,} ({sp[k]/tot:.1%})" for k in CLS_COLOR]
    gp = stats.get("gaps", {})
    if gp:
        L += [f"* {gp['negative']:,} backward transitions covering "
              f"{_fmt_bytes(gp['negative_bytes'])} of revisited range"]
    sh = stats.get("sharing", {})
    if sh:
        L += ["", "## Sharing", "",
              f"* workload label **{sh['label']}** over {sh['nranks']} ranks",
              f"* files by class: {sh['files_by_class']}"]
    rr = stats.get("reread", {})
    if rr:
        L += ["", "## Re-reads", "",
              f"* {_fmt_bytes(rr['blocks_read_once'] * rr['block_bytes'])} read "
              f"exactly once; {_fmt_bytes(rr['bytes_in_multiread_blocks_UPPER_BOUND'])} "
              f"in blocks read more than once (max {rr['max_read_count']}x) — "
              f"an *upper bound* on cache need, since it assumes every such "
              f"block must be resident at once",
              f"* {rr['repeat_accesses']:,} of {rr['block_accesses']:,} block "
              f"accesses were repeats; reuse distance p50 "
              f"{rr['reuse_distance_blocks']['p50']} blocks, p90 "
              f"{rr['reuse_distance_blocks']['p90']} blocks"]
        for k in ("capture_50pct_GB", "capture_90pct_GB", "capture_99pct_GB"):
            if k in rr:
                L += [f"* an LRU cache of **{rr[k]:.2f} GB** turns "
                      f"{k.split('_')[1].replace('pct','%')} of the duplicate "
                      f"reads into hits"]
    lat = stats.get("latency", {})
    if lat:
        L += ["", "## Latency", "",
              f"* p50 {lat['p50_us']:,.0f} us, p95 {lat['p95_us']:,.0f} us, "
              f"p99 {lat['p99_us']:,.0f} us",
              f"* {lat['frac_under_10us']:.1%} of reads under 10 us "
              f"({'WARM CACHE — rerun with cold caches' if lat['frac_under_10us'] > .25 else 'cache looks cold'})"]
    if "effective_bw_MBps_large_ops" in stats:
        L += [f"* effective bandwidth on >=1 MB operations: "
              f"{stats['effective_bw_MBps_large_ops']:,.0f} MB/s"]
    tg = stats.get("targets", {})
    if tg:
        L += ["", "## Striping", "",
              f"* {tg['label']}s used: {len(tg['bytes'])}, "
              f"max/min imbalance {tg['imbalance']:.2f}x",
              f"* {'resolved to real target IDs' if tg['resolved'] else '**slots, not disks** — pass --target-map'}"]
    L += ["", "## Caveats", "",
          f"* cross-rank clock alignment: "
          f"{'yes (start_abs)' if aligned else '**NO** — temporal figures are per-process clocks'}",
          "* latency is blocking-syscall duration; it does not decompose into "
          "client cache / network / server queue / media",
          "* reads served from Python's BufferedReader never enter the kernel "
          "and are invisible here, correctly so"]
    (out / "stats.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out/'stats.json'}\n      {out/'stats.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", type=Path,
                    help="results prefix; .ops.csv/.files.csv/.darshan.json "
                         "are derived from it")
    ap.add_argument("--ops", type=Path)
    ap.add_argument("--files", type=Path)
    ap.add_argument("--json", dest="jsonf", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated figure ids to draw, e.g. P3,P12")
    args = ap.parse_args()

    ops, files, meta, aligned = load(args.prefix, args.ops, args.files, args.jsonf)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.prefix.name if args.prefix else "run")
    nranks = ops["rank"].nunique()

    print(f"=== plotting {tag} ===")
    print(f"  {len(ops):,} read ops, {ops['file'].nunique()} files, "
          f"{nranks} ranks, {_fmt_bytes(ops['length'].sum())} read")
    if not aligned:
        print("  ! no start_abs column — cross-rank temporal figures use "
              "per-process clocks and may be misaligned")
    if meta.get("clock_skew_s"):
        print(f"  process start skew {meta['clock_skew_s']:.3f} s")

    stats: dict = {"tag": tag, "ranks": int(nranks),
                   "files": int(ops["file"].nunique()),
                   "read_ops": int(len(ops)),
                   "bytes_read": int(ops["length"].sum())}
    want = set(args.only.split(",")) if args.only else None

    def run(pid, fn, *a):
        if want and pid not in want:
            return
        try:
            fn(*a)
        except Exception as e:  # one bad figure must not lose the rest
            print(f"  ! {pid} failed: {type(e).__name__}: {e}")

    streams, gaps = classify_streams(ops)
    per_share, label, by_bytes = classify_sharing(ops, nranks)

    run("P1", p1_size_hist, ops, out, stats)
    run("P2", p2_count_vs_bytes, ops, out, stats)
    run("P3", p3_offset_time, ops, out, stats)
    run("P4", p4_stream_class, streams, out, stats)
    run("P5", p5_gap_hist, gaps, out, stats)
    run("P6", p6_access_matrix, ops, out, stats)
    run("P7", p7_sharing, per_share, label, by_bytes, nranks, out, stats)
    run("P8", p8_throughput, ops, out, stats, aligned)
    run("P9", p9_gantt, ops, out, stats)
    run("P10", p10_interarrival, ops, out, stats)
    run("P11", p11_file_table, ops, files, out, stats)
    run("P12", p12_reread_map, ops, out, stats)
    run("P13", p13_targets, ops, out, stats, meta)
    run("P14", p14_latency, ops, out, stats)
    run("P15", p15_latency_vs_size, ops, out, stats)

    streams.to_csv(out / "streams.csv", index=False)
    write_report(stats, out, tag, meta, aligned)


if __name__ == "__main__":
    main()
