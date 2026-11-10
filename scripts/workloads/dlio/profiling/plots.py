"""Approved A/B/C views and exact-event HTML detail, without a web service."""

from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analyze_profiles import dump, table, union

COLORS = {"read":"#2878ad", "write":"#d77b29"}
BANDS = ("<4 KiB (includes EOF)", "4–64 KiB", "64 KiB–1 MiB", "≥1 MiB")
SIZE_SQL = "CASE WHEN length<4096 THEN 0 WHEN length<65536 THEN 1 WHEN length<1048576 THEN 2 ELSE 3 END"


def bins(db, seconds):
    return list(db.execute(f"SELECT cast(max(0,end)/? AS INTEGER),op,{SIZE_SQL},count(*),sum(length) FROM events GROUP BY 1,2,3", (seconds,)))


def intersections(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a,b = max(left[i][0],right[j][0]),min(left[i][1],right[j][1])
        if b>a:
            result.append((a,b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def telemetry_phases(case, coverage):
    """Only wholly contained counter intervals: never invent boundary bytes."""
    folder = case/"telemetry"
    if not (folder/"clocks.json").exists():
        return []
    clocks = json.loads((folder/"clocks.json").read_text())
    rows = []
    for host, calibration in clocks.items():
        first, last = calibration["before"], calibration.get("after")
        if last is None:
            continue
        drift = abs(first["offset_s"]-last["offset_s"])
        error = max(first["uncertainty_s"],last["uncertainty_s"])+drift
        offset = (first["offset_s"]+last["offset_s"])/2
        with (folder/f"{host}.jsonl").open() as file:
            samples = [json.loads(line) for line in file]
        for phase in coverage["phases"]:
            totals = defaultdict(lambda: {"read_bytes":0,"write_bytes":0,"covered_s":0})
            if samples:
                for category in ("devices","interfaces"):
                    for name in samples[0][category]:
                        totals[(category,name)]
            for before,after in zip(samples,samples[1:]):
                a = before["wall_s"]-offset-coverage["clock_origin_wall_s"]
                b = after["wall_s"]-offset-coverage["clock_origin_wall_s"]
                elapsed = after["monotonic_s"]-before["monotonic_s"]
                if elapsed <= 0 or abs((b-a)-elapsed) > .1 or a-error < phase["start_s"] or b+error > phase["end_s"]:
                    continue
                for category,read,write in (("devices","read_bytes","write_bytes"),("interfaces","rx","tx")):
                    for name,old in before[category].items():
                        if name not in after[category]:
                            continue
                        new = after[category][name]
                        r,w = new[read]-old[read],new[write]-old[write]
                        if r<0 or w<0:
                            continue
                        item=totals[(category,name)]
                        item["read_bytes"]+=r; item["write_bytes"]+=w; item["covered_s"]+=elapsed
            for (category,name),values in totals.items():
                rows.append({"phase":phase["phase"],"host":host,"category":category,"name":name,
                    "read_or_rx_bytes":values["read_bytes"] if values["covered_s"] else "",
                    "write_or_tx_bytes":values["write_bytes"] if values["covered_s"] else "",
                    "observed_interval_s":values["covered_s"],
                    "phase_coverage_fraction":values["covered_s"]/(phase["end_s"]-phase["start_s"]),
                    "clock_uncertainty_s":error,"scope":"host/device-wide; excludes boundary-straddling intervals"})
    table(case/"analysis/telemetry-phases.csv",rows)
    return rows


def render(cases, seconds=5):
    # One area scale, bin width and request-size definition for the whole study.
    largest = 1
    for case in cases:
        with sqlite3.connect(case/"analysis/events.sqlite") as db:
            largest = max(largest, max((row[3] for row in bins(db,seconds)),default=1))
    for case in cases:
        manifest=json.loads((case/"case.json").read_text())
        coverage=json.loads((case/"analysis/coverage.json").read_text())
        if not coverage["valid"]:
            raise ValueError(f"Refusing publication plots for incomplete trace: {case}")
        phases=coverage["phases"]
        with (case/"analysis/phases.csv").open() as file:
            summaries={row["phase"]:row for row in csv.DictReader(file)}
        with sqlite3.connect(case/"analysis/events.sqlite") as db:
            activity={op:union(db.execute("SELECT max(0,start),end FROM events WHERE op=? AND end>=0 ORDER BY start",(op,))) for op in COLORS}
            fig,axes=plt.subplots(3,1,figsize=(17,12),sharex=True,gridspec_kw={"height_ratios":[2.3,2.2,1.6]})
            fig.subplots_adjust(top=.85,bottom=.17,left=.17,right=.98,hspace=.45)
            label=f"{manifest['workload']} · {manifest['suite']} pool {manifest['pool']}"
            fig.suptitle(label+" · measured POSIX/DXT phase/access trace",fontsize=17,fontweight="bold",y=.97)
            fig.text(.5,.93,f"One profiled invocation · {coverage['processes']} processes · "
                     f"end-to-end {coverage['end_to_end_samples_s']:.3f} samples/s (native full-run timestamps)",ha="center",fontsize=10)
            fig.legend(handles=[Patch(color=value,label=key.title()) for key,value in COLORS.items()],
                       loc="upper center",bbox_to_anchor=(.58,.91),ncol=2,frameon=False)
            duration=coverage["chart_duration_s"]
            for ax in axes:
                ax.set_xlim(0,duration)
                ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
                ax.spines[["top","right"]].set_visible(False)
                for phase in phases:
                    if phase["phase"].startswith("E"):
                        ax.axvline(phase["start_s"],color="#aebbc8",lw=.7,ls="--")
            ax=axes[0]
            ax.set_title("A. Complete epochs, nested rank-0 checkpoint timers, all-rank I/O activity",loc="left",fontweight="bold")
            for key in ("dlio_start_s","dlio_end_s"):
                if key in coverage:
                    ax.axvline(coverage[key],color="#4b5563",ls=":",lw=1)
            for phase in phases:
                a,b=phase["start_s"],phase["end_s"]
                ax.broken_barh([(a,b-a)],(3.8,.6),facecolors="#dfedf8" if phase["phase"].startswith("E") else "#e3e6eb")
                if b-a > duration*.025:
                    ax.text((a+b)/2,4.1,phase["phase"],ha="center",va="center",fontsize=9)
                if phase["phase"].startswith("E"):
                    row=summaries[phase["phase"]]
                    quantiles="/".join(f"{float(row['read_'+q+'_bytes'])/1024:.0f}" if row['read_'+q+'_bytes'] else "n/a" for q in ("p50","p95"))
                    ax.text((a+b)/2,1.95,
                        f"R {float(row['read_bytes'])/2**30:.2f} GiB\n"
                        f"p50/p95 {quantiles} KiB",
                        ha="center",va="top",fontsize=8,color="#21465f")
                    if float(row["write_bytes"]):
                        center = next(((event["start_s"]+event["end_s"])/2
                                       for event in coverage["checkpoints"] if f"E{event['epoch']}" == phase["phase"]), (a+b)/2)
                        sizes = "/".join(f"{float(row['write_'+q+'_bytes'])/1024:.0f}"
                                         if row['write_'+q+'_bytes'] else "n/a" for q in ("p50","p95"))
                        ax.text(center,.98,f"W {float(row['write_bytes'])/2**20:.1f} MiB\np50/p95 {sizes} KiB",
                                ha="center",va="top",fontsize=8,color="#88501f")
            for event in coverage["checkpoints"]:
                a,b=event["start_s"],event["end_s"]
                ax.broken_barh([(a,b-a)],(3.05,.4),facecolors="#ffe5c9",edgecolors=COLORS["write"])
                ax.text((a+b)/2,3.5,event["phase"],ha="center",fontsize=8)
            for op,y in (("read",2.2),("write",1.1)):
                ax.broken_barh([(a,b-a) for a,b in activity[op]],(y,.4),facecolors=COLORS[op],rasterized=True)
            overlap=intersections(activity["read"],activity["write"])
            ax.broken_barh([(a,b-a) for a,b in overlap],(.35,.18),facecolors="#9163b6",rasterized=True)
            ax.set_ylim(0,4.8)
            ax.set_yticks([4.1,3.25,2.4,1.3,.44],["Complete epochs","Checkpoint timer\n(rank 0)","Read-call activity","Write-call activity","Temporal overlap"])
            ax=axes[1]
            ax.set_title("B. Rank-owned call activity (includes reader workers; overlapping same-operation calls merged)",loc="left",fontweight="bold")
            for rank in range(8):
                for op,offset in (("read",-.3),("write",.02)):
                    spans=union(db.execute("SELECT max(0,start),end FROM events WHERE rank=? AND op=? AND end>=0 ORDER BY start",(rank,op)))
                    ax.broken_barh([(a,b-a) for a,b in spans],(rank+offset,.28),facecolors=COLORS[op],rasterized=True)
            ax.set_yticks(range(8),[f"Rank {rank}" for rank in range(8)])
            ax.set_ylim(7.6,-.6)
            ax=axes[2]
            ax.set_title(f"C. Request sizes · {seconds:g}s completion bins · area proportional to count (2 pt² visibility floor)",loc="left",fontweight="bold")
            histogram=bins(db,seconds)
            rows=[]
            for index,op,band,count,byte_count in histogram:
                right=min((index+1)*seconds,duration)
                center=(index*seconds+right)/2
                spread=min(seconds*.12,(right-index*seconds)*.15)
                ax.scatter([center+(spread if op=="write" else -spread)],
                           [band+(.1 if op=="write" else -.1)],s=max(2,240*count/largest),color=COLORS[op],alpha=.8)
                rows.append({"bin_start_s":index*seconds,"bin_end_s":right,"op":op,
                             "size_band":BANDS[band],"calls":count,"bytes":byte_count})
            for value in sorted({1,max(1,largest//10),largest}):
                ax.scatter([],[],s=max(2,240*value/largest),color="#74808b",label=f"{value} calls")
            ax.legend(frameon=False,fontsize=8,loc="upper right")
            ax.set_yticks(range(4),BANDS); ax.set_ylim(-.6,3.6)
            ax.set_xlabel("Elapsed seconds from process-start clock anchors (DLIO's measured interval is recorded separately)",labelpad=12)
            interval = (f"Native DLIO timed interval: {coverage['dlio_start_s']:.3f}–{coverage['dlio_end_s']:.3f}s. "
                        if "dlio_start_s" in coverage else "")
            fig.text(.5,.035,interval+"API-call activity is not physical-device service time; white gaps are unattributed. "
                     "Checkpoint timers are nested within epochs, not added twice.\n"
                     f"Read/write overlap union: {sum(b-a for a,b in overlap):.3f}s. Exact events, crossed boundaries, phase statistics and telemetry coverage are retained in analysis tables.",
                     ha="center",fontsize=8.5,linespacing=1.6)
            output=case/"analysis"
            for suffix in ("png","svg"):
                fig.savefig(output/f"phase_access.{suffix}",dpi=150,facecolor="white")
            plt.close(fig)
            table(output/"size-bins.csv",rows)
            telemetry_phases(case,coverage)
            dump(output/"rendering.json",{"bin_seconds":seconds,"shared_max_call_count":largest,
                "glyph_max_area_pt2":240,"glyph_min_area_pt2":2,"size_quantiles":"count weighted",
                "B":"union per rank/operation; raw requests retained in events.sqlite",
                "C":"completion-time count bins; not physical bandwidth"})
            text=("<!doctype html><meta charset='utf-8'><title>DLIO phase trace</title>"
                  "<style>body{font:16px sans-serif;margin:2em}img{width:100%}</style>"
                  f"<h1>{html.escape(label)}</h1><p>Validated POSIX/DXT coverage; one instrumented invocation.</p>"
                  "<img src='phase_access.svg'><p>For exact-event hover and file/rank/time selection, run "
                  "analyze_profiles.py CASE --detail --start SECONDS --end SECONDS [--rank N] [--file PATH]. "
                  "Full events are preserved in events.sqlite; native phases and metrics are in the adjacent CSV/JSON files.</p>")
            (output/"overview.html").write_text(text)


def comparison(cases, output):
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for case in cases:
        manifest=json.loads((case/"case.json").read_text())
        coverage=json.loads((case/"analysis/coverage.json").read_text())
        groups[manifest["workload"]].append((manifest,coverage,case))
    for workload,items in groups.items():
        items.sort(key=lambda x:(x[0]["suite"]!="normal",x[0]["pool"]))
        fig,ax=plt.subplots(figsize=(14,max(4,len(items)*.7+2)))
        maximum=max(item[1]["chart_duration_s"] for item in items)
        for row,(manifest,coverage,case) in enumerate(items):
            for phase in coverage["phases"]:
                ax.broken_barh([(phase["start_s"],phase["end_s"]-phase["start_s"])],(row-.35,.7),facecolors="#e5edf5")
                if phase["end_s"]-phase["start_s"] > maximum*.04:
                    ax.text((phase["start_s"]+phase["end_s"])/2,row-.17,phase["phase"],ha="center",fontsize=8)
            with sqlite3.connect(case/"analysis/events.sqlite") as db:
                for op,y in (("read",row+.02),("write",row+.18)):
                    spans=union(db.execute("SELECT max(0,start),end FROM events WHERE op=? AND end>=0 ORDER BY start",(op,)))
                    ax.broken_barh([(a,b-a) for a,b in spans],(y,.13),facecolors=COLORS[op],rasterized=True)
        ax.set_yticks(range(len(items)),[f"{m['suite']} pool {m['pool']}" for m,_,_ in items])
        ax.set_ylim(len(items)-.5,-.5); ax.set_xlim(0,maximum*1.03)
        ax.set_xlabel("Elapsed seconds from process start · common absolute scale")
        ax.set_title(f"{workload}: measured phases and call activity · one profiled invocation/configuration",fontweight="bold")
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=.2)
        fig.tight_layout()
        fig.savefig(output/f"{workload}-configuration-phases.png",dpi=160)
        fig.savefig(output/f"{workload}-configuration-phases.svg")
        plt.close(fig)


def detail(case, rank=None, filename=None, start=0, end=None):
    """Exact, bounded SQLite query -> standalone SVG/HTML with request hover."""
    coverage=json.loads((case/"analysis/coverage.json").read_text())
    end=coverage["chart_duration_s"] if end is None else end
    if end<=start:
        raise ValueError("--end must exceed --start")
    query,params="SELECT pid,rank,file,op,offset,length,start,end,thread FROM events WHERE end>=? AND start<=?",[start,end]
    if rank is not None:
        query+=" AND rank=?"; params.append(rank)
    if filename:
        query+=" AND file=?"; params.append(filename)
    with sqlite3.connect(case/"analysis/events.sqlite") as db:
        events=list(db.execute(query+" ORDER BY rank,pid,start LIMIT 50001",params))
    if len(events)>50000:
        raise ValueError("Detail exceeds 50,000 events; narrow --start/--end, --rank or --file (no silent sampling)")
    identities=sorted({(row[1],row[0]) for row in events})
    lanes={identity:index for index,identity in enumerate(identities)}
    width,height=1600,max(220,70+len(lanes)*35)
    parts=[f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' style='width:100%'>"]
    for (owner,pid),lane in lanes.items():
        parts.append(f"<text x='5' y='{50+35*lane}'>Rank {owner}, PID {pid}</text>")
    for pid,owner,name,op,offset,length,a,b,thread in events:
        x=190+1380*(max(a,start)-start)/(end-start)
        w=max(.4,1380*(min(b,end)-max(a,start))/(end-start))
        y=35+35*lanes[(owner,pid)]+(0 if op=="read" else 12)
        title=html.escape(f"{op} | rank {owner} PID {pid} thread {thread} | {name} | offset {offset} length {length} bytes | {a:.6f}–{b:.6f}s; duration {b-a:.6f}s")
        parts.append(f"<rect x='{x:.4f}' y='{y}' width='{w:.4f}' height='11' fill='{COLORS[op]}'><title>{title}</title></rect>")
    for index in range(11):
        x=190+1380*index/10
        parts.append(f"<text x='{x}' y='{height-12}' text-anchor='middle'>{start+(end-start)*index/10:.3f}s</text>")
    parts.append("</svg>")
    text=("<!doctype html><meta charset='utf-8'><title>Exact DXT calls</title>"
          "<style>body{font:14px sans-serif}#view{overflow:auto}svg{min-width:100%}rect:hover{stroke:black}</style>"
          f"<h1>Exact DXT call detail · {len(events)} events</h1><p>Hover for file, rank, process, thread, size, offset and duration. "
          "Subpixel calls have a 0.4-pixel visibility floor; tooltip durations are exact. Blue reads, orange writes.</p>"
          "<label>Zoom <input type='range' min='100' max='800' value='100' oninput=\"document.querySelector('svg').style.width=this.value+'%'\"></label>"
          "<div id='view'>"+"".join(parts)+"</div>")
    (case/"analysis/detail.html").write_text(text)
