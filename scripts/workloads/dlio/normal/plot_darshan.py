#!/usr/bin/env python3
"""Plot one saved DLIO invocation (including a two-epoch pilot), offline."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import dlio_darshan


def parser_lines(tool, path):
    with tempfile.TemporaryFile(mode="w+") as output:
        subprocess.run([tool,str(path)],stdout=output,check=True)
        output.seek(0)
        yield from output


def collect(case, native_utc_offset=None):
    config=yaml.safe_load((case/".hydra/config.yaml").read_text())["workload"]
    ranks=int(json.loads((case/"summary.json").read_text())["num_accelerators"])
    native=json.loads((case/"per_epoch_stats.json").read_text())
    origin=datetime.fromisoformat(native["1"]["start"])
    if native_utc_offset is None:
        native_utc_offset=datetime.now().astimezone().utcoffset().total_seconds()
    absolute_origin=(origin if origin.tzinfo else origin.replace(
        tzinfo=timezone(timedelta(seconds=native_utc_offset)))).timestamp()
    dlio_darshan.check_logs(case,ranks=ranks)
    identities=[json.loads(path.read_text()) for pattern in ("rank-*.json","worker-*.json")
                for path in (case/"darshan").glob(pattern)]
    owners={item["pid"]:item["rank"] for item in identities}
    paths=[str(Path(config[section][field])).rstrip("/")+"/" for section,field in
           (("dataset","data_folder"),("checkpoint","checkpoint_folder"))]
    measured=lambda name:any(name.startswith(path) for path in paths)
    totals=[Counter() for _ in range(ranks)]
    sizes={op:Counter() for op in ("read","write")}
    events=[]
    clocks={}
    for log in sorted((case/"darshan").rglob("*.darshan")):
        match=re.search(r"_id\d+-(\d+)_",log.name)
        if not match:
            raise ValueError(f"Unknown native log filename: {log}")
        rank=owners.get(int(match[1]))
        job_start=None
        for line in parser_lines("darshan-parser",log):
            if line.startswith("# start_time:"):
                job_start=float(line.split(":",1)[1])
            fields=line.split()
            if len(fields)<6 or fields[0] not in ("POSIX","STDIO") or not measured(fields[5]):
                continue
            if fields[3] not in ("POSIX_BYTES_READ","POSIX_BYTES_WRITTEN","STDIO_BYTES_READ","STDIO_BYTES_WRITTEN"):
                continue
            value=int(fields[4])
            if value<0:
                raise ValueError(f"Unavailable byte counter: {log}: {line.strip()}")
            if rank is None and value:
                raise ValueError(f"Measured I/O in unmapped process log: {log}")
            if rank is not None:
                totals[rank][fields[3]]+=value
        if rank is not None:
            if job_start is None:
                raise ValueError(f"Missing job start timestamp: {log}")
            clocks[match[1]]={"job_start_s":job_start,"rank":rank}
        name=""
        for line in parser_lines("darshan-dxt-parser",log):
            if line.startswith("# DXT, file_id:"):
                name=line.split("file_name:",1)[1].strip()
            fields=line.split()
            if len(fields)<8 or fields[0]!="X_POSIX" or fields[2] not in sizes or not measured(name):
                continue
            if rank is None:
                raise ValueError(f"DXT workload events in unmapped process log: {log}")
            length=int(fields[5])
            if length<0:
                raise ValueError("Negative DXT request length")
            sizes[fields[2]][length]+=1
            start,end=map(float,fields[6:8])
            if end<start:
                raise ValueError(f"Negative DXT request duration: {log}")
            events.append({"rank":rank,"pid":int(match[1]),"file":name,"op":fields[2],
                           "offset":int(fields[4]),"length":length,
                           "dxt_start_s":start,"dxt_end_s":end,
                           "start_s":job_start-absolute_origin+start,
                           "end_s":job_start-absolute_origin+end,
                           "thread":fields[8] if len(fields)>8 else None})
    relative=lambda value:(datetime.fromisoformat(value)-origin).total_seconds()
    phases=[]
    for epoch,record in sorted(native.items(),key=lambda item:int(item[0])):
        start,end=relative(record["start"]),relative(record["end"])
        if end<=start:
            raise ValueError(f"Invalid epoch interval: {epoch}")
        checkpoints=[]
        for name,event in record.items():
            if not name.startswith("ckpt"):
                continue
            left,right=relative(event["start"]),relative(event["end"])
            if not start<=left<right<=end:
                raise ValueError(f"Checkpoint is outside epoch {epoch}")
            checkpoints.append([left,right])
        phases.append({"epoch":int(epoch),"start_s":start,"end_s":end,"rank0_checkpoints":checkpoints})
    pool_match=re.search(r"pool(\d+)$",case.name)
    pool=int(pool_match[1]) if pool_match else None
    repetition_match=re.match(r"rep(\d+)-pool\d+$",case.name)
    repetition=int(repetition_match[1]) if repetition_match else None
    pool_label={1:"Default",3:"HDD",6:"SSD",14:"4 fast / 0 slow",
                15:"3 fast / 1 slow",16:"2 fast / 2 slow",
                17:"1 fast / 3 slow",18:"0 fast / 4 slow"}.get(pool,"Pool")
    return {"model":config["model"],"pool":pool,"repetition":repetition,
            "pool_label":f"{pool_label} (pool {pool})" if pool is not None else "Single invocation",
            "source":str(case.resolve()),"phases":phases,
            "rank_api_bytes":[dict(row) for row in totals],
            "dxt_request_sizes":{op:dict(sorted(counts.items())) for op,counts in sizes.items()},
            "events":events,"clocks":clocks,"native_utc_offset_s":native_utc_offset,
            "limitations":["STDIO and POSIX are separate API views; do not sum their bytes.",
                            "Epoch bands include checkpoint time; checkpoint markers are rank-0 timers.",
                            "Request sizes are DXT POSIX API requests, not file or stripe sizes.",
                            "COARSE alignment only: whole-second job starts; initialization offset is uncalibrated.",
                            "No precise subsecond overlap or phase-boundary attribution is supported."]}


COLORS={"read":"#2878B5","write":"#D08734"}
SIZE_LABELS=["0 B (EOF)","1–4 KiB",">4–64 KiB",">64 KiB–1 MiB",">1 MiB"]
CLOCK_NOTE="COARSE clock alignment: whole-second job starts; initialization offset uncalibrated"


def size_category(length):
    return next(i for i,limit in enumerate((0,4096,65536,1048576,float("inf"))) if length<=limit)


def format_bytes(value):
    for unit in ("B","KiB","MiB","GiB"):
        if value<1024 or unit=="GiB":
            return f"{value:g} B" if unit=="B" else f"{value:.2f} {unit}"
        value/=1024


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:
            result[-1][1]=max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def phase_bands(ax, data):
    for phase in data["phases"]:
        ax.axvspan(phase["start_s"],phase["end_s"],color="#8DB4D6",alpha=.12)
        ax.axvline(phase["start_s"],color=".55",linewidth=.6)
        for start,end in phase["rank0_checkpoints"]:
            ax.axvspan(start,end,color=COLORS["write"],alpha=.2)


def activity(ax, intervals, y, color):
    merged=union(intervals)
    if merged:
        ax.hlines([y]*len(merged),[p[0] for p in merged],[p[1] for p in merged],color=color,linewidth=5,rasterized=True)
        zero=[p[0] for p in merged if p[0]==p[1]]
        if zero:
            ax.scatter(zero,[y]*len(zero),marker="|",s=12,color=color,rasterized=True)


def render(data, output, bin_seconds=1, name=None):
    if bin_seconds<=0:
        raise ValueError("bin_seconds must be positive")
    if name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",name):
        raise ValueError("Output name must contain only letters, digits, underscores or hyphens")
    events=data["events"]
    if not events:
        raise ValueError("No measured DXT events for A/B/C")
    left=min(0,math.floor(min(event["start_s"] for event in events)/bin_seconds)*bin_seconds)
    right=max(data["phases"][-1]["end_s"],max(event["end_s"] for event in events),
              (math.floor(max(event["start_s"] for event in events)/bin_seconds)+1)*bin_seconds)
    output.mkdir(parents=True,exist_ok=True)
    ranks=len(data["rank_api_bytes"])
    fig,axes=plt.subplots(4,1,figsize=(14,max(11,ranks*.4+8)),sharex=True,
                          gridspec_kw={"height_ratios":[.6,1.1,max(2,ranks*.27),1.6]})
    ax=axes[0]
    for phase in data["phases"]:
        start,end=phase["start_s"],phase["end_s"]
        ax.broken_barh([(start,end-start)],(.15,.7),facecolors="#8DB4D6")
        ax.vlines(start,.15,.85,color=".5",linewidth=.6)
        ax.text((start+end)/2,.5,f"Epoch {phase['epoch']}",ha="center",va="center")
        for start,end in phase["rank0_checkpoints"]:
            ax.broken_barh([(start,end-start)],(.25,.5),facecolors=COLORS["write"])
    ax.set(ylim=(0,1),yticks=[],ylabel="Native phases",title="A · Complete epoch bands with nested rank-0 checkpoint timers")
    ax=axes[1]
    phase_bands(ax,data)
    annotations=[]
    for y,op in enumerate(("read","write")):
        selected=[event for event in events if event["op"]==op]
        activity(ax,[(event["start_s"],event["end_s"]) for event in selected],y,COLORS[op])
        annotations.append(f"{op}: {len(selected):,} requests, {format_bytes(sum(e['length'] for e in selected))}")
    ax.set(yticks=[0,1],yticklabels=["Read union","Write union"],ylim=(-.5,1.5),xlim=(left,right),
           title="All-rank + reader activity · "+"; ".join(annotations))

    ax=axes[2]
    phase_bands(ax,data)
    for rank in range(len(data["rank_api_bytes"])):
        for delta,op in ((-.15,"read"),(.15,"write")):
            activity(ax,[(e["start_s"],e["end_s"]) for e in events if e["rank"]==rank and e["op"]==op],rank+delta,COLORS[op])
    for op,color in COLORS.items():
        ax.plot([],[],color=color,linewidth=4,label=op)
    ax.legend(loc="upper right",ncol=2)
    ax.set(yticks=range(ranks),ylabel="MPI owner rank",xlim=(left,right),
           ylim=(ranks-.5,-.5),title="B · Rank-owned activity, including recorded readers (same-operation unions)")

    counts=Counter((math.floor(e["start_s"]/bin_seconds),size_category(e["length"]),e["op"]) for e in events)
    scale=max(counts.values())
    ax=axes[3]
    phase_bands(ax,data)
    for op,delta in (("read",-.12),("write",.12)):
        selected=[(key,count) for key,count in counts.items() if key[2]==op]
        ax.scatter([(key[0]+.5)*bin_seconds for key,count in selected],
                   [key[1]+delta for key,count in selected],
                   s=[8+192*count/scale for key,count in selected],color=COLORS[op],alpha=.7,label=op)
    ax.set(yticks=range(5),yticklabels=SIZE_LABELS,ylim=(-.6,4.6),xlim=(left,right),
           xlabel="Seconds from native epoch-1 start",title=f"C · Request sizes over time · {bin_seconds:g} s bins · largest glyph = {scale:,} requests (shared read/write scale)")
    data["request_time_bins"]={"bin_seconds":bin_seconds,"largest_glyph_count":scale,
                              "counts":[{"bin":key[0],"category":key[1],"op":key[2],"count":count}
                                        for key,count in sorted(counts.items())]}
    model={"cosmoflow":"CosmoFlow","unet3d":"U-Net 3D"}.get(data["model"],data["model"])
    repetition=f" · repetition {data['repetition']}" if data.get("repetition") is not None else ""
    fig.suptitle(f"{model} · {data.get('pool_label','Single invocation')}{repetition} · {len(data['phases'])} epochs",fontsize=16)
    fig.text(.5,.955,CLOCK_NOTE,ha="center",fontsize=9,color="#795123")
    stdio_write=sum(row.get("STDIO_BYTES_WRITTEN",0) for row in data["rank_api_bytes"])
    fig.text(.5,.025,f"STDIO writes: {format_bytes(stdio_write)} aggregate only; no individual STDIO request timing.",ha="center",fontsize=9)
    fig.text(.5,.01,"POSIX API calls, not disk service. Blank time is unclassified. Orange phase shading marks rank-0 checkpoint timers.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(output/(f"{name}.png" if name else "ABC.png"),dpi=150)
    plt.close(fig)
    summary={key:value for key,value in data.items() if key!="events"}
    summary["event_count"]=len(events)
    summary["raw_event_source"]="Original per-process .darshan logs in the invocation's darshan/ directory"
    (output/(f"{name}.json" if name else "plot-data.json")).write_text(json.dumps(summary,indent=2)+"\n")
    # Remove only the superseded outputs of this plotter after successful save.
    if name is None:
        for old in ("A-phase-activity","B-rank-activity","C-request-sizes","pilot-darshan"):
            for suffix in ("png","svg"):
                (output/f"{old}.{suffix}").unlink(missing_ok=True)
        (output/"pilot-darshan.json").unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,help="One invocation directory, or pilot directory containing rep1-pool1")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--name",help="Output basename for a shared plot directory (letters, digits, underscore, hyphen)")
    parser.add_argument("--bin-seconds",type=float,default=1)
    parser.add_argument("--native-utc-offset",type=float,help="Original run host UTC offset in seconds; defaults to this host's offset")
    args=parser.parse_args()
    case=args.run if (args.run/"per_epoch_stats.json").is_file() else args.run/"rep1-pool1"
    if args.bin_seconds<=0:
        parser.error("--bin-seconds must be positive")
    data=collect(case,args.native_utc_offset)
    output=args.output or case/"darshan-plots"
    render(data,output,args.bin_seconds,name=args.name)
    print(f"Saved combined {args.name or 'ABC'}.png and plotted values in {output}")


if __name__=="__main__":
    main()
