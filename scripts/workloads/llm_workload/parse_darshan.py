#!/usr/bin/env python3
"""
parse_darshan.py — turn a run's Darshan logs into a per-file I/O profile for the
checkpoint-read experiment.

The Darshan counterpart to `parse_eval_strace.py`. It reads both views Darshan
offers and uses each for what only it can do:

  darshan-parser       per-file, per-rank aggregate counters. Cheap, always
                       present. Answers "which files, how many ops, how many
                       bytes" -- enough for the file-classification result.

  darshan-dxt-parser   every individual operation with offset, length and
                       timing. Must have been enabled with DXT_ENABLE_IO_TRACE.
                       The only way to show that different ranks read the *same
                       byte ranges*, which is the evidence for the read
                       amplification in METHODS 4.2, and the only way to measure
                       buffered over-fetch rather than estimate it.

Non-MPI Darshan writes one log per process, each self-reporting as rank 0, so
processes are identified by the pid in the log filename. Where an
`--emit-reads` CSV is supplied, each pid is matched back to its DCP rank by
comparing the set of (file, offset) pairs it touched.

Trust checks, because a quiet wrong answer is the failure mode here
------------------------------------------------------------------
  * Darshan bytes must EXCEED DCP's logical bytes (by the zip framing). Below
    1.0 means the log is incomplete.
  * DXT silently truncates when its per-process record buffer fills. The DXT
    operation count is compared against the aggregate POSIX_READS/WRITES
    counters, and a shortfall is reported as truncation rather than as data.
  * Files present in the checkpoint but absent from the logs are reported as
    never-opened only when the checkpoint directory is readable, so "never
    opened" cannot be confused with "not instrumented".

Usage
-----
    parse_darshan.py LOGDIR_OR_LOGS...
        [--ckpt DIR]           classify files as model / optimizer / index
        [--emit-reads CSV]     cross-check against DCP's own request list
        [--summary JSON]       cross-check against DCP's own byte total
        [--chunksize 524288 --numtargets 4]   attribute ops to stripe slots
        [--target-map JSON]    resolve slots to real BeeGFS target IDs:
            for f in CKPT/*.distcp; do
              printf '%s ' "$(basename $f)"
              beegfs-ctl --getentryinfo "$f" | awk '/^[+] *[0-9]+ @/ {print $2}' \\
                | paste -sd,
            done > targets.txt     # then convert to {"file": [ids]} JSON
        [--out PREFIX]         write PREFIX.files.csv, .ops.csv, .darshan.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

RE_PID = re.compile(r"_id\d+-(\d+)_")
RE_DXT_FILE = re.compile(r"^# DXT, file_id: \d+, file_name: (.*)$")
RE_DXT_OP = re.compile(
    r"^\s*X_POSIX\s+(\d+)\s+(read|write)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s*$"
)
RE_HDR = re.compile(r"^# (exe|jobid|nprocs|start_time|run time|darshan log version): (.*)$")

# aggregate counters we keep; the rest are noise for this experiment
KEEP = {
    "POSIX_OPENS", "POSIX_READS", "POSIX_WRITES", "POSIX_SEEKS",
    "POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN", "POSIX_SEQ_READS",
    "POSIX_CONSEC_READS", "POSIX_MAX_BYTE_READ", "POSIX_STATS",
}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def run_tool(tool: str, log: Path) -> str:
    try:
        r = subprocess.run([tool, "--show-incomplete", str(log)],
                           capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        raise SystemExit(f"{tool} not on PATH (darshan-util not installed?)")
    if r.returncode != 0 and not r.stdout:
        raise SystemExit(f"{tool} failed on {log}:\n{r.stderr[:500]}")
    return r.stdout


def parse_aggregate(text: str):
    """-> (job_info, {file_name: {counter: value}}, flags)

    `flags` carries two things the counters cannot show: which modules the log
    actually contains, and whether Darshan reported running out of record
    memory. The second matters enormously -- a truncated POSIX module reports
    *some* files and silently omits the rest, so a zero looks like "no I/O"
    when it means "we stopped recording before your I/O happened".
    """
    job, per_file = {}, collections.defaultdict(dict)
    flags = {"incomplete": False, "modules": set()}
    for line in text.splitlines():
        if line.startswith("#"):
            m = RE_HDR.match(line)
            if m:
                job[m.group(1)] = m.group(2).strip()
            if "contains incomplete data" in line:
                flags["incomplete"] = True
            mm = re.match(r"^# (\w+) module: ", line)
            if mm:
                flags["modules"].add(mm.group(1))
            continue
        parts = line.split("\t")
        if len(parts) < 6 or parts[0] != "POSIX":
            continue
        counter, value, fname = parts[3], parts[4], parts[5]
        if counter in KEEP:
            try:
                per_file[fname][counter] = per_file[fname].get(counter, 0) + int(value)
            except ValueError:
                pass
    return job, per_file, flags


def parse_dxt(text: str):
    """-> list of {file, op, offset, length, start, end}, POSIX module only."""
    ops, cur, in_posix = [], None, False
    for line in text.splitlines():
        if line.startswith("# DXT_POSIX module data"):
            in_posix = True
            continue
        if line.startswith("# DXT_MPIIO module data"):
            in_posix = False
            continue
        if not in_posix:
            continue
        m = RE_DXT_FILE.match(line)
        if m:
            cur = m.group(1).strip()
            continue
        m = RE_DXT_OP.match(line)
        if m and cur:
            ops.append({
                "file": cur, "op": m.group(2),
                "offset": int(m.group(4)), "length": int(m.group(5)),
                "start": float(m.group(6)), "end": float(m.group(7)),
            })
    return ops


def union_bytes(intervals) -> int:
    """Distinct bytes covered by a set of (offset, length) ranges."""
    if not intervals:
        return 0
    spans = sorted((o, o + n) for o, n in intervals if n > 0)
    if not spans:
        return 0
    total, cs, ce = 0, *spans[0]
    for s, e in spans[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def chunk_spans(offset: int, length: int, chunksize: int, numtargets: int):
    """Bytes this operation places on each stripe slot it touches.

    Yields ``(slot, nbytes)``, one entry per *distinct* slot, summed over every
    chunk the operation crosses.

    An earlier version attributed each operation wholly to the slot of its
    starting offset. That is wrong for exactly the operations that matter: a
    262 MB read (the largest in the 8B eval trace) spans ~500 chunks and lands
    on every target, but was counted once, on one. The resulting histogram
    looked like a 230x load imbalance that does not exist.
    """
    if length <= 0 or not chunksize or not numtargets:
        return
    acc: dict = collections.defaultdict(int)
    pos, end = offset, offset + length
    while pos < end:
        chunk_end = (pos // chunksize + 1) * chunksize
        n = min(chunk_end, end) - pos
        acc[(pos // chunksize) % numtargets] += n
        pos += n
    yield from sorted(acc.items())


def resolve_slots(spans, tmap):
    """Map stripe slots to real BeeGFS target IDs when a target map is given."""
    if not tmap:
        return list(spans)
    return [(tmap[s] if s < len(tmap) else s, n) for s, n in spans]


# ---------------------------------------------------------------------------
# checkpoint context
# ---------------------------------------------------------------------------


def checkpoint_context(ckpt: Path):
    """-> (kind_of_file, size_of_file) from the checkpoint's own .metadata."""
    md_path = ckpt / ".metadata"
    if not md_path.exists():
        return {}, {}
    md = pickle.load(open(md_path, "rb"))
    kinds = collections.defaultdict(set)
    extent: dict[str, int] = {}
    for key, si in md.storage_data.items():
        k = "optim" if key.fqn.startswith("optim.") else "model"
        kinds[si.relative_path].add(k)
        extent[si.relative_path] = max(extent.get(si.relative_path, 0),
                                       si.offset + si.length)
    kind = {f: ("mixed" if len(v) > 1 else next(iter(v))) for f, v in kinds.items()}
    kind[".metadata"] = "index"
    size = {}
    for f in list(kind):
        p = ckpt / f
        size[f] = p.stat().st_size if p.exists() else extent.get(f, 0)
    return kind, size


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="log files or a directory")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--emit-reads", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--chunksize", type=int, default=None,
                    help="BeeGFS stripe chunk size, for stripe-slot attribution")
    ap.add_argument("--numtargets", type=int, default=None)
    ap.add_argument("--target-map", type=Path, default=None,
                    help='JSON {"__0_0.distcp": [401,403,405,406], ...} mapping '
                         "each file to its ordered target list, so slots resolve "
                         "to real target IDs (see the docstring for how to build it)")
    ap.add_argument("--match", default=None,
                    help="only consider paths containing this substring; "
                         "defaults to the --ckpt directory. Without it, every "
                         "file the process touched is counted -- including "
                         "Python's own shared libraries.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--parser", default="darshan-parser")
    ap.add_argument("--dxt-parser", default="darshan-dxt-parser")
    args = ap.parse_args()

    logs: list[Path] = []
    for p in args.logs:
        logs.extend(sorted(p.glob("*.darshan")) if p.is_dir() else [p])
    if not logs:
        raise SystemExit("no .darshan logs found")
    for tool in (args.parser, args.dxt_parser):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not on PATH")

    # Default the filter to the checkpoint directory. A Darshan log covers
    # everything the process opened, so without this the counts include libc,
    # every .so the interpreter loaded, and anything else on the way past.
    if args.match is None:
        args.match = str(args.ckpt) if args.ckpt else ""
    if not args.match:
        print("  NOTE: no --ckpt or --match given, so every file in the logs is"
              " counted,\n        including shared libraries. Numbers will be"
              " inflated.\n")

    kind, disk_size = checkpoint_context(args.ckpt) if args.ckpt else ({}, {})
    target_map = json.load(open(args.target_map)) if args.target_map else {}

    agg = collections.defaultdict(lambda: collections.Counter())
    agg_seen: dict[str, bool] = {}
    all_seen: collections.Counter = collections.Counter()  # for the diagnostic
    ops: list[dict] = []
    procs, job_info = [], {}
    modules: set = set()
    incomplete: list = []
    # Non-MPI Darshan writes one log per process, and DXT timestamps are
    # relative to *that process's* start_time. Two ranks' t=0 are therefore
    # different wall-clock instants, separated by the spawn skew. Keeping each
    # process's start_time lets the plots put all ranks on one absolute clock;
    # without it every cross-rank temporal figure is silently misaligned.
    proc_start: dict = {}
    for log in logs:
        pid_m = RE_PID.search(log.name)
        pid = pid_m.group(1) if pid_m else log.stem
        procs.append(pid)
        job, per_file, flags = parse_aggregate(run_tool(args.parser, log))
        job_info = job_info or job
        try:
            proc_start[pid] = float(job.get("start_time", "nan"))
        except ValueError:
            proc_start[pid] = float("nan")
        modules |= flags["modules"]
        if flags["incomplete"]:
            incomplete.append(pid)
        for fname, counters in per_file.items():
            if args.match and args.match not in fname:
                all_seen[os.path.basename(fname)] += counters.get(
                    "POSIX_BYTES_READ", 0)
                continue
            agg_seen[pid] = True
            agg[os.path.basename(fname)].update(counters)
        for o in parse_dxt(run_tool(args.dxt_parser, log)):
            if args.match and args.match not in o["file"]:
                continue
            o["proc"] = pid
            o["base"] = os.path.basename(o["file"])
            ops.append(o)

    # DXT is "on" if the module is in the log -- NOT if ops survived filtering.
    # Inferring it from the filtered ops reported "DXT OFF" on logs that plainly
    # contained a 30 KB DXT_POSIX module.
    dxt_on = any(m.startswith("DXT_") for m in modules) or bool(ops)

    # ---- pid -> DCP rank, by matching (file, offset) sets --------------
    rank_of = {}
    if args.emit_reads and args.emit_reads.exists() and dxt_on:
        want = collections.defaultdict(set)
        for r in csv.DictReader(open(args.emit_reads)):
            want[r["rank"]].add((r["file"], int(r["offset"])))
        have = collections.defaultdict(set)
        for o in ops:
            have[o["proc"]].add((o["base"], o["offset"]))
        for pid in procs:
            if pid in rank_of or pid not in have:
                continue
            best = max(want, key=lambda rk: len(want[rk] & have[pid]), default=None)
            if best is not None and (want[best] & have[pid]):
                rank_of[pid] = best
                want.pop(best)

    # ---- per-file rollup ----------------------------------------------
    by_file = collections.defaultdict(lambda: {"ops": [], "procs": set()})
    for o in ops:
        e = by_file[o["base"]]
        e["ops"].append(o)
        e["procs"].add(o["proc"])

    rows = []
    for fname in sorted(set(agg) | set(by_file)):
        a, d = agg.get(fname, collections.Counter()), by_file.get(fname)
        reads = [o for o in (d["ops"] if d else []) if o["op"] == "read"]
        got = sum(o["length"] for o in reads)
        uniq = union_bytes([(o["offset"], o["length"]) for o in reads])
        sizes = sorted(o["length"] for o in reads)
        tgts = set()
        if args.chunksize and args.numtargets:
            tmap = target_map.get(fname)
            for o in reads:
                spans = chunk_spans(o["offset"], o["length"],
                                    args.chunksize, args.numtargets)
                tgts.update(t for t, _ in resolve_slots(spans, tmap))
        bytes_read = got or a.get("POSIX_BYTES_READ", 0)
        on_disk = disk_size.get(fname, 0)
        rows.append({
            "file": fname,
            "kind": kind.get(fname, ""),
            "bytes_on_disk": on_disk,
            "readers": len(d["procs"]) if d else (1 if a.get("POSIX_READS") else 0),
            "opens": a.get("POSIX_OPENS", 0),
            "reads": a.get("POSIX_READS", len(reads)),
            "bytes_read": bytes_read,
            "read_frac": round(uniq / on_disk, 4) if on_disk else "",
            "amplif": round(got / uniq, 3) if uniq else "",
            "mean_req": round(statistics.mean(sizes)) if sizes else "",
            "p95_req": sizes[int(0.95 * (len(sizes) - 1))] if sizes else "",
            "seq_frac": (round(a["POSIX_SEQ_READS"] / a["POSIX_READS"], 3)
                         if a.get("POSIX_READS") else ""),
            "first_s": round(min(o["start"] for o in reads), 4) if reads else "",
            "last_s": round(max(o["end"] for o in reads), 4) if reads else "",
            "targets": ",".join(str(t) for t in sorted(tgts)),
        })

    touched = {r["file"] for r in rows}
    never = sorted(set(kind) - touched - {".metadata"}) if kind else []
    never_bytes = sum(disk_size.get(f, 0) for f in never)

    tot_reads = sum(r["reads"] for r in rows)
    tot_bytes = sum(r["bytes_read"] for r in rows)
    all_sizes = sorted(o["length"] for o in ops if o["op"] == "read")
    global_uniq = sum(
        union_bytes([(o["offset"], o["length"])
                     for o in by_file[f]["ops"] if o["op"] == "read"])
        for f in by_file)

    # ---- trust checks --------------------------------------------------
    warn = []
    agg_reads = sum(agg[f].get("POSIX_READS", 0) for f in agg)
    dxt_reads = len(all_sizes)
    if dxt_on and agg_reads and dxt_reads < agg_reads * 0.99:
        warn.append(f"DXT TRUNCATED: {dxt_reads:,} traced vs {agg_reads:,} counted "
                    f"({100*dxt_reads/agg_reads:.1f}%). Raise the DXT buffer or "
                    f"treat per-offset results as a sample.")
    if incomplete:
        warn.append(
            f"DARSHAN RAN OUT OF RECORD MEMORY in {len(incomplete)} of "
            f"{len(logs)} log(s) (pids {', '.join(incomplete[:6])}). Records are "
            f"allocated in open() order, so the interpreter's thousands of "
            f"import files consumed the buffer and the checkpoint reads were "
            f"never recorded. Raise DARSHAN_MODMEM (MiB, default 2) and set "
            f"DARSHAN_EXCLUDE_DIRS to skip the interpreter's directories -- "
            f"note it REPLACES the built-in list. run/env.sh does both.")
    if not dxt_on:
        warn.append("No DXT module in any log. Amplification and over-fetch are "
                    "unavailable; set DXT_ENABLE_IO_TRACE=1 and re-run.")
    elif dxt_on and not ops:
        warn.append("DXT module present but no records matched the checkpoint "
                    "-- the trace was filled by other files before the "
                    "workload's I/O. Same fix as above.")
    logical = None
    if args.summary and args.summary.exists():
        logical = json.load(open(args.summary)).get("total_bytes")
    if logical:
        ratio = tot_bytes / logical
        if ratio < 0.98:
            warn.append(f"TRACE INCOMPLETE: Darshan {tot_bytes/1e9:.2f} GB is BELOW "
                        f"DCP's logical {logical/1e9:.2f} GB ({ratio:.2f}x). "
                        f"Darshan should exceed it by the zip framing.")

    # ---- report --------------------------------------------------------
    # A log that touched no checkpoint file is not the workload -- it is some
    # other command that inherited LD_PRELOAD. Counting those as ranks is how a
    # 2-rank job came out as 4 processes.
    useful = {o["proc"] for o in ops} | {p for p in procs if agg_seen.get(p)}
    stray = [p for p in procs if p not in useful]

    print(f"=== darshan: {len(logs)} log(s), {len(useful)} process(es) touched "
          f"the checkpoint ===")
    print(f"modules          {', '.join(sorted(modules)) or 'none'}"
          + (f"   [{len(incomplete)} log(s) TRUNCATED]" if incomplete else ""))
    if stray:
        print(f"  ignoring {len(stray)} log(s) that touched no checkpoint file "
              f"(pids {', '.join(stray[:6])}{'...' if len(stray) > 6 else ''})")
    if job_info.get("exe"):
        print(f"exe              {job_info['exe'][:70]}")
    if kind:
        n_shard = len([f for f in kind if f != ".metadata"])
        n_touch = len([f for f in touched if f != ".metadata"])
        print(f"shard files      {n_touch:,} of {n_shard:,} opened"
              f"    never opened {len(never):,} ({never_bytes/1e9:.1f} GB)")
        print(f".metadata        {'read' if '.metadata' in touched else 'NOT READ'}"
              f"   (every rank reads it before any data read)")
    else:
        print(f"files touched    {len(touched):,}   (--ckpt not given: no "
              f"never-opened figure)")
    print(f"read ops         {tot_reads:,}              bytes read "
          f"{tot_bytes/1e9:.3f} GB")
    if logical:
        print(f"vs DCP logical   {logical/1e9:.3f} GB  ->  {tot_bytes/logical:.2f}x"
              f"   (over-fetch {(tot_bytes-logical)/1e9:+.3f} GB)")
    if all_sizes:
        print(f"request sizes    p50 {all_sizes[len(all_sizes)//2]/1e3:,.1f} KB"
              f"   p95 {all_sizes[int(0.95*(len(all_sizes)-1))]/1e6:,.2f} MB"
              f"   max {all_sizes[-1]/1e6:,.2f} MB")
        print(f"amplification    {tot_bytes/global_uniq:.2f}x"
              f"   (distinct bytes {global_uniq/1e9:.3f} GB)")
    if rank_of:
        print(f"pid->rank        {', '.join(f'{p}->{r}' for p, r in sorted(rank_of.items()))}")
    if args.chunksize and args.numtargets:
        # Bytes, not ops, and split across every chunk an operation crosses --
        # see chunk_spans(). Ops are also counted, but an op that spans several
        # slots counts once toward each, so the op totals exceed len(ops).
        per_t_bytes: dict = collections.Counter()
        per_t_ops: dict = collections.Counter()
        for o in ops:
            if o["op"] != "read":
                continue
            spans = resolve_slots(
                chunk_spans(o["offset"], o["length"], args.chunksize,
                            args.numtargets), target_map.get(o["base"]))
            for t, n in spans:
                per_t_bytes[t] += n
                per_t_ops[t] += 1
        label = "target" if target_map else "stripe slot"
        tot_t = sum(per_t_bytes.values()) or 1
        print(f"{label+'s':<16} {len(per_t_bytes)} of {args.numtargets} used")
        for t in sorted(per_t_bytes):
            print(f"  {label} {t:<8} {per_t_bytes[t]/1e9:7.3f} GB "
                  f"({100*per_t_bytes[t]/tot_t:5.1f}%)  "
                  f"{per_t_ops[t]:,} ops touching it")
        if not target_map:
            print("                 NB: these are slots WITHIN each file's own "
                  "target list,")
            print("                 not target IDs. BeeGFS gives each file a "
                  "different starting")
            print("                 target, so slot 0 of two files is usually two "
                  "different")
            print("                 disks. Pass --target-map to resolve them.")
    if kind:
        bk = collections.Counter()
        for r in rows:
            bk[r["kind"] or "?"] += r["bytes_read"]
        print(f"bytes by kind    "
              + "  ".join(f"{k}={v/1e9:.2f} GB" for k, v in sorted(bk.items())))
    if not agg and not ops:
        print("\n  *** Darshan recorded NO I/O on the checkpoint. What it did "
              "record, by bytes read:")
        for fname, b in all_seen.most_common(12):
            print(f"        {b/1e6:10.2f} MB  {fname}")
        if not all_seen:
            print("        (nothing at all -- the traced processes did no "
                  "instrumented I/O)")
        print(f"      Filter in use: --match {args.match!r}")
        if incomplete:
            print("      The logs are TRUNCATED (see below) -- this is a record"
                  " memory\n      problem, not a filter problem.")
        else:
            print("      If the checkpoint files appear above, the filter is"
                  " wrong.\n      If they do not, the workload processes were"
                  " not traced at all.")

    for w in warn:
        print(f"\n  *** {w}")

    # ---- outputs -------------------------------------------------------
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fcsv = Path(f"{args.out}.files.csv")
        # Fixed schema: deriving it from rows[0] broke whenever the trace was
        # empty, because the never-opened rows still carry the full set of keys.
        cols = ["file", "kind", "bytes_on_disk", "readers", "opens", "reads",
                "bytes_read", "read_frac", "amplif", "mean_req", "p95_req",
                "seq_frac", "first_s", "last_s", "targets"]
        with open(fcsv, "w", newline="") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(rows)
            for n in never:
                w.writerow({"file": n, "kind": kind.get(n, ""),
                            "bytes_on_disk": disk_size.get(n, 0), "readers": 0,
                            "opens": 0, "reads": 0, "bytes_read": 0,
                            "read_frac": 0, "amplif": ""})
        ocsv = Path(f"{args.out}.ops.csv")
        with open(ocsv, "w", newline="") as f:
            w = csv.writer(f)
            # `target` is the slot the operation STARTS on, kept because it is
            # cheap to group by. `targets`/`target_bytes` are the honest view:
            # parallel comma-separated lists of every slot the operation
            # touches and how many bytes land on each. Plots must use those.
            w.writerow(["rank", "pid", "file", "op", "offset", "length",
                        "start_s", "end_s", "start_abs", "end_abs", "target",
                        "targets", "target_bytes", "n_targets"])
            t0 = min((v for v in proc_start.values() if v == v), default=0.0)
            for o in ops:
                t, tl, bl = "", "", ""
                if args.chunksize and args.numtargets:
                    spans = resolve_slots(
                        chunk_spans(o["offset"], o["length"], args.chunksize,
                                    args.numtargets), target_map.get(o["base"]))
                    if spans:
                        # the slot the op starts on -- not spans[0], which is
                        # the lowest-numbered slot, since chunk_spans sorts
                        s0 = (o["offset"] // args.chunksize) % args.numtargets
                        tm = target_map.get(o["base"])
                        t = tm[s0] if tm and s0 < len(tm) else s0
                        tl = ",".join(str(x) for x, _ in spans)
                        bl = ",".join(str(n) for _, n in spans)
                # shift each process's relative clock onto a common origin
                base = proc_start.get(o["proc"], float("nan"))
                sa = ea = ""
                if base == base:  # not NaN
                    sa = round(base - t0 + o["start"], 6)
                    ea = round(base - t0 + o["end"], 6)
                w.writerow([rank_of.get(o["proc"], ""), o["proc"], o["base"],
                            o["op"], o["offset"], o["length"],
                            o["start"], o["end"], sa, ea, t, tl, bl,
                            len(tl.split(",")) if tl else ""])
        jpath = Path(f"{args.out}.darshan.json")
        jpath.write_text(json.dumps({
            "logs": [str(p) for p in logs], "processes": sorted(set(procs)),
            "dxt": dxt_on, "job": job_info,
            "files_touched": len(touched), "files_never_opened": never,
            "never_opened_bytes": never_bytes,
            "read_ops": tot_reads, "bytes_read": tot_bytes,
            "distinct_bytes": global_uniq,
            "amplification": (tot_bytes / global_uniq) if global_uniq else None,
            "logical_bytes": logical,
            "pid_to_rank": rank_of, "warnings": warn,
            "proc_start_time": proc_start,
            "clock_skew_s": (max(v for v in proc_start.values() if v == v)
                             - min(v for v in proc_start.values() if v == v))
            if any(v == v for v in proc_start.values()) else None,
            "chunksize": args.chunksize, "numtargets": args.numtargets,
            "target_map": bool(target_map),
        }, indent=2))
        print(f"\nwrote {fcsv}\n      {ocsv}\n      {jpath}")


if __name__ == "__main__":
    main()
