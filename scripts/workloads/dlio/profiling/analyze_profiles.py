#!/usr/bin/env python3
"""Validate per-process Darshan/DXT capture and plot DLIO phase/access evidence."""

import argparse
import bisect
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows, fields=None):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or (list(rows[0]) if rows else []))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(lines):
    metadata, files, flags = {}, defaultdict(dict), []
    for line in lines:
        if line.startswith("#"):
            if "incomplete data" in line.lower() or "ran out" in line.lower():
                flags.append(line.strip())
            match = re.match(r"# (exe|nprocs|start_time|end_time|darshan log version):\s*(.*)", line)
            if match:
                metadata[match[1]] = match[2].strip()
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            parts = line.split(None, 7)
        if len(parts) >= 6 and parts[0] in ("POSIX", "STDIO"):
            if int(parts[1]) not in (0,):
                flags.append("Expected non-MPI, unreduced per-process records")
            files[parts[5]][parts[3]] = float(parts[4])
    return metadata, files, flags


def log_pid(path):
    # Fixed names from early captures remain readable for diagnostics.
    match = re.fullmatch(r"pid-(\d+)[.]darshan",path.name)
    if match:
        return int(match[1])
    match = re.search(r"_id\d+-(\d+)_",path.name)
    if match:
        return int(match[1])
    raise ValueError(f"Cannot identify native Darshan log PID: {path.name}")


def helper_log(path, manifest, output, run):
    """Ignore a helper only after proving it has no measured-path records."""
    with tempfile.TemporaryDirectory(prefix="helper-",dir=output) as directory:
        parsed=Path(directory)/"aggregate.txt"
        with parsed.open("w") as file, (output/"helper-parser.stderr").open("a") as stderr:
            run(["darshan-parser","--show-incomplete",str(path)],stdout=file,stderr=stderr,check=True,text=True)
        with parsed.open() as file:
            metadata,counters,flags=aggregate(file)
    if int(metadata.get("nprocs",0)) != 1 or flags:
        raise ValueError(f"Unvalidated helper log: {path}")
    if any(role(name,manifest) in ("dataset","checkpoint","anchor") for name in counters):
        raise ValueError(f"Unmapped process touched a measured path: {path}; refusing to discard its I/O")
    return {"path":str(path),"pid":log_pid(path),"exe":metadata.get("exe",""),
            "reason":"No dataset/checkpoint/clock-marker records in complete POSIX/STDIO log"}


def dxt_events(lines):
    """Darshan 3.4/3.5 text formats, including optional pthread IDs."""
    filename, enabled = None, False
    for line in lines:
        if line.startswith("# DXT_POSIX module data"):
            enabled = True
        elif line.startswith("# DXT_MPIIO module data"):
            enabled = False
        if not enabled:
            continue
        match = re.match(r"# DXT, file_id:\s*\d+, file_name: (.*)", line)
        if match:
            filename = match[1].strip()
        parts = line.split()
        if parts and parts[0] == "X_POSIX":
            if filename is None or len(parts) < 8:
                raise ValueError("Malformed DXT POSIX event")
            rank, op = int(parts[1]), parts[2]
            offset, length = int(parts[4]), int(parts[5])
            start, end = float(parts[6]), float(parts[7])
            if rank != 0 or op not in ("read", "write") or length < 0 or end < start:
                raise ValueError("Invalid event or unexpected MPI-mode Darshan record")
            yield {"file": filename, "op": op, "offset": offset, "length": length,
                   "start": start, "end": end,
                   "thread": parts[8] if len(parts) > 8 else "N/A"}


def role(path, manifest):
    for name, root in (("dataset", manifest["dataset"]), ("checkpoint", manifest["checkpoints"])):
        if path == root or path.startswith(root.rstrip("/") + "/"):
            return name
    return "anchor" if ".anchor-" in path else "other"


def clock_offset(anchor, markers, max_error=.1):
    estimates = []
    for kind in ("start", "end"):
        event = markers.get(anchor.get(kind, {}).get("path"))
        if event is None:
            raise ValueError(f"PID {anchor['pid']}: missing {kind} clock-marker event")
        recorded = anchor[kind]
        # Text utilities round DXT times to four decimals on some versions.
        low = recorded["wall_before_ns"] / 1e9 - event["start"] - .0001
        high = recorded["wall_after_ns"] / 1e9 - event["end"] + .0001
        if high < low or (high-low)/2 > max_error:
            raise ValueError(f"PID {anchor['pid']}: unreliable clock alignment")
        estimates.append((low, high))
    low, high = max(x[0] for x in estimates), min(x[1] for x in estimates)
    if high < low:
        raise ValueError(f"PID {anchor['pid']}: clock drift exceeds marker uncertainty")
    return (low+high)/2, (high-low)/2


def resolve_markers(anchor, markers, counters):
    """Use measured POSIX timestamps if only a DXT marker record is absent.

    A single known-size write has the same Darshan clock in either module.
    Never substitute file mtimes, job start seconds or native epoch times.
    """
    result, sources = dict(markers), {}
    marker_bytes = len(b"PFS-DXT-clock-anchor\n")
    for kind in ("start","end"):
        path = anchor[kind]["path"]
        if path in result:
            sources[path] = "DXT write event"
            continue
        values = counters.get(path,{})
        start, end = values.get("POSIX_F_WRITE_START_TIMESTAMP"), values.get("POSIX_F_WRITE_END_TIMESTAMP")
        if (values.get("POSIX_WRITES") != 1 or values.get("POSIX_BYTES_WRITTEN") != marker_bytes
                or values.get("POSIX_READS",0) != 0 or start is None or end is None
                or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start):
            raise ValueError(f"PID {anchor['pid']}: missing {kind} clock-marker event and no unambiguous POSIX marker timestamp; inspect analysis/coverage.json clock_markers")
        result[path] = {"start":start,"end":end}
        sources[path] = "POSIX timestamps of one verified marker write; DXT marker absent"
    return result, sources


def stamp(value, manifest):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(seconds=manifest["timezone_offset_s"])))
    return parsed.timestamp()


def reconstruct(case, manifest, anchors):
    summary = json.loads((case / "raw/summary.json").read_text())
    native = json.loads((case / "raw/per_epoch_stats.json").read_text())
    if sorted(map(int, native)) != [1, 2, 3, 4, 5] or summary["num_accelerators"] != 8:
        raise ValueError("Incomplete five-epoch/eight-rank DLIO evidence")
    start = min(a["start"]["wall_before_ns"] / 1e9 for a in anchors if a["role"] == "rank")
    finish = max(a["end"]["wall_after_ns"] / 1e9 for a in anchors)
    phases, checkpoints = [], []
    cursor = 0.0
    for epoch in range(1, 6):
        data = native[str(epoch)]
        left, right = stamp(data["start"], manifest)-start, stamp(data["end"], manifest)-start
        if left < cursor or right <= left:
            raise ValueError("Native epoch boundaries overlap or are invalid")
        if left > cursor:
            phases.append({"phase": "initialization" if epoch == 1 else f"gap_before_E{epoch}",
                           "start_s": cursor, "end_s": left})
        phases.append({"phase": f"E{epoch}", "start_s": left, "end_s": right})
        cursor = right
        if epoch in (2, 4):
            event = data["ckpt1"]
            a, b = stamp(event["start"], manifest)-start, stamp(event["end"], manifest)-start
            if a < left or b > right or b <= a:
                raise ValueError("Checkpoint timer must be nested within its epoch")
            checkpoints.append({"phase": f"C{epoch//2}", "epoch": epoch,
                                "start_s": a, "end_s": b, "scope": "native rank-0 timer"})
    if finish-start < cursor:
        raise ValueError("Process end anchor precedes last native epoch")
    if finish-start > cursor:
        phases.append({"phase": "finalization", "start_s": cursor, "end_s": finish-start})
    elapsed = stamp(summary["end"], manifest)-stamp(summary["start"], manifest)
    import yaml
    config = yaml.safe_load((case / "raw/.hydra/config.yaml").read_text())["workload"]
    expected = manifest["settings"]
    for section, keys in expected.items():
        for key, value in keys.items():
            if config[section].get(key) != value:
                raise ValueError(f"Workload setting changed: {section}.{key}")
    steps = config["dataset"]["num_files_train"] * config["dataset"]["num_samples_per_file"] // (8*config["reader"]["batch_size"])
    for rank in range(8):
        data = json.loads((case / f"raw/{rank}_output.json").read_text())
        if any(len(data[str(epoch)]["load"]["block1"]) != steps for epoch in range(1, 6)):
            raise ValueError(f"Rank {rank}: missing load steps")
    if elapsed <= 0:
        raise ValueError("Invalid end-to-end duration")
    return start, phases, checkpoints, {"chart_duration_s": finish-start,
        "dlio_start_s": stamp(summary["start"], manifest)-start,
        "dlio_end_s": stamp(summary["end"], manifest)-start,
        "whole_run_s": elapsed, "total_processed_samples": steps*8*config["reader"]["batch_size"]*5,
        "end_to_end_samples_s": steps*8*config["reader"]["batch_size"]*5/elapsed}


def union(spans):
    merged = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def analyze_case(case, execute=None, max_clock_error=.1):
    """Fail closed on missing processes, truncation or wrong byte accounting."""
    case = Path(case)
    output = case / "analysis"
    output.mkdir(exist_ok=True)
    manifest = json.loads((case / "case.json").read_text())
    report = {"valid": False, "api_layer": "POSIX", "errors": [], "clock_alignment": {},
              "limitations": ["API calls are not physical-device service times",
                             "Metadata counters are whole-process, not phase-resolved RPC timings"]}
    database = output / "events.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.executescript("DROP TABLE IF EXISTS events; CREATE TABLE events (pid INTEGER, rank INTEGER, role TEXT, file TEXT, op TEXT, offset INTEGER, length INTEGER, start REAL, end REAL, thread TEXT, phase TEXT, crossing INTEGER);")
        anchors = [json.loads(path.read_text()) for path in sorted((case / "anchors").glob("pid-*.json"))]
        roots = [a for a in anchors if a["role"] == "rank"]
        if sorted(a["rank"] for a in roots) != list(range(8)):
            raise ValueError("Expected exactly eight rank-process anchors")
        if len({a["hostname"] for a in anchors}) != 1:
            raise ValueError("This protocol expects all ranks/readers on one client")
        logs = defaultdict(list)
        for path in (case/"darshan").rglob("*.darshan"):
            logs[log_pid(path)].append(path)
        anchored={a["pid"] for a in anchors}
        if not anchored <= set(logs):
            raise ValueError(f"Missing process logs: {sorted(anchored-set(logs))}")
        run = execute or subprocess.run
        report["ignored_helper_logs"] = [helper_log(path,manifest,output,run)
            for pid,paths in logs.items() if pid not in anchored for path in paths]
        for pid in anchored:
            if len(logs[pid]) != 1:
                raise ValueError(f"Multiple logs for anchored PID {pid}; inspect process lifecycle")
        for anchor in anchors:
            if anchor["owner_pid"] not in {a["pid"] for a in roots} or not 0 <= anchor["rank"] < 8:
                raise ValueError("Unmapped reader process")
            if "end" not in anchor:
                raise ValueError(f"PID {anchor['pid']}: incomplete process lifecycle")
        if manifest.get("smoke"):
            origin = min(a["start"]["wall_before_ns"]/1e9 for a in anchors)
            finish = max(a["end"]["wall_after_ns"]/1e9 for a in anchors)
            phases, checkpoints, timings = [{"phase": "smoke", "start_s": 0, "end_s": finish-origin}], [], {}
        else:
            origin, phases, checkpoints, timings = reconstruct(case, manifest, anchors)
        ends = [phase["end_s"] for phase in phases]
        metadata_rows, batch = [], []
        for anchor in anchors:
            pid = anchor["pid"]
            with tempfile.TemporaryDirectory(prefix="parser-", dir=output) as temporary:
                parsed, traced = Path(temporary)/"aggregate.txt", Path(temporary)/"dxt.txt"
                for tool, destination in (("darshan-parser", parsed), ("darshan-dxt-parser", traced)):
                    with destination.open("w") as file, (output / f"pid-{pid}-{tool}.stderr").open("w") as err:
                        run([tool, "--show-incomplete", str(logs[pid][0])], stdout=file, stderr=err, check=True, text=True)
                with parsed.open() as file:
                    metadata, counters, flags = aggregate(file)
                report.setdefault("process_logs",{})[str(pid)] = {
                    "path":str(logs[pid][0]),"exe":metadata.get("exe",""),
                    "start_time":metadata.get("start_time"),"end_time":metadata.get("end_time")}
                if int(metadata.get("nprocs", 0)) != 1 or flags:
                    raise ValueError(f"PID {pid}: incomplete or MPI-reduced log: {flags}")
                markers = {}
                with traced.open() as file:
                    for event in dxt_events(file):
                        if event["file"] in {anchor["start"]["path"], anchor["end"]["path"]} and event["op"] == "write":
                            if event["file"] in markers:
                                raise ValueError("Clock marker must contain exactly one write")
                            markers[event["file"]] = event
                report.setdefault("clock_markers",{})[str(pid)] = {
                    "expected_paths":{kind:anchor[kind]["path"] for kind in ("start","end")},
                    "dxt_marker_paths":sorted(markers),
                    "posix_marker_records":{path:values for path,values in counters.items() if ".anchor-" in path}}
                markers, marker_sources = resolve_markers(anchor,markers,counters)
                offset, error = clock_offset(anchor, markers, max_clock_error)
                report["clock_alignment"][str(pid)] = {"wall_minus_dxt_s": offset, "uncertainty_s": error,
                                                       "marker_sources":marker_sources}
                observed = defaultdict(Counter)
                with traced.open() as file:
                    for event in dxt_events(file):
                        name, op = event["file"], event["op"]
                        observed[name][op+"_count"] += 1
                        observed[name][op+"_bytes"] += event["length"]
                        kind = role(name, manifest)
                        if kind not in ("dataset", "checkpoint"):
                            continue
                        left, right = event["start"]+offset-origin, event["end"]+offset-origin
                        index = bisect.bisect_left(ends, right)
                        phase = phases[index] if index < len(phases) and right >= phases[index]["start_s"] else None
                        crossing = int(phase is None or left < phase["start_s"])
                        batch.append((pid, anchor["rank"], kind, name, op, event["offset"], event["length"],
                                      left, right, event["thread"], phase["phase"] if phase else "outside", crossing))
                        if len(batch) >= 10000:
                            connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                            batch.clear()
                            if execute:
                                # The reservation runner supplies a time-budget check.
                                from pool_common import remaining
                                remaining()
                for name in set(counters) | set(observed):
                    values = counters.get(name, {})
                    actual = observed[name]
                    if name in marker_sources and marker_sources[name].startswith("POSIX timestamps"):
                        if actual["read_count"] or actual["write_count"]:
                            raise ValueError("Ambiguous partially traced clock marker")
                        continue  # Sole marker write validated above; workload coverage remains strict.
                    for op, suffix in (("read", "READ"), ("write", "WRITTEN")):
                        count_key = "POSIX_READS" if op == "read" else "POSIX_WRITES"
                        expected_count = int(values.get(count_key, 0))
                        expected_bytes = int(values.get("POSIX_BYTES_"+suffix, 0))
                        if expected_count != actual[op+"_count"] or expected_bytes != actual[op+"_bytes"]:
                            raise ValueError(f"PID {pid}, {name}: DXT/aggregate count or byte mismatch ({op}); possible truncation")
                    if role(name, manifest) in ("dataset", "checkpoint"):
                        if (values.get("STDIO_BYTES_READ", 0) or values.get("STDIO_BYTES_WRITTEN", 0)) and not (actual["read_bytes"] or actual["write_bytes"]):
                            raise ValueError(f"STDIO-only file lacks POSIX DXT coverage: {name}")
                        metadata_rows.append({"pid": pid, "rank": anchor["rank"], "file": name,
                            "role": role(name, manifest), "opens": values.get("POSIX_OPENS", ""),
                            "stats": values.get("POSIX_STATS", ""), "seeks": values.get("POSIX_SEEKS", ""),
                            "read_calls": actual["read_count"], "write_calls": actual["write_count"],
                            "read_bytes": actual["read_bytes"], "write_bytes": actual["write_bytes"],
                            "stdio_read_calls":values.get("STDIO_READS", ""),
                            "stdio_write_calls":values.get("STDIO_WRITES", ""),
                            "stdio_read_bytes":values.get("STDIO_BYTES_READ", ""),
                            "stdio_write_bytes":values.get("STDIO_BYTES_WRITTEN", ""),
                            "metadata_time_s": values.get("POSIX_F_META_TIME", "")})
        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        connection.executescript("CREATE INDEX event_phase ON events(phase,op,length); CREATE INDEX event_time ON events(start,end); CREATE INDEX event_stream ON events(pid,file,op,thread,start);")
        connection.commit()
        for rank in range(8):
            if connection.execute("SELECT coalesce(sum(length),0) FROM events WHERE rank=? AND role='dataset' AND op='read'", (rank,)).fetchone()[0] == 0:
                raise ValueError(f"Rank {rank}: no dataset-read coverage in rank or reader children")
            if not manifest.get("smoke") and connection.execute("SELECT count(*) FROM events WHERE rank=? AND role='checkpoint' AND op='write'", (rank,)).fetchone()[0] == 0:
                raise ValueError(f"Rank {rank}: no checkpoint-write coverage")
        if manifest.get("smoke") and not all(any(a["role"] == "worker" and a["rank"] == rank for a in anchors) for rank in range(8)):
            raise ValueError("Preflight did not capture a forked reader for every rank")
        if manifest.get("smoke"):
            for anchor in anchors:
                if anchor["role"] == "worker" and connection.execute("SELECT coalesce(sum(length),0) FROM events WHERE pid=? AND op='read'", (anchor["pid"],)).fetchone()[0] == 0:
                    raise ValueError("Forked preflight process has no read coverage")
        sizes_path = case / "checkpoint-sizes.csv"
        if sizes_path.exists():
            saved = Counter()
            with sizes_path.open() as file:
                for item in csv.DictReader(file):
                    saved[int(item["rank"])] += int(item["bytes"])
            for rank,size in saved.items():
                traced = connection.execute("SELECT coalesce(sum(length),0) FROM events WHERE rank=? AND role='checkpoint' AND op='write'", (rank,)).fetchone()[0]
                if traced < .9*size:
                    raise ValueError(f"Rank {rank}: checkpoint trace bytes substantially below saved file sizes")
        status = case / "telemetry/status.json"
        if status.exists() and json.loads(status.read_text())["errors"]:
            report["limitations"].append("Some passive telemetry is incomplete; inspect telemetry/status.json")
        for path in (case / "telemetry").glob("client.jsonl"):
            with path.open() as file:
                for line in file:
                    # A sampler can catch a child before its startup hook writes
                    # the anchor. Check against final coverage, not that instant.
                    if any(process["pid"] not in logs for process in json.loads(line).get("processes", [])):
                        raise ValueError("Sampler observed descendant processes without instrumentation anchors")
        report.update(valid=True, processes=len(anchors), rank_processes=len(roots),
                      worker_processes=len(anchors)-len(roots), clock_origin_wall_s=origin,
                      phases=phases, checkpoints=checkpoints, **timings)
        report["outside_phase_calls"] = connection.execute("SELECT count(*) FROM events WHERE phase='outside'").fetchone()[0]
        if report["outside_phase_calls"]:
            raise ValueError("Workload calls fall outside the clock-aligned phase partition")
        table(output / "files.csv", metadata_rows)
        phase_tables(connection, phases, output)
        pattern_tables(connection, output)
        dump(output / "coverage.json", report)
        return report
    except BaseException as error:
        report["valid"] = False
        report["errors"].append(str(error))
        dump(output / "coverage.json", report)
        for name in ("phase_access.png","phase_access.svg","overview.html","detail.html"):
            (output/name).unlink(missing_ok=True)
        raise
    finally:
        connection.close()


def phase_tables(db, phases, output):
    rows = []
    for phase in phases:
        row = {**phase, "duration_s": phase["end_s"]-phase["start_s"]}
        for op in ("read", "write"):
            count, size, crossing, files, ranks, processes, zeros = db.execute("SELECT count(*),coalesce(sum(length),0),coalesce(sum(crossing),0),count(distinct file),count(distinct rank),count(distinct pid),coalesce(sum(length=0),0) FROM events WHERE phase=? AND op=?", (phase["phase"],op)).fetchone()
            row.update({op+"_calls": count, op+"_bytes": size, op+"_crossing_calls": crossing, op+"_files": files,
                        op+"_ranks":ranks, op+"_processes":processes, op+"_zero_length_calls":zeros})
            row[op+"_API_MiB_s_wall"] = size/row["duration_s"]/1048576
            row[op+"_calls_s_wall"] = count/row["duration_s"]
            for name, fraction in (("p50", .5), ("p95", .95), ("max", 1)):
                value = db.execute("SELECT length FROM events WHERE phase=? AND op=? ORDER BY length LIMIT 1 OFFSET ?", (phase["phase"],op,max(0,math.ceil(count*fraction)-1))).fetchone()
                row[op+"_"+name+"_bytes"] = value[0] if value else ""
            modal = db.execute("SELECT length FROM events WHERE phase=? AND op=? GROUP BY length ORDER BY count(*) DESC,length LIMIT 1", (phase["phase"],op)).fetchone()
            row[op+"_modal_bytes"] = modal[0] if modal else ""
            spans = union(db.execute("SELECT max(start,?),min(end,?) FROM events WHERE op=? AND end>? AND start<? ORDER BY start", (phase["start_s"],phase["end_s"],op,phase["start_s"],phase["end_s"])))
            row[op+"_active_union_s"] = sum(b-a for a,b in spans)
        rows.append(row)
    table(output / "phases.csv", rows)


def pattern_tables(db, output):
    sharing = [{"file":name,"role":kind,"ranks":ranks,"processes":processes,"read_bytes":reads,"write_bytes":writes}
               for name,kind,ranks,processes,reads,writes in db.execute("SELECT file,role,count(distinct rank),count(distinct pid),sum(CASE WHEN op='read' THEN length ELSE 0 END),sum(CASE WHEN op='write' THEN length ELSE 0 END) FROM events GROUP BY file,role")]
    table(output/"file-sharing.csv",sharing)
    rows, previous, current = [], None, None
    for pid,rank,name,op,thread,start,offset,length in db.execute("SELECT pid,rank,file,op,thread,start,offset,length FROM events WHERE length>0 ORDER BY pid,file,op,thread,start,end"):
        key = (pid,rank,name,op,thread)
        if key != current:
            if previous is not None:
                rows.append(previous)
            current = key
            previous = {"pid":pid,"rank":rank,"file":name,"op":op,"thread":thread,"calls":0,
                        "contiguous_transitions":0,"forward_gap_transitions":0,"backward_or_overlap_transitions":0,
                        "stream_identity":"thread" if thread != "N/A" else "unknown-thread-interleaving-possible"}
            last_end = None
        if last_end is not None:
            field = "contiguous_transitions" if offset == last_end else "forward_gap_transitions" if offset > last_end else "backward_or_overlap_transitions"
            previous[field] += 1
        previous["calls"] += 1
        last_end = offset+length
    if previous is not None:
        rows.append(previous)
    table(output/"streams.csv",rows)


def recover_run(run):
    """Validate finished attempts in place; never rerun workloads or move pools."""
    path = run/"state.json"
    state = json.loads(path.read_text())
    if state["status"] == "running":
        raise ValueError("Stop the profiling runner before recovering its state")
    recovered = []
    for key,progress in state["cases"].items():
        if progress["status"] == "complete":
            continue
        case = run/progress["attempt"]
        required = [case/name for name in ("raw/summary.json","raw/per_epoch_stats.json",
                                          "raw/.hydra/config.yaml","checkpoint-sizes.csv")]
        required.extend(case/f"raw/{rank}_output.json" for rank in range(8))
        if not all(item.is_file() for item in required):
            print(f"{key}: native evidence incomplete; invocation still needs restarting")
            continue
        analyze_case(case)
        progress["status"] = "complete"
        progress["recovered_validation"] = True
        state["status"] = "interrupted"  # Controller must still verify/restore pools.
        temporary = path.with_suffix(".tmp")
        dump(temporary,state)
        temporary.replace(path)
        recovered.append(key)
        print(f"{key}: finished trace validated in place; --continue will skip it")
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A profile run directory, or one case attempt")
    parser.add_argument("--bin-seconds", type=float, default=5)
    parser.add_argument("--detail", action="store_true", help="Write exact-event interactive detail instead of overview plots")
    parser.add_argument("--recover",action="store_true",help="Revalidate finished failed/interrupted attempts and mark passed cases complete")
    parser.add_argument("--validate-only",action="store_true",help="Validate/recover without rendering charts")
    parser.add_argument("--rank", type=int, choices=range(8))
    parser.add_argument("--file", help="Exact file path for a detail view")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    if args.bin_seconds <= 0:
        parser.error("--bin-seconds must be positive")
    if args.recover:
        if not (args.run/"state.json").is_file():
            parser.error("--recover requires a study directory with state.json")
        recover_run(args.run)
    if (args.run / "case.json").exists():
        cases = [args.run]
    else:
        state = json.loads((args.run / "state.json").read_text())
        cases = [args.run / item["attempt"] for item in state["cases"].values() if item["status"] == "complete"]
    if not cases:
        parser.error("No completed profiling cases")
    for case in cases:
        if not args.recover:
            analyze_case(case)
    if args.validate_only:
        print(f"Validated {len(cases)} completed case(s); no workload execution or pool changes")
        return
    from plots import render, comparison, detail
    if args.detail:
        if len(cases) != 1:
            parser.error("--detail requires one case attempt")
        detail(cases[0], args.rank, args.file, args.start, args.end)
    else:
        render(cases, args.bin_seconds)
        comparison(cases, args.run / "analysis")
    print(f"Validated and plotted {len(cases)} profiling case(s)")


if __name__ == "__main__":
    main()
