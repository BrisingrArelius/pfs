#!/usr/bin/env python3
"""Validate native local-FIO evidence and create read-only derived summaries."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


MEASUREMENT_FIELDS = [
    "run_id", "mode", "host", "target_id", "media", "device", "mount",
    "workload", "rw", "block_size", "jobs", "repetition", "session", "attempt",
    "completion_reason", "io_bytes", "gib", "fio_runtime_ms", "seconds",
    "bw_bytes_s", "bw_mib_s", "iops", "total_ios", "clat_mean_ns",
    "clat_p50_ns", "clat_p95_ns", "clat_p99_ns", "clat_p99_9_ns",
    "command_wall_seconds", "preparation_generation", "artifact",
]
PREPARATION_FIELDS = [
    "run_id", "host", "target_id", "media", "generation", "io_bytes",
    "fio_runtime_ms", "fio_write_mib_s", "command_wall_seconds", "sync_overhead_seconds",
    "artifact",
]
SUMMARY_FIELDS = [
    "host", "target_id", "media", "workload", "runs", "bw_mib_s_median",
    "bw_mib_s_min", "bw_mib_s_max", "iops_median", "iops_min", "iops_max",
    "clat_mean_ms_median", "clat_p99_ms_median", "byte_limit_runs", "time_limit_runs",
]


def parse_args(argv=None):
    """Require source run directories and a separate derived-output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Host result directories or parents containing them")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def atomic_text(path, text):
    """Publish a derived file atomically without modifying source evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_csv(path, fields, rows):
    """Serialize normalized dictionaries with a stable column order."""
    from io import StringIO
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def discover_manifests(inputs, output_dir):
    """Find unique manifests while excluding the derived-output directory."""
    output = output_dir.resolve(strict=False)
    manifests = set()
    for source in inputs:
        source = source.resolve()
        candidates = [source / "manifest.json"] if source.is_dir() else []
        if source.is_dir() and not candidates[0].is_file():
            candidates = source.rglob("manifest.json")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(output):
                continue
            manifests.add(resolved)
    if not manifests:
        raise ValueError("no manifest.json found in the supplied inputs")
    return sorted(manifests)


def safe_artifact(root, relative):
    """Resolve a recorded artifact without permitting traversal outside its run."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes run directory: {relative}")
    return path


def percentile(percentiles, wanted):
    """Read a FIO percentile whose JSON key contains decimal formatting."""
    for key, value in percentiles.items():
        if math.isclose(float(key), wanted):
            return value
    raise ValueError(f"missing p{wanted:g} completion latency")


def latency_ns(stats):
    """Normalize FIO completion-latency units to nanoseconds."""
    for name, scale in (("clat_ns", 1), ("clat_us", 1000), ("clat_ms", 1_000_000)):
        if name in stats:
            clat = stats[name]
            values = clat.get("percentile", {})
            return {
                "clat_mean_ns": clat["mean"] * scale,
                "clat_p50_ns": percentile(values, 50) * scale,
                "clat_p95_ns": percentile(values, 95) * scale,
                "clat_p99_ns": percentile(values, 99) * scale,
                "clat_p99_9_ns": percentile(values, 99.9) * scale,
            }
    raise ValueError("missing completion-latency statistics")


def parse_native(path, expected_names, operation, size, runtime, time_based=False):
    """Validate native jobs and return one aggregate target-level measurement."""
    result = json.loads(path.read_text())
    jobs = result.get("jobs", [])
    expected_names = [expected_names] if isinstance(expected_names, str) else expected_names
    if (len(jobs) != len(expected_names)
            or [job.get("jobname") for job in jobs] != expected_names
            or any(job.get("error") != 0 for job in jobs)):
        raise ValueError("unexpected or failed named FIO jobs")
    opposite = "write" if operation == "read" else "read"
    if any(job.get(opposite, {}).get("io_bytes", 0) for job in jobs):
        raise ValueError(f"unexpected {opposite} bytes")
    stats = [job[operation] for job in jobs]
    if any(item["io_bytes"] <= 0 or item["runtime"] <= 0 or item["total_ios"] <= 0 for item in stats):
        raise ValueError("invalid byte, runtime or operation accounting")
    io_bytes = sum(item["io_bytes"] for item in stats)
    elapsed = max(item["runtime"] for item in stats)
    total_ios = sum(item["total_ios"] for item in stats)
    if time_based:
        if any(item["runtime"] < runtime * 1000 - 1000 for item in stats):
            raise ValueError("job stopped before the runtime limit")
        reason = "time_limit"
    else:
        if io_bytes > size:
            raise ValueError("job exceeded byte limit")
        reason = "byte_limit" if io_bytes == size else "time_limit"
        if reason == "time_limit" and elapsed < runtime * 1000 - 1000:
            raise ValueError("job reached neither byte nor runtime limit")
    latencies = [latency_ns(item) for item in stats]
    weighted_mean = sum(value["clat_mean_ns"] * item["total_ios"]
                        for value, item in zip(latencies, stats)) / total_ios
    row = {
        "completion_reason": reason, "io_bytes": io_bytes, "gib": io_bytes / 2**30,
        "fio_runtime_ms": elapsed, "seconds": elapsed / 1000,
        "bw_bytes_s": sum(item["bw_bytes"] for item in stats),
        "bw_mib_s": sum(item["bw_bytes"] for item in stats) / 2**20,
        "iops": sum(item["iops"] for item in stats), "total_ios": total_ios,
        "jobs": len(jobs), "clat_mean_ns": weighted_mean,
    }
    for field in ("clat_p50_ns", "clat_p95_ns", "clat_p99_ns", "clat_p99_9_ns"):
        row[field] = max(value[field] for value in latencies)
    return row


def parse_manifest(manifest_path, report):
    """Parse one host manifest, recording errors without altering its evidence."""
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    inventory = {target["target_id"]: target for target in manifest["inventory"]}
    config = manifest["config"]
    measurements, preparations = [], []
    host_report = {"manifest": str(manifest_path), "host": manifest["host"],
                   "expected_cases": len(manifest["cases"]), "parsed_cases": 0,
                   "errors": [], "warnings": []}
    report["hosts"].append(host_report)
    for target_id, target_state in manifest["targets"].items():
        target = inventory[int(target_id)]
        if target_state.get("cleanup") != "completed":
            host_report["errors"].append(f"target {target_id}: cleanup is {target_state.get('cleanup')}")
        completed_preparations = [item for item in target_state["preparations"] if item["state"] == "completed"]
        for item in completed_preparations:
            try:
                artifact = safe_artifact(root, item["artifacts"])
                native = parse_native(artifact / "fio.json", f"target-{target_id}", "write",
                                       config["fio"]["size"] * config["fio"].get("numjobs", 1), 0)
                preparations.append({
                    "run_id": manifest["run_id"], "host": manifest["host"],
                    "target_id": int(target_id), "media": target["media"],
                    "generation": item["preparation_generation"], "io_bytes": native["io_bytes"],
                    "fio_runtime_ms": native["fio_runtime_ms"],
                    "fio_write_mib_s": native["bw_mib_s"],
                    "command_wall_seconds": item["command_wall_seconds"],
                    "sync_overhead_seconds": item["command_wall_seconds"] - native["seconds"],
                    "artifact": item["artifacts"],
                })
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                host_report["errors"].append(f"target {target_id} preparation: {error}")
        if not completed_preparations:
            host_report["errors"].append(f"target {target_id}: no completed preparation")
    for case in manifest["cases"]:
        try:
            attempts = case["attempts"]
            if not attempts or attempts[-1]["state"] != "completed":
                raise ValueError("latest attempt is not completed")
            attempt = attempts[-1]
            target = inventory[case["target_id"]]
            workload = case["workload"]
            artifact = safe_artifact(root, attempt["artifacts"])
            operation = "write" if "write" in workload["rw"] else "read"
            job_count = config["fio"].get("numjobs", 1)
            names = ([f"target-{case['target_id']}-job-{index + 1}" for index in range(job_count)]
                     if job_count > 1 else f"target-{case['target_id']}")
            row = parse_native(artifact / "fio.json", names, operation,
                               config["fio"]["size"] * job_count, config["fio"]["runtime"],
                               bool(config["fio"].get("time_based", 0)))
            if row["completion_reason"] != attempt["completion_reason"] or row["io_bytes"] != attempt["io_bytes"]:
                raise ValueError("manifest/native completion accounting differs")
            row.update({
                "run_id": manifest["run_id"], "mode": manifest.get("mode", "full"),
                "host": manifest["host"], "target_id": case["target_id"],
                "media": target["media"], "device": target["device"], "mount": target["mount"],
                "workload": workload["name"], "rw": workload["rw"],
                "block_size": workload["bs"], "repetition": case["repetition"],
                "session": attempt["session"], "attempt": attempt["id"],
                "command_wall_seconds": attempt["command_wall_seconds"],
                "preparation_generation": attempt["preparation_generation"],
                "artifact": attempt["artifacts"],
            })
            measurements.append(row)
            host_report["parsed_cases"] += 1
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            host_report["errors"].append(f"{case.get('id', 'unknown case')}: {error}")
    outcomes = [session.get("outcome") for session in manifest.get("sessions", [])]
    if not outcomes or outcomes[-1] != "completed":
        host_report["warnings"].append(f"latest session outcome is {outcomes[-1] if outcomes else 'missing'}")
    return measurements, preparations


def summarize(rows):
    """Aggregate repetitions without pooling different targets or workloads."""
    groups = {}
    for row in rows:
        key = (row["host"], row["target_id"], row["media"], row["workload"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, values in sorted(groups.items()):
        bandwidth = [row["bw_mib_s"] for row in values]
        iops = [row["iops"] for row in values]
        summary.append({
            "host": key[0], "target_id": key[1], "media": key[2], "workload": key[3],
            "runs": len(values), "bw_mib_s_median": statistics.median(bandwidth),
            "bw_mib_s_min": min(bandwidth), "bw_mib_s_max": max(bandwidth),
            "iops_median": statistics.median(iops), "iops_min": min(iops), "iops_max": max(iops),
            "clat_mean_ms_median": statistics.median(row["clat_mean_ns"] for row in values) / 1e6,
            "clat_p99_ms_median": statistics.median(row["clat_p99_ns"] for row in values) / 1e6,
            "byte_limit_runs": sum(row["completion_reason"] == "byte_limit" for row in values),
            "time_limit_runs": sum(row["completion_reason"] == "time_limit" for row in values),
        })
    return summary


def markdown(summary, report):
    """Render a compact human-readable summary with validation status first."""
    errors = sum(len(host["errors"]) for host in report["hosts"])
    lines = ["# Local-storage FIO results", "", f"Validation: **{'PASS' if not errors else 'FAIL'}**",
             f"Parsed measurements: **{report['measurements']}**", "",
             "| Host | Target | Media | Workload | Runs | Median MiB/s | Range MiB/s | Median IOPS | Mean clat ms | p99 clat ms | Stops |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        stops = f"{row['byte_limit_runs']} byte / {row['time_limit_runs']} time"
        lines.append(f"| {row['host']} | {row['target_id']} | {row['media']} | {row['workload']} | {row['runs']} | "
                     f"{row['bw_mib_s_median']:.1f} | {row['bw_mib_s_min']:.1f}–{row['bw_mib_s_max']:.1f} | "
                     f"{row['iops_median']:.0f} | {row['clat_mean_ms_median']:.3f} | "
                     f"{row['clat_p99_ms_median']:.3f} | {stops} |")
    for host in report["hosts"]:
        if host["errors"] or host["warnings"]:
            lines.extend(["", f"## {host['host']} validation"])
            lines.extend(f"- ERROR: {message}" for message in host["errors"])
            lines.extend(f"- WARNING: {message}" for message in host["warnings"])
    return "\n".join(lines) + "\n"


def configuration_markdown(manifest_paths):
    """Describe a copied multi-host run without requiring manifest inspection."""
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    first = manifests[0]
    config, fio = first["config"], first["config"]["fio"]
    job_count = fio.get("numjobs", 1)
    lines = ["# FIO run configuration", "",
             "> Derived from host manifests; `manifest.json` and native FIO output remain authoritative.", "",
             "## Hosts", "", "| Host | Run ID | Mode | Protocol | FIO |", "|---|---|---|---:|---|"]
    lines.extend(f"| {item['host']} | `{item['run_id']}` | {item.get('mode', 'full')} | "
                 f"{item['config'].get('protocol_version', 'unknown')} | `{item.get('fio_version', 'unknown')}` |"
                 for item in manifests)
    lines.extend(["", "## Measurement protocol", "",
                  f"- Jobs per OST: **{job_count}**",
                  f"- Per-job size/region: **{fio['size'] / 2**30:g} GiB**",
                  f"- Prepared file per OST: **{fio['size'] * job_count / 2**30:g} GiB**",
                  f"- Engine/direct I/O: `{fio.get('ioengine', 'unknown')}`, `direct={fio.get('direct', 'unknown')}`",
                  f"- Queue depth: **{fio.get('iodepth', 'unknown')} per job**",
                  f"- Timing: **{fio.get('ramp_time', 0)} s ramp + {fio['runtime']} s runtime**, "
                  f"`time_based={fio.get('time_based', 0)}`",
                  f"- Repetitions: **{config['repetitions']}**", "",
                  "## Workloads", "", "| Name | Pattern | Block size |", "|---|---|---:|"])
    lines.extend(f"| `{item['name']}` | `{item['rw']}` | `{item['bs']}` |"
                 for item in config["workloads"])
    lines.extend(["", "For multi-job protocol 5 runs, bandwidth and IOPS are summed across jobs. ",
                  "Mean completion latency is operation-weighted; percentile columns use the worst per-job percentile.", ""])
    return "\n".join(lines)


def main(argv=None):
    """Parse all hosts, publish derived artifacts and fail if evidence is incomplete."""
    args = parse_args(argv)
    try:
        manifests = discover_manifests(args.inputs, args.output_dir)
        report = {"generated_at": datetime.now(timezone.utc).isoformat(), "hosts": []}
        measurements, preparations = [], []
        for manifest in manifests:
            host_measurements, host_preparations = parse_manifest(manifest, report)
            measurements.extend(host_measurements)
            preparations.extend(host_preparations)
        summary = summarize(measurements)
        report.update(manifests=[str(path) for path in manifests], measurements=len(measurements),
                      preparations=len(preparations), summary_rows=len(summary))
        write_csv(args.output_dir / "measurements.csv", MEASUREMENT_FIELDS, measurements)
        write_csv(args.output_dir / "preparations.csv", PREPARATION_FIELDS, preparations)
        write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary)
        atomic_text(args.output_dir / "summary.md", markdown(summary, report))
        atomic_text(args.output_dir / "run_configuration.md", configuration_markdown(manifests))
        atomic_text(args.output_dir / "parse_report.json", json.dumps(report, indent=2) + "\n")
        errors = sum(len(host["errors"]) for host in report["hosts"])
        print(f"Parsed {len(measurements)} measurements from {len(manifests)} host manifest(s).")
        print(f"Summary: {args.output_dir / 'summary.md'}")
        return 2 if errors else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot parse results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
