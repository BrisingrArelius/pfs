#!/usr/bin/env python3
"""Validate native iperf3 evidence and create transport summaries."""

from __future__ import annotations

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

from run_iperf3 import (fingerprints, plan_units, validate_config,
                        validate_inventory, validate_member_artifacts)


MEASUREMENT_FIELDS = [
    "run_id", "run_mode", "unit_id", "unit_mode", "repetition", "attempt",
    "client", "server", "direction", "streams", "port", "source_address",
    "source_interface", "destination_address", "destination_interface",
    "sender_bits_per_second", "receiver_bits_per_second",
    "sender_gbits_per_second", "receiver_gbits_per_second", "receiver_mib_per_second",
    "sender_bytes", "receiver_bytes", "sender_seconds", "receiver_seconds",
    "retransmits", "client_cpu_percent", "server_cpu_percent", "link_mbps",
    "line_rate_percent", "launch_skew_seconds", "artifact",
]

EPOCH_FIELDS = [
    "run_id", "unit_id", "unit_mode", "repetition", "attempt", "path_sessions",
    "aggregate_receiver_gbits_per_second", "aggregate_receiver_mib_per_second",
    "launch_skew_seconds", "artifact",
]

SUMMARY_FIELDS = [
    "unit_mode", "client", "server", "direction", "streams", "runs",
    "receiver_gbits_per_second_median", "receiver_gbits_per_second_min",
    "receiver_gbits_per_second_max", "receiver_mib_per_second_median",
    "retransmits_median", "line_rate_percent_median",
]

COUNTER_FIELDS = [
    "run_id", "unit_id", "unit_mode", "repetition", "attempt", "host", "interface",
    "rx_bytes", "rx_packets", "rx_errors", "rx_dropped", "tx_bytes", "tx_packets",
    "tx_errors", "tx_dropped",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run directory containing manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_csv(path, fields, rows):
    from io import StringIO
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def numeric_or_none(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("metric is not finite")
    return value


def interface_counters(path, expected_interface):
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"unexpected interface telemetry: {path}")
    if payload[0].get("ifname") != expected_interface:
        raise ValueError(f"telemetry interface differs: {path}")
    stats = payload[0].get("stats64") or payload[0].get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"missing interface counters: {path}")
    result = {}
    for direction in ("rx", "tx"):
        values = stats.get(direction, {})
        for field in ("bytes", "packets", "errors", "dropped"):
            value = values.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {direction}_{field} counter: {path}")
            result[f"{direction}_{field}"] = value
    return result


def parse_manifest(root, manifest, report):
    config, inventory = manifest["config"], manifest["inventory"]
    measurements, epochs, counters = [], [], []
    for unit in manifest["units"]:
        attempts = unit.get("attempts", [])
        if (not attempts or attempts[-1].get("state") != "completed"
                or attempts[-1].get("cleanup") != "completed"):
            report["errors"].append(f"{unit['id']}: latest attempt or cleanup is not completed")
            continue
        attempt = attempts[-1]
        if len(attempt.get("members", [])) != len(unit["members"]):
            report["errors"].append(f"{unit['id']}: member count differs")
            continue
        attempt_rows = []
        attempt_counters = []
        try:
            artifact = Path(attempt["artifacts"])
            attempt_folder = (root / artifact).resolve()
            if not attempt_folder.is_relative_to(root.resolve()):
                raise ValueError("attempt artifact escapes run directory")
            expected_endpoints = {(member["client"], member["source_interface"])
                                  for member in unit["members"]}
            expected_endpoints |= {(member["server"], member["destination_interface"])
                                   for member in unit["members"]}
            for host, interface in expected_endpoints:
                telemetry_paths = {phase: attempt_folder / f"telemetry-{phase}-{host}-{interface}.json"
                                   for phase in ("before", "after")}
                if any(not path.is_file() or path.is_symlink()
                       for path in telemetry_paths.values()):
                    raise ValueError(f"missing telemetry for {host}")
                before = interface_counters(telemetry_paths["before"], interface)
                after = interface_counters(telemetry_paths["after"], interface)
                delta = {field: after[field] - before[field] for field in before}
                if any(value < 0 for value in delta.values()):
                    raise ValueError(f"interface counters decreased for {host}")
                attempt_counters.append({
                    "run_id": manifest["run_id"], "unit_id": unit["id"],
                    "unit_mode": unit["mode"], "repetition": unit["repetition"],
                    "attempt": attempt["id"], "host": host,
                    "interface": interface, **delta,
                })
            for saved, expected in zip(attempt["members"], unit["members"]):
                if saved["member"] != expected:
                    raise ValueError(f"saved member differs for {expected['id']}")
                summary = validate_member_artifacts(root, saved["artifacts"], expected, config)
                link_mbps = expected["path_link_mbps"]
                receiver_bps = summary["receiver_bits_per_second"]
                row = {
                    "run_id": manifest["run_id"],
                    "run_mode": manifest["mode"],
                    "unit_id": unit["id"],
                    "unit_mode": unit["mode"],
                    "repetition": unit["repetition"],
                    "attempt": attempt["id"],
                    **expected,
                    **summary,
                    "sender_gbits_per_second": summary["sender_bits_per_second"] / 1e9,
                    "receiver_gbits_per_second": receiver_bps / 1e9,
                    "receiver_mib_per_second": receiver_bps / (8 * 2**20),
                    "link_mbps": link_mbps,
                    "line_rate_percent": receiver_bps / (link_mbps * 1e6) * 100,
                    "launch_skew_seconds": attempt["launch_skew_seconds"],
                    "artifact": saved["artifacts"],
                }
                for field in ("retransmits", "client_cpu_percent", "server_cpu_percent"):
                    row[field] = numeric_or_none(row[field])
                attempt_rows.append(row)
            measurements.extend(attempt_rows)
            counters.extend(attempt_counters)
            total_bps = sum(row["receiver_bits_per_second"] for row in attempt_rows)
            epochs.append({
                "run_id": manifest["run_id"], "unit_id": unit["id"],
                "unit_mode": unit["mode"], "repetition": unit["repetition"],
                "attempt": attempt["id"], "path_sessions": len(attempt_rows),
                "aggregate_receiver_gbits_per_second": total_bps / 1e9,
                "aggregate_receiver_mib_per_second": total_bps / (8 * 2**20),
                "launch_skew_seconds": attempt["launch_skew_seconds"],
                "artifact": attempt["artifacts"],
            })
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            report["errors"].append(f"{unit['id']}: {error}")
    return measurements, epochs, counters


def validate_manifest_plan(manifest):
    config, inventory = manifest["config"], manifest["inventory"]
    validate_config(config)
    validate_inventory(inventory)
    if manifest.get("mode") not in {"pilot", "full"}:
        raise ValueError("manifest mode is invalid")
    fingerprint, inventory_fingerprint = fingerprints(
        config, inventory, manifest["tool_observations"])
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("manifest scientific fingerprint differs")
    if manifest.get("inventory_fingerprint") != inventory_fingerprint:
        raise ValueError("manifest inventory fingerprint differs")
    expected = plan_units(config, inventory, manifest["mode"] == "pilot")
    actual = manifest.get("units", [])
    if len(actual) != len(expected):
        raise ValueError("manifest unit count differs from the canonical plan")
    for saved, canonical in zip(actual, expected):
        core = {key: saved.get(key) for key in ("id", "mode", "repetition", "members")}
        canonical_core = {key: canonical[key] for key in core}
        if core != canonical_core:
            raise ValueError(f"manifest unit differs from canonical plan: {canonical['id']}")


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["unit_mode"], row["client"], row["server"],
               row["direction"], row["streams"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        gbps = [row["receiver_gbits_per_second"] for row in values]
        retransmits = [row["retransmits"] for row in values if row["retransmits"] is not None]
        line_rate = [row["line_rate_percent"] for row in values]
        result.append({
            "unit_mode": key[0], "client": key[1], "server": key[2],
            "direction": key[3], "streams": key[4], "runs": len(values),
            "receiver_gbits_per_second_median": statistics.median(gbps),
            "receiver_gbits_per_second_min": min(gbps),
            "receiver_gbits_per_second_max": max(gbps),
            "receiver_mib_per_second_median": statistics.median(
                row["receiver_mib_per_second"] for row in values),
            "retransmits_median": statistics.median(retransmits) if retransmits else "",
            "line_rate_percent_median": statistics.median(line_rate),
        })
    return result


def summary_markdown(summary, epochs, report):
    lines = [
        "# iperf3 network transport results", "",
        f"Validation: **{'PASS' if not report['errors'] else 'FAIL'}**",
        f"Parsed path sessions: **{report['measurements']}**",
        f"Parsed restartable units: **{report['epochs']}**",
    ]
    if report["errors"]:
        lines.extend(["", "## Validation errors"])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.extend(["", "## Per-path results", "",
                  "| Mode | Client | OSS | Direction | Streams | Runs | Median Gbit/s | Range Gbit/s | Median MiB/s | Median retransmits | Line rate |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary:
        retransmits = row["retransmits_median"]
        retransmits = "n/a" if retransmits == "" else f"{retransmits:g}"
        lines.append(
            f"| {row['unit_mode']} | {row['client']} | {row['server']} | {row['direction']} | "
            f"{row['streams']} | {row['runs']} | {row['receiver_gbits_per_second_median']:.3f} | "
            f"{row['receiver_gbits_per_second_min']:.3f}-{row['receiver_gbits_per_second_max']:.3f} | "
            f"{row['receiver_mib_per_second_median']:.1f} | {retransmits} | "
            f"{row['line_rate_percent_median']:.1f}% |"
        )
    concurrent = [row for row in epochs if row["unit_mode"] != "isolated"]
    if concurrent:
        lines.extend(["", "## Concurrent epochs", "",
                      "| Unit | Mode | Paths | Aggregate Gbit/s | Launch skew s |",
                      "|---|---|---:|---:|---:|"])
        for row in concurrent:
            lines.append(f"| `{row['unit_id']}` | {row['unit_mode']} | {row['path_sessions']} | "
                         f"{row['aggregate_receiver_gbits_per_second']:.3f} | "
                         f"{row['launch_skew_seconds']:.3f} |")
    return "\n".join(lines) + "\n"


def configuration_markdown(manifest):
    config, inventory = manifest["config"], manifest["inventory"]
    lines = [
        "# iperf3 run configuration", "",
        "> Derived from `manifest.json`; the manifest and native JSON are authoritative.", "",
        "## Protocol", "", f"- Protocol version: `{config['protocol_version']}`",
        f"- Transport: `{config['transport']}`",
        f"- Timing: {config['omit_seconds']} s omitted + {config['duration_seconds']} s measured",
        f"- Isolated streams: `{config['isolated_streams']}`",
        f"- Concurrent streams: `{config['concurrent_streams']}`",
        f"- Repetitions: **{1 if manifest['mode'] == 'pilot' else config['repetitions']}**",
        f"- Inventory fingerprint: `{manifest['inventory_fingerprint']}`", "",
        "## Fixed paths", "",
        "| Client | OSS | Source endpoint | Destination endpoint | Evidence |",
        "|---|---|---|---|---|",
    ]
    for client in sorted(inventory["paths"]):
        for server, path in sorted(inventory["paths"][client].items()):
            lines.append(f"| {client} | {server} | `{path['source_address']}` "
                         f"(`{path['source_interface']}`) | `{path['destination_address']}` "
                         f"(`{path['destination_interface']}`) | {path['evidence_level']} |")
    lines.extend(["", "## Tool versions", "", "| Host | iperf3 |", "|---|---|"])
    for host, item in sorted(manifest["tool_observations"].items()):
        lines.append(f"| {host} | `{item['iperf3_version']}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    try:
        root = args.input.resolve()
        output = args.output_dir.resolve(strict=False)
        if output == root or not output.is_relative_to(root):
            raise ValueError("output directory must be a dedicated child of the run directory")
        manifest = json.loads((root / "manifest.json").read_text())
        report = {"generated_at": datetime.now(timezone.utc).isoformat(), "errors": []}
        validate_manifest_plan(manifest)
        measurements, epochs, counters = parse_manifest(root, manifest, report)
        summary = summarize(measurements)
        report.update(manifest=str(root / "manifest.json"), measurements=len(measurements),
                      epochs=len(epochs), counter_rows=len(counters), summary_rows=len(summary),
                      expected_units=len(manifest["units"]),
                      expected_measurements=sum(len(unit["members"]) for unit in manifest["units"]))
        if len(epochs) != report["expected_units"]:
            report["errors"].append("parsed restartable-unit count differs from the manifest plan")
        if len(measurements) != report["expected_measurements"]:
            report["errors"].append("parsed path-session count differs from the manifest plan")
        write_csv(output / "measurements.csv", MEASUREMENT_FIELDS, measurements)
        write_csv(output / "epochs.csv", EPOCH_FIELDS, epochs)
        write_csv(output / "interface_counters.csv", COUNTER_FIELDS, counters)
        write_csv(output / "summary.csv", SUMMARY_FIELDS, summary)
        atomic_text(output / "summary.md", summary_markdown(summary, epochs, report))
        atomic_text(output / "run_configuration.md", configuration_markdown(manifest))
        atomic_text(output / "parse_report.json", json.dumps(report, indent=2) + "\n")
        print(f"Parsed {len(measurements)} path sessions across {len(epochs)} units.")
        print(f"Summary: {output / 'summary.md'}")
        return 2 if report["errors"] else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot parse results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
