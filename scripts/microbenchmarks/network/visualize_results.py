#!/usr/bin/env python3
"""Validate raw iperf3 run evidence and plot transport performance directly."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import uuid

from run_iperf3 import (fingerprints, plan_units, validate_config,
                        validate_inventory, revalidate_attempt_evidence)
from run_support import safe_artifact

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as error:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error


DIRECTIONS = ("client_to_oss", "oss_to_client")
DIRECTION_LABELS = {"client_to_oss": "Client to OSS", "oss_to_client": "OSS to client"}
DIRECTION_COLORS = {"client_to_oss": "#167D91", "oss_to_client": "#C66A32"}
STREAM_COLORS = {1: "#315A7D", 4: "#D08B32"}
MODE_LABELS = {
    "one_client_four_oss": "1 client / 4 OSS",
    "two_clients_four_oss": "2 clients / 4 OSS",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory with manifest.json and raw/ attempts")
    parser.add_argument("--output-dir", type=Path, help="defaults to <run>/plots")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.run / "plots"
    return args


def load_raw(run):
    """Reject truncated plans, incomplete units and corrupt native path sessions."""
    if not run.is_dir() or run.is_symlink():
        raise ValueError("raw iperf3 run directory is missing or is a symlink")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config, inventory = manifest["config"], manifest["inventory"]
    validate_config(config)
    validate_inventory(inventory)
    if manifest.get("mode") not in ("pilot", "full"):
        raise ValueError("invalid network run mode")
    scientific, inventory_hash = fingerprints(config, inventory, manifest["tool_observations"])
    if manifest.get("fingerprint") != scientific or manifest.get("inventory_fingerprint") != inventory_hash:
        raise ValueError("network scientific or inventory fingerprint differs")
    planned = plan_units(config, inventory, manifest["mode"] == "pilot")
    if len(manifest["units"]) != len(planned):
        raise ValueError("manifest unit count differs from the canonical plan")
    measurements, epochs = [], []
    for unit, canonical in zip(manifest["units"], planned):
        if any(unit.get(key) != canonical[key] for key in ("id", "mode", "repetition", "members")):
            raise ValueError("manifest unit differs from the canonical plan")
        attempts = unit.get("attempts", [])
        if not attempts or attempts[-1].get("state") != "completed" or attempts[-1].get("cleanup") != "completed":
            raise ValueError(f"{unit['id']}: measurement or cleanup incomplete")
        attempt = attempts[-1]
        if len(attempt.get("members", [])) != len(unit["members"]):
            raise ValueError(f"{unit['id']}: path-session count differs")
        for clock in attempt["clock_observations"].values():
            offset, uncertainty = clock["offset_seconds"], clock["uncertainty_seconds"]
            if (not math.isfinite(offset) or not math.isfinite(uncertainty)
                    or abs(offset) > config["validation"]["maximum_clock_offset_seconds"]
                    or uncertainty < 0
                    or uncertainty > config["validation"]["maximum_clock_uncertainty_seconds"]):
                raise ValueError(f"{unit['id']}: saved clock synchronization differs")
        summaries, skew, lateness = revalidate_attempt_evidence(run, unit, attempt, config)
        if (abs(skew - attempt["launch_skew_seconds"]) > .01
                or abs(lateness - attempt["maximum_release_lateness_seconds"]) > .01):
            raise ValueError(f"{unit['id']}: saved epoch synchronization differs")
        folder = safe_artifact(run, attempt["artifacts"])
        endpoints = {(member["client"], member["source_interface"]) for member in unit["members"]}
        endpoints |= {(member["server"], member["destination_interface"]) for member in unit["members"]}
        for host, interface in endpoints:
            snapshots = []
            for phase in ("before", "after"):
                telemetry = folder / f"telemetry-{phase}-{host}-{interface}.json"
                if not telemetry.is_file() or telemetry.is_symlink():
                    raise ValueError(f"{unit['id']}: missing {phase} interface counters")
                counters = json.loads(telemetry.read_text(encoding="utf-8"))
                if (not isinstance(counters, list) or len(counters) != 1
                        or counters[0].get("ifname") != interface):
                    raise ValueError(f"{unit['id']}: unexpected interface counter evidence")
                stats = counters[0].get("stats64") or counters[0].get("stats")
                if not isinstance(stats, dict):
                    raise ValueError(f"{unit['id']}: missing interface counters")
                values = []
                for direction in ("rx", "tx"):
                    for field in ("bytes", "packets", "errors", "dropped"):
                        value = stats.get(direction, {}).get(field)
                        if type(value) is not int or value < 0:
                            raise ValueError(f"{unit['id']}: invalid interface counter {field}")
                        values.append(value)
                snapshots.append(values)
            if any(after < before for before, after in zip(*snapshots)):
                raise ValueError(f"{unit['id']}: interface counters decreased")
        unit_rows = []
        for saved, expected, native in zip(attempt["members"], unit["members"], summaries):
            if saved.get("member") != expected:
                raise ValueError(f"{unit['id']}: saved path differs from the plan")
            delivered = native["receiver_bits_per_second"]
            if (not math.isfinite(delivered) or delivered <= 0
                    or native["receiver_seconds"] <= 0
                    or abs(delivered - native["receiver_bytes"] * 8 / native["receiver_seconds"])
                    > max(1000, delivered * .02)):
                raise ValueError(f"{unit['id']}: receiver rate contradicts delivered bytes and time")
            unit_rows.append({**expected, **native, "unit_mode": unit["mode"],
                              "repetition": unit["repetition"],
                              "receiver_gbits_per_second": delivered / 1e9,
                              "unit_id": unit["id"]})
        measurements.extend(unit_rows)
        directions = {row["direction"] for row in unit_rows}
        if len(directions) != 1:
            raise ValueError("epoch contains mixed directions")
        epochs.append({"unit_id": unit["id"], "unit_mode": unit["mode"],
                       "direction": directions.pop(), "repetition": unit["repetition"],
                       "aggregate_receiver_gbits_per_second": sum(
                           row["receiver_gbits_per_second"] for row in unit_rows)})
    return measurements, epochs


def grouped(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        result.setdefault(key, []).append(row)
    return result


def style(axis, ylabel):
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, path):
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("plot output may not follow a symlink")
    if path.exists():
        manifest = path.parent / "plot_manifest.json"
        if (manifest.is_symlink() or not manifest.is_file()
                or path.name not in json.loads(manifest.read_text()).get("plots", [])):
            raise ValueError("existing network plot is not owned by this visualizer")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as target:
            figure.savefig(target, format="png", dpi=220, bbox_inches="tight",
                           facecolor="#FCFCFA")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def plot_isolated(rows, output_dir):
    selected = [row for row in rows if row["unit_mode"] == "isolated"]
    if not selected:
        return None
    groups = grouped(selected, ("direction", "client", "server", "streams"))
    paths = sorted({(row["client"], row["server"]) for row in selected})
    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(paths) * 1.25), 10), sharex=True)
    width = 0.34
    for axis, direction in zip(axes, DIRECTIONS):
        for stream_index, streams in enumerate((1, 4)):
            offset = (stream_index - 0.5) * width
            for position, (client, server) in enumerate(paths):
                values = [row["receiver_gbits_per_second"]
                          for row in groups.get((direction, client, server, streams), [])]
                if not values:
                    continue
                median = statistics.median(values)
                axis.bar(position + offset, median, width=width * 0.9,
                         color=STREAM_COLORS[streams], edgecolor="#26343D", linewidth=0.5)
                axis.errorbar(position + offset, median,
                              yerr=[[median - min(values)], [max(values) - median]],
                              fmt="none", color="#26343D", capsize=3, linewidth=1)
                axis.scatter([position + offset] * len(values), values, marker="_",
                             color="#17242C", s=45, linewidths=0.9, zorder=4)
        axis.set_title(DIRECTION_LABELS[direction], loc="left", fontweight="bold")
        style(axis, "Receiver throughput (Gbit/s)")
    axes[-1].set_xticks(range(len(paths)), [f"{client} to {server}" for client, server in paths],
                       rotation=35, ha="right")
    axes[-1].set_xlabel("Fixed client/OSS path")
    figure.legend(handles=[Patch(facecolor=STREAM_COLORS[value], label=f"{value} stream(s)")
                           for value in (1, 4)], loc="upper center", bbox_to_anchor=(.5, .968),
                  ncol=2, frameon=False)
    figure.suptitle("Isolated TCP throughput by path", fontsize=15, fontweight="bold", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .91))
    relative = "isolated_throughput.png"
    save(figure, output_dir / relative)
    return relative


def plot_concurrent_epochs(rows, output_dir):
    selected = [row for row in rows if row["unit_mode"] in MODE_LABELS]
    if not selected:
        return None
    groups = grouped(selected, ("unit_mode", "direction"))
    modes = tuple(MODE_LABELS)
    figure, axis = plt.subplots(figsize=(10, 6.5))
    width = 0.34
    for direction_index, direction in enumerate(DIRECTIONS):
        offset = (direction_index - 0.5) * width
        for position, mode in enumerate(modes):
            values = [row["aggregate_receiver_gbits_per_second"]
                      for row in groups.get((mode, direction), [])]
            if not values:
                continue
            median = statistics.median(values)
            axis.bar(position + offset, median, width=width * 0.9,
                     color=DIRECTION_COLORS[direction], edgecolor="#26343D", linewidth=0.6)
            axis.errorbar(position + offset, median,
                          yerr=[[median - min(values)], [max(values) - median]],
                          fmt="none", color="#26343D", capsize=4, linewidth=1.1)
            axis.scatter([position + offset] * len(values), values, marker="_",
                         color="#17242C", s=55, linewidths=1, zorder=4)
    axis.set_xticks(range(len(modes)), [MODE_LABELS[mode] for mode in modes])
    axis.set_xlabel("Simultaneous topology")
    axis.set_title("Concurrent epoch aggregate throughput", loc="left", fontweight="bold")
    style(axis, "Aggregate receiver throughput (Gbit/s)")
    axis.legend(handles=[Patch(facecolor=DIRECTION_COLORS[value], label=DIRECTION_LABELS[value])
                         for value in DIRECTIONS], frameon=False)
    figure.tight_layout()
    relative = "concurrent_epoch_aggregate.png"
    save(figure, output_dir / relative)
    return relative


def plot_concurrent_sessions(rows, output_dir):
    selected = [row for row in rows if row["unit_mode"] in MODE_LABELS]
    if not selected:
        return None
    categories = [(mode, direction) for mode in MODE_LABELS for direction in DIRECTIONS]
    values = [[row["receiver_gbits_per_second"] for row in selected
               if row["unit_mode"] == mode and row["direction"] == direction]
              for mode, direction in categories]
    figure, axis = plt.subplots(figsize=(12, 6.5))
    boxes = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.58,
                         medianprops={"color": "#17242C", "linewidth": 1.5},
                         whiskerprops={"color": "#42535D"}, capprops={"color": "#42535D"})
    for patch, (_, direction) in zip(boxes["boxes"], categories):
        patch.set_facecolor(DIRECTION_COLORS[direction])
        patch.set_alpha(0.78)
    for position, category_values in enumerate(values, 1):
        axis.scatter([position] * len(category_values), category_values, marker="_",
                     color="#17242C", alpha=0.5, s=28, linewidths=0.7, zorder=3)
    axis.set_xticks(range(1, len(categories) + 1),
                    [f"{MODE_LABELS[mode]}\n{DIRECTION_LABELS[direction]}"
                     for mode, direction in categories], rotation=15, ha="right")
    axis.set_xlabel("Simultaneous topology and direction")
    axis.set_title("Per-session throughput under concurrency", loc="left", fontweight="bold")
    style(axis, "Receiver throughput per path (Gbit/s)")
    figure.tight_layout()
    relative = "concurrent_session_distribution.png"
    save(figure, output_dir / relative)
    return relative


def remove_obsolete(output_dir, current, previous):
    root = output_dir.resolve()
    owned = {"isolated_throughput.png", "concurrent_epoch_aggregate.png",
             "concurrent_session_distribution.png"}
    for relative in previous:
        if relative not in owned:
            continue
        candidate = (output_dir / relative).resolve()
        if (candidate.is_relative_to(root) and candidate.suffix == ".png"
                and not (output_dir / relative).is_symlink() and relative not in current):
            candidate.unlink(missing_ok=True)


def publish_plots(staging, output, files, manifest_record):
    """Publish a complete plot generation through one durable manifest switch."""
    manifest = output / "plot_manifest.json"
    if manifest.is_symlink():
        raise ValueError("plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("raw_run") != manifest_record["raw_run"]:
        raise ValueError("existing network plot manifest belongs to another raw run")
    previous = set(old.get("plots", []))
    for relative in files:
        path = output / relative
        if path.is_symlink() or (path.exists() and relative not in previous):
            raise ValueError(f"plot destination is not owned by this visualizer: {relative}")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("network plot generations may not be a symlink")
    generations.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    final = generations / generation
    committed = False
    try:
        shutil.copytree(staging, final)
        (final / "plot_owner.json").write_text(json.dumps({"raw_run": manifest_record["raw_run"],
                                                           "generation": generation}) + "\n")
        for file in final.iterdir():
            with file.open("rb") as source:
                os.fsync(source.fileno())
        final_fd = os.open(final, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(final_fd)
        finally:
            os.close(final_fd)
        parent_fd = os.open(generations, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        published = dict(manifest_record, generation=f"generations/{generation}",
                         plots=[f"generations/{generation}/{name}" for name in files])
        temporary = output / f".plot-manifest-{uuid.uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as target:
                target.write(json.dumps(published, indent=2) + "\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, manifest)
            committed = True
            directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        if not committed and final.is_dir():
            shutil.rmtree(final)
        raise
    remove_obsolete(output, files, previous)
    for directory in generations.iterdir():
        if directory == final or directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / "plot_owner.json"
        if (re.fullmatch(r"[0-9a-f]{32}", directory.name) and marker.is_file()
                and json.loads(marker.read_text()).get("raw_run") == manifest_record["raw_run"]):
            shutil.rmtree(directory)


def main(argv=None):
    args = parse_args(argv)
    try:
        run = args.run.resolve()
        output = args.output_dir.resolve(strict=False)
        if (output != run / "plots"
                or args.run.is_symlink() or args.output_dir.is_symlink()):
            raise ValueError("plots must be saved in the selected raw run's plots directory")
        measurements, epochs = load_raw(run)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=args.output_dir) as temporary_dir:
            staged = Path(temporary_dir)
            plots = [plot_isolated(measurements, staged),
                     plot_concurrent_epochs(epochs, staged),
                     plot_concurrent_sessions(measurements, staged)]
            plots = [plot for plot in plots if plot]
            manifest = {
                "raw_run": str(run),
                "measurements": len(measurements), "epochs": len(epochs), "plots": plots,
                "semantics": {
                    "isolated": "path medians with repetition marks and min-max whiskers",
                    "concurrent_epochs": "aggregate receiver throughput per atomic simultaneous epoch",
                    "concurrent_sessions": "per-path throughput distributions; paths are not summed",
                    "separation": "isolated and simultaneous modes are never combined",
                },
            }
            publish_plots(staged, args.output_dir, plots, manifest)
        print(f"Created {len(plots)} plots under {args.output_dir}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot visualize results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
