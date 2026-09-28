#!/usr/bin/env python3
"""Plot verified cache reads directly from raw IOR attempts and telemetry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cache import (RUNS, STATES, pilot_units, plan_units,
                       preflight, validate_config, validate_measurement, build_command)


LABELS = {"client_miss_server_miss": "Backend\n(client + OSS miss)",
          "client_miss_server_hit": "OSS RAM\n(client miss)",
          "client_hit": "Client RAM\n(client hit)"}
COLORS = ("#B86B25", "#16858C", "#42619B")


def artifact(root, relative, digest, prefix=None, *, text=False):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or prefix and not str(path).startswith(prefix + "/"):
        raise ValueError("unsafe or unrelated cache artifact")
    candidate = root / path
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("cache evidence escapes run directory")
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("cache native evidence changed after capture")
    return data.decode("utf-8") if text else json.loads(data)


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def save_figure(fig, output, name):
    target_path = output / name
    manifest = output / "plot_manifest.json"
    if (target_path.is_symlink() or manifest.is_symlink()
            or target_path.exists() and (not manifest.is_file()
                or name not in json.loads(manifest.read_text()).get("plots", []))):
        raise ValueError("cache plot filename is not owned by this visualizer")
    temp = output / f".{name}-{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("xb") as destination:
            fig.savefig(destination, format="png", dpi=180)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp, target_path)
    finally:
        temp.unlink(missing_ok=True)
        plt.close(fig)


def publish_plots(staging, output, files, metadata):
    manifest = output / "plot_manifest.json"
    if manifest.is_symlink():
        raise ValueError("cache plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("raw_run") != metadata["raw_run"]:
        raise ValueError("existing cache plot manifest belongs to another run")
    previous = set(old.get("plots", []))
    for name in files:
        target = output / name
        if target.is_symlink() or target.exists() and name not in previous:
            raise ValueError("cache plot destination belongs to another file")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("cache plot generations may not be a symlink")
    generations.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    final = generations / generation
    committed = False
    try:
        shutil.copytree(staging, final)
        (final / "plot_owner.json").write_text(json.dumps({"raw_run": metadata["raw_run"],
                                                           "generation": generation}) + "\n")
        for file in final.iterdir():
            with file.open("rb") as source:
                os.fsync(source.fileno())
        for directory in (final, generations):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        published = dict(metadata, generation=f"generations/{generation}",
                         plots=[f"generations/{generation}/{name}" for name in files])
        temporary = output / f".plot-manifest-{uuid.uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as target:
                target.write(json.dumps(published, indent=2) + "\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, manifest)
            committed = True
            descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        if not committed and final.is_dir():
            shutil.rmtree(final)
        raise
    for directory in generations.iterdir():
        if directory == final or directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / "plot_owner.json"
        if (re.fullmatch(r"[0-9a-f]{32}", directory.name) and marker.is_file()
                and json.loads(marker.read_text()).get("raw_run") == metadata["raw_run"]):
            shutil.rmtree(directory)


def load_raw(root):
    root = Path(root).resolve()
    if not root.is_dir() or (root / "manifest.json").is_symlink():
        raise ValueError("raw cache result directory missing")
    owner = json.loads((root / "owner.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = json.loads((root / "inventory.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    domain = owner["domain"]
    if owner["run_id"] != root.name or domain not in ("cache", "cache-pilot"):
        raise ValueError("cache run identity differs")
    config = plan["config"]
    validate_config(config)
    preflight(inventory)
    units = pilot_units(config) if domain.endswith("-pilot") else plan_units(config)
    if (plan["units"] != units or len(manifest["units"]) != len(units)
            or manifest["domain"] != domain):
        raise ValueError("cache plan is incomplete")
    plan_hash = fingerprint({"domain": domain, "config": config, "units": units})
    run_hash = fingerprint({"domain": domain, "plan": plan_hash, "inventory": inventory})
    if plan.get("fingerprint") != plan_hash or manifest.get("fingerprint") != run_hash:
        raise ValueError("cache scientific fingerprint differs")
    rows = []
    for saved, planned in zip(manifest["units"], units):
        if any(saved.get(key) != value for key, value in planned.items()):
            raise ValueError("cache unit differs from planned state")
        attempts = saved.get("attempts", [])
        if (not attempts or attempts[-1].get("state") != "completed"
                or attempts[-1].get("cleanup") != "completed"):
            raise ValueError(f"{saved['id']}: measurement or cleanup incomplete")
        attempt = attempts[-1]
        evidence = attempt["evidence"]
        prefix = f"attempts/{saved['id']}/{attempt['id']}"
        raw = {name: artifact(root, evidence[name], evidence["sha256"][name], prefix)
               for name in ("native", "telemetry", "state", "command", "exit", "rank_map")}
        rank_report = artifact(root, evidence["rank_report"], evidence["sha256"]["rank_report"],
                               prefix, text=True)
        if (not isinstance(raw["rank_map"], list) or len(raw["rank_map"]) != 1
                or not isinstance(raw["rank_map"][0], dict)
                or raw["rank_map"][0].get("rank") != 0
                or raw["rank_map"][0].get("host") != "anjuna2"
                or type(raw["rank_map"][0].get("core")) is not int):
            raise ValueError("cache IOR rank was not placed on anjuna2")
        pattern = inventory["tools"]["mpi_binding_pattern"]
        bindings = list(re.finditer(pattern, rank_report, re.MULTILINE))
        if (len(bindings) != 1 or int(bindings[0].group("rank")) != 0
                or bindings[0].group("host") != "anjuna2"
                or int(bindings[0].group("core")) != raw["rank_map"][0]["core"]):
            raise ValueError("native MPI binding report differs from saved rank map")
        if raw["exit"].get("returncode") != 0 or raw["exit"].get("timeout"):
            raise ValueError("IOR exited abnormally")
        state = raw["state"]
        telemetry = raw["telemetry"]
        if (not state.get("exclusive_allocation") or not state.get("writeback_settled")
                or state.get("file_identity_before") != state.get("file_identity_after")
                or not state.get("file_identity_before") or state.get("client_residency")
                != telemetry.get("client_residency")):
            raise ValueError("cache state preparation lacks evidence")
        from run_cache import prepare_cache_state
        if state.get("steps") != prepare_cache_state(saved["state"], evidence=state):
            raise ValueError("cache preparation steps differ")
        native = raw["native"]
        measured = validate_measurement(saved, native, telemetry, config,
                                        owned_file=evidence["owned_path"],
                                        ior_version=inventory["tools"]["ior_version"])
        if (evidence["summary_path"] != str(root / evidence["native"])
                or raw["command"] != build_command(inventory["tools"]["ior"],
                    inventory["tools"]["mpirun"], evidence["owned_path"],
                    evidence["summary_path"])):
            raise ValueError("measured cache command differs")
        logical = telemetry["logical_bytes"]
        rows.append({"unit_id": saved["id"], "state": saved["state"],
                     "achieved": measured["achieved_state"],
                     "throughput": measured["mib_per_second"],
                     "network_ratio": telemetry["network_bytes"] / logical,
                     "backend_ratio": telemetry["backend_read_bytes"] / logical})
    if manifest.get("restoration") != "completed":
        raise ValueError("cluster state restoration is incomplete")
    restoration = manifest["restoration_evidence"]
    restored = artifact(root, restoration["path"], restoration["sha256"])
    if (restored.get("run_fingerprint") != run_hash
            or restored.get("baseline") != inventory.get("restoration_baseline")
            or restored.get("restored") != inventory.get("restoration_baseline")
            or restored.get("netbench_off") != {"anjuna2": True, "anjuna3": True}
            or not restored.get("watchdog_released")):
        raise ValueError("cache mode/pool baseline was not verified restored")
    return rows


def plot(rows, output):
    verified = [row for row in rows if row["state"] == row["achieved"]]
    if not verified:
        raise ValueError("no verified cache states to plot")
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 6))
    for position, state in enumerate(STATES):
        values = [row["throughput"] for row in verified if row["state"] == state]
        if not values:
            continue
        median = statistics.median(values)
        axis.bar(position, median, color=COLORS[position], width=.63)
        axis.errorbar(position, median, yerr=[[median - min(values)], [max(values) - median]],
                      fmt="none", color="#26343D", capsize=4)
        axis.scatter([position] * len(values), values, marker="_", color="#26343D", zorder=3)
    axis.set_xticks(range(3), [LABELS[state] for state in STATES])
    axis.set_ylabel("BeeGFS POSIX read throughput (MiB/s)")
    axis.set_title("Verified cache paths · 8-GiB sequential read")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save_figure(fig, output, "cache_throughput.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    for position, state in enumerate(STATES):
        state_rows = [row for row in rows if row["state"] == state]
        for offset, field, color in ((-.12, "network_ratio", "#16858C"),
                                     (.12, "backend_ratio", "#B86B25")):
            for row in state_rows:
                axis.scatter(position + offset, row[field], color=color,
                             marker="o" if row["state"] == row["achieved"] else "x")
    axis.set_xticks(range(3), [LABELS[state] for state in STATES])
    axis.set_ylabel("Observed bytes / IOR logical bytes")
    axis.set_title("Cache-path evidence · crosses are unverified states")
    axis.scatter([], [], color="#16858C", label="client/OSS network")
    axis.scatter([], [], color="#B86B25", label="backend device reads")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save_figure(fig, output, "cache_path_evidence.png")
    return ["cache_throughput.png", "cache_path_evidence.png"], len(rows) - len(verified)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.run.resolve()
        output = (args.output_dir or root / "plots").resolve(strict=False)
        if root.parent != RUNS or output != root / "plots" or args.run.is_symlink():
            raise ValueError("cache plots must stay in the selected run's plots directory")
        rows = load_raw(root)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=output) as directory:
            staged = Path(directory)
            files, unverified = plot(rows, staged)
            publish_plots(staged, output, files, {
                "raw_run": str(root), "measurements": len(rows), "unverified": unverified,
                "plots": files, "semantics": "Only traffic-verified states enter throughput comparisons; no NetBench or direct I/O."})
        print(f"Created {len(files)} cache plots in {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot visualize cache results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
