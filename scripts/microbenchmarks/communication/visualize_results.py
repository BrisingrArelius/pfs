#!/usr/bin/env python3
"""Plot raw NetBench IOR attempts without an intermediate analysis step."""

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

from run_communication import (RUNS, build_command, netbench_transaction, pilot_units,
                               plan_units, preflight, validate_config, validate_measurement)


COLORS = {1: "#315A7D", 4: "#D08B32"}


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def save_figure(fig, output, name):
    target_path = output / name
    manifest = output / "plot_manifest.json"
    if (target_path.is_symlink() or manifest.is_symlink()
            or target_path.exists() and (not manifest.is_file()
                or name not in json.loads(manifest.read_text()).get("plots", []))):
        raise ValueError("communication plot filename is not owned by this visualizer")
    temp = output / f".{name}-{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("xb") as target:
            fig.savefig(target, format="png", dpi=180)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, target_path)
    finally:
        temp.unlink(missing_ok=True)
        plt.close(fig)


def publish_plots(staging, output, files, metadata):
    manifest = output / "plot_manifest.json"
    if manifest.is_symlink():
        raise ValueError("communication plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("raw_run") != metadata["raw_run"]:
        raise ValueError("existing communication plot manifest belongs to another run")
    previous = set(old.get("plots", []))
    for name in files:
        target = output / name
        if target.is_symlink() or target.exists() and name not in previous:
            raise ValueError("communication plot destination belongs to another file")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("communication plot generations may not be a symlink")
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


def native_artifact(root, relative, digest, prefix, *, text=False):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not str(path).startswith(prefix + "/"):
        raise ValueError("unsafe or unrelated communication artifact")
    candidate = root / path
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("communication artifact escapes the run")
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("communication artifact changed after capture")
    return data.decode("utf-8") if text else json.loads(data)


def validate_ranks(rank_map, report, pattern, clients, ranks_per_client):
    tasks = len(clients) * ranks_per_client
    if (not isinstance(rank_map, list) or len(rank_map) != tasks
            or any(not isinstance(entry, dict) for entry in rank_map)
            or {entry.get("rank") for entry in rank_map} != set(range(tasks))
            or {entry.get("host") for entry in rank_map} != set(clients)):
        raise ValueError("communication MPI rank placement differs")
    for host in clients:
        cores = [entry.get("core") for entry in rank_map if entry["host"] == host]
        if (len(cores) != ranks_per_client or any(type(core) is not int or core < 0 for core in cores)
                or len(set(cores)) != len(cores)):
            raise ValueError("communication MPI core binding differs")
    matches = list(re.finditer(pattern, report, re.MULTILINE))
    if len(matches) != tasks or sorted(rank_map, key=lambda row: row["rank"]) != sorted((
            {"rank": int(match.group("rank")), "host": match.group("host"),
             "core": int(match.group("core"))} for match in matches), key=lambda row: row["rank"]):
        raise ValueError("native MPI binding report differs from rank map")


def load_raw(root):
    root = Path(root).resolve()
    owner = json.loads((root / "owner.json").read_text())
    domain = owner["domain"]
    if owner["run_id"] != root.name or domain not in ("communication", "communication-pilot"):
        raise ValueError("wrong communication run identity")
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = json.loads((root / "inventory.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    config = plan["config"]
    validate_config(config)
    preflight(inventory)
    expected = pilot_units(config) if domain.endswith("-pilot") else plan_units(config)
    if plan["units"] != expected or len(manifest["units"]) != len(expected):
        raise ValueError("communication canonical plan incomplete")
    plan_hash = fingerprint({"domain": domain, "config": config, "units": expected})
    run_hash = fingerprint({"domain": domain, "plan": plan_hash, "inventory": inventory})
    if plan.get("fingerprint") != plan_hash or manifest.get("fingerprint") != run_hash:
        raise ValueError("communication scientific fingerprint differs")
    rows = []
    for unit, planned in zip(manifest["units"], expected):
        if any(unit.get(field) != value for field, value in planned.items()):
            raise ValueError("communication factors differ from the canonical plan")
        attempts = unit.get("attempts", [])
        if (not attempts or attempts[-1].get("state") != "completed"
                or attempts[-1].get("cleanup") != "completed"):
            raise ValueError(f"{unit['id']}: measured attempt or cleanup incomplete")
        attempt = attempts[-1]
        evidence = attempt["evidence"]
        prefix = f"attempts/{unit['id']}/{attempt['id']}"
        saved = {name: native_artifact(root, evidence[name], evidence["sha256"][name], prefix)
                 for name in ("native", "telemetry", "state", "command", "exit", "rank_map")}
        if saved["exit"].get("returncode") != 0 or saved["exit"].get("timeout"):
            raise ValueError("IOR exited abnormally")
        clients = ("anjuna2", "anjuna3") if unit["placement"] == "dual" else (unit["placement"],)
        report = native_artifact(root, evidence["rank_report"], evidence["sha256"]["rank_report"],
                                 prefix, text=True)
        validate_ranks(saved["rank_map"], report,
                       inventory["tools"]["mpi_binding_pattern"], clients,
                       unit["ranks_per_client"])
        state, telemetry = saved["state"], saved["telemetry"]
        netbench_transaction(clients, before=state["before_mode"], during=state["during_mode"],
                             after=state["after_mode"], watchdog=state["watchdog_verified"])
        for field in ("before_mode", "during_mode", "after_mode", "watchdog_verified", "actual_layouts"):
            if state.get(field) != telemetry.get(field):
                raise ValueError("NetBench transition and saved telemetry differ")
        layouts = state["actual_layouts"]
        expected_files = len(clients) * unit["ranks_per_client"] if unit["organization"] == "fpp" else 1
        eligible = {target["id"]: target["oss"] for target in inventory["targets"]}
        if len(layouts) != expected_files:
            raise ValueError("not every communication file was checked")
        for layout in layouts:
            if (len(layout) != unit["stripes"] or len(set(layout)) != len(layout)
                    or not set(layout).issubset(eligible)
                    or unit["stripes"] == 4 and {eligible[target] for target in layout}
                    != {"colva1", "colva2", "colva3", "colva4"}):
                raise ValueError("communication actual target fan-out differs")
        if (evidence["summary_path"] != str(root / evidence["native"])
                or saved["command"] != build_command(unit, inventory["tools"]["mpirun"],
                    inventory["tools"]["ior"], evidence["owned_path"], evidence["summary_path"])):
            raise ValueError("communication measured argv differs")
        metrics = validate_measurement(unit, saved["native"], telemetry, config,
                                       owned_path=evidence["owned_path"],
                                       ior_version=inventory["tools"]["ior_version"])
        rows.append({"unit_id": unit["id"], "placement": unit["placement"],
                     "ranks": unit["ranks_per_client"], "direction": unit["direction"],
                     "organization": unit["organization"], "stripes": unit["stripes"],
                     "throughput": metrics["mib_per_second"],
                     "backend_ratio": metrics["backend_ratio"],
                     "evidence_status": metrics["evidence_status"]})
    if manifest.get("restoration") != "completed":
        raise ValueError("NetBench/pool/chooser restoration incomplete")
    restoration = manifest["restoration_evidence"]
    restored = native_artifact(root, restoration["path"], restoration["sha256"],
                               "state-transitions")
    if (restored.get("run_fingerprint") != run_hash
            or restored.get("baseline") != inventory.get("restoration_baseline")
            or restored.get("restored") != inventory.get("restoration_baseline")
            or restored.get("netbench_off") != {"anjuna2": True, "anjuna3": True}
            or not restored.get("watchdog_released")):
        raise ValueError("NetBench/pool/chooser baseline not restored")
    return rows


def plots(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for direction in ("read", "write"):
        subset = [row for row in rows if row["direction"] == direction]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for axis, organization in zip(axes, ("fpp", "shared")):
            for position, (placement, ranks) in enumerate(
                    (placement, ranks) for placement in ("anjuna2", "anjuna3", "dual")
                    for ranks in (1, 4)):
                for stripe in (1, 4):
                    values = [row["throughput"] for row in subset
                              if (row["organization"], row["placement"], row["ranks"], row["stripes"])
                              == (organization, placement, ranks, stripe)]
                    if not values:
                        continue
                    x = position + (-.17 if stripe == 1 else .17)
                    median = statistics.median(values)
                    axis.bar(x, median, width=.32, color=COLORS[stripe])
                    axis.errorbar(x, median, yerr=[[median - min(values)], [max(values) - median]],
                                  fmt="none", color="#26343D", capsize=2)
                    axis.scatter([x] * len(values), values, marker="_", color="#26343D")
            axis.set_title("File per rank" if organization == "fpp" else "Shared file")
            axis.set_xticks(range(6), [f"{host}\n{n} rank/client" for host in
                                      ("anjuna2", "anjuna3", "dual") for n in (1, 4)],
                            rotation=25, ha="right")
            axis.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Synthetic IOR throughput (MiB/s)")
        axes[0].bar([], [], color=COLORS[1], label="1 eligible target")
        axes[0].bar([], [], color=COLORS[4], label="4 OSS targets")
        axes[0].legend(frameon=False)
        fig.suptitle(f"NetBench synthetic {direction} · distinct organizations")
        fig.tight_layout()
        name = f"synthetic_{direction}.png"
        save_figure(fig, output, name)
        files.append(name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axis, direction in zip(axes, ("read", "write")):
        for position, placement in enumerate(("anjuna2", "anjuna3", "dual")):
            values = [row["backend_ratio"] for row in rows
                      if row["direction"] == direction and row["placement"] == placement]
            axis.scatter([position] * len(values), values, alpha=.65, color="#8D4C49")
        axis.set_xticks(range(3), ("anjuna2", "anjuna3", "dual"))
        axis.set_title(f"Synthetic {direction}")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Backend device bytes / IOR logical bytes")
    fig.suptitle("NetBench backend evidence (not normal storage throughput)")
    fig.tight_layout()
    save_figure(fig, output, "backend_ratio.png")
    files.append("backend_ratio.png")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.run.resolve()
        output = (args.output_dir or root / "plots").resolve(strict=False)
        if root.parent != RUNS or output != root / "plots" or args.run.is_symlink():
            raise ValueError("communication plots must stay in the selected raw run")
        rows = load_raw(root)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=output) as directory:
            staged = Path(directory)
            files = plots(rows, staged)
            publish_plots(staged, output, files, {
                "raw_run": str(root), "measurements": len(rows), "plots": files,
                "evidence_status": sorted({row["evidence_status"] for row in rows}),
                "max_backend_ratio_by_direction": {
                    direction: max((row["backend_ratio"] for row in rows if row["direction"] == direction),
                                   default=None) for direction in ("read", "write")},
                "semantics": "Synthetic BeeGFS requests; reads/writes, file organizations and stripe fan-out stay separate."})
        print(f"Created {len(files)} communication plots in {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot visualize communication results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
