#!/usr/bin/env python3
"""Plot native end-to-end IOR results at fixed workload and placement factors."""

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

from run_placement import (RUNS, build_command, eligible_targets, pilot_units, plan_units,
                           validate_config, validate_measurement)


COLORS = {"D": "#55718F", "S": "#16858C", "H": "#B86B25"}


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def save_figure(fig, output, name):
    target_path = output / name
    manifest = output / "plot_manifest.json"
    if (target_path.is_symlink() or manifest.is_symlink()
            or target_path.exists() and (not manifest.is_file()
                or name not in json.loads(manifest.read_text()).get("plots", []))):
        raise ValueError("placement plot filename is not owned by this visualizer")
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
        raise ValueError("placement plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("raw_run") != metadata["raw_run"]:
        raise ValueError("existing placement plot manifest belongs to another run")
    previous = set(old.get("plots", []))
    for name in files:
        target = output / name
        if target.is_symlink() or target.exists() and name not in previous:
            raise ValueError("placement plot destination belongs to another file")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("placement plot generations may not be a symlink")
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


def evidence_file(root, relative, digest, prefix, *, text=False):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not str(path).startswith(prefix + "/"):
        raise ValueError("placement artifact is not in the planned attempt")
    candidate = root / path
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("placement artifact escapes the run")
    content = candidate.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("placement native evidence changed")
    return content.decode("utf-8") if text else json.loads(content)


def validate_ranks(rank_map, report, pattern, hosts, ranks_per_host):
    count = len(hosts) * ranks_per_host
    if (not isinstance(rank_map, list) or len(rank_map) != count
            or any(not isinstance(entry, dict) for entry in rank_map)
            or {entry.get("rank") for entry in rank_map} != set(range(count))
            or {entry.get("host") for entry in rank_map} != set(hosts)):
        raise ValueError("placement rank map differs")
    for host in hosts:
        cores = [entry.get("core") for entry in rank_map if entry["host"] == host]
        if (len(cores) != ranks_per_host or any(type(core) is not int or core < 0 for core in cores)
                or len(set(cores)) != len(cores)):
            raise ValueError("placement ranks are oversubscribed")
    matches = list(re.finditer(pattern, report, re.MULTILINE))
    if (len(matches) != count or sorted(rank_map, key=lambda row: row["rank"]) != sorted((
            {"rank": int(match.group("rank")), "host": match.group("host"),
             "core": int(match.group("core"))} for match in matches), key=lambda row: row["rank"])):
        raise ValueError("native MPI binding report differs from placement rank map")


def load_raw(root):
    owner = json.loads((root / "owner.json").read_text())
    domain = owner["domain"]
    if owner["run_id"] != root.name or domain not in ("placement", "placement-pilot"):
        raise ValueError("placement run marker differs")
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = json.loads((root / "inventory.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    config = plan["config"]
    validate_config(config)
    expected = pilot_units(config) if domain.endswith("-pilot") else plan_units(config)
    if plan["units"] != expected or len(manifest["units"]) != len(expected):
        raise ValueError("placement canonical plan incomplete")
    plan_hash = fingerprint({"domain": domain, "config": config, "units": expected})
    run_hash = fingerprint({"domain": domain, "plan": plan_hash, "inventory": inventory})
    if plan.get("fingerprint") != plan_hash or manifest.get("fingerprint") != run_hash:
        raise ValueError("placement scientific fingerprint differs")
    counts = {storage: len(eligible_targets(inventory, storage)) for storage in ("D", "S", "H")}
    rows = []
    for unit, planned in zip(manifest["units"], expected):
        if any(unit.get(field) != value for field, value in planned.items()):
            raise ValueError("placement unit differs from canonical plan")
        attempts = unit.get("attempts", [])
        if (not attempts or attempts[-1].get("state") != "completed"
                or attempts[-1].get("cleanup") != "completed"):
            raise ValueError(f"{unit['id']}: measured attempt or cleanup incomplete")
        attempt = attempts[-1]
        evidence = attempt["evidence"]
        prefix = f"attempts/{unit['id']}/{attempt['id']}"
        saved = {name: evidence_file(root, evidence[name], evidence["sha256"][name], prefix)
                 for name in ("native", "telemetry", "state", "command", "exit", "rank_map")}
        if saved["exit"].get("returncode") != 0 or saved["exit"].get("timeout"):
            raise ValueError("placement IOR failed")
        from run_placement import CONCURRENCY
        hosts, per_host = CONCURRENCY[unit["concurrency"]]
        report = evidence_file(root, evidence["rank_report"], evidence["sha256"]["rank_report"],
                               prefix, text=True)
        validate_ranks(saved["rank_map"], report, inventory["tools"]["mpi_binding_pattern"],
                       hosts, per_host)
        state = saved["state"]
        if (state.get("after") != inventory["restoration_baseline"]
                or not state.get("watchdog_verified")):
            raise ValueError("placement block restoration lacks evidence")
        from run_placement import apply_block_state
        apply_block_state(unit, observed=state["during"], inventory=inventory)
        if (evidence["summary_path"] != str(root / evidence["native"])
                or saved["command"] != build_command(unit, inventory["tools"]["mpirun"],
                    inventory["tools"]["ior"], evidence["owned_path"], evidence["summary_path"])):
            raise ValueError("placement measured argv differs")
        metrics = validate_measurement(unit, saved["native"], saved["telemetry"],
                                       config, inventory, owned_path=evidence["owned_path"],
                                       ior_version=inventory["tools"]["ior_version"])
        rows.append({"unit_id": unit["id"], "capacity": unit["capacity"],
                     "storage": unit["storage"], "chooser": unit["chooser"],
                     "stripes": unit["stripes"], "chunk_bytes": unit["chunk_bytes"],
                     "workload": unit["workload"], "concurrency": unit["concurrency"],
                     "completion_reason": metrics["completion_reason"],
                     "mib_per_second": metrics["mib_per_second"], "iops": metrics["iops"]})
    if manifest.get("restoration") != "completed":
        raise ValueError("pool/chooser/capacity restoration incomplete")
    restoration = manifest["restoration_evidence"]
    restored = evidence_file(root, restoration["path"], restoration["sha256"],
                             "state-transitions")
    if (restored.get("run_fingerprint") != run_hash
            or restored.get("baseline") != inventory.get("restoration_baseline")
            or restored.get("restored") != inventory.get("restoration_baseline")
            or restored.get("netbench_off") != {"anjuna2": True, "anjuna3": True}
            or not restored.get("watchdog_released")):
        raise ValueError("placement pool/chooser baseline not restored")
    return rows, counts


def plot(rows, counts, output):
    output.mkdir(parents=True, exist_ok=True)
    files = []
    factors = sorted({(row["capacity"], row["chooser"], row["workload"],
                       row["completion_reason"]) for row in rows})
    for capacity, chooser, workload, completion in factors:
        selected = [row for row in rows if (row["capacity"], row["chooser"], row["workload"],
                    row["completion_reason"]) == (capacity, chooser, workload, completion)]
        metric = "iops" if workload.startswith("rand_") else "mib_per_second"
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
        for axis, concurrency in zip(axes.flat, ("a2_r1", "a2_r4", "dual_r1", "dual_r4")):
            for storage, offset in (("D", -.22), ("S", 0), ("H", .22)):
                available = []
                for position, (stripes, chunk) in enumerate((stripes, chunk)
                         for stripes in (1, 2, 4) for chunk in (262144, 524288, 1048576)):
                    values = [row[metric] for row in selected
                              if (row["concurrency"], row["storage"], row["stripes"], row["chunk_bytes"])
                              == (concurrency, storage, stripes, chunk)]
                    if not values:
                        continue
                    x = position + offset
                    median = statistics.median(values)
                    axis.errorbar(x, median, yerr=[[median - min(values)], [max(values) - median]],
                                  fmt="o", color=COLORS[storage], capsize=2, markersize=4)
                    available.append((x, median))
                if available:
                    axis.plot([x for x, _ in available], [value for _, value in available],
                              color=COLORS[storage], linewidth=1,
                              label=f"{storage} · {counts[storage]} eligible targets")
            axis.set_title(concurrency)
            axis.set_xticks(range(9), [f"{stripes}×{size}" for stripes in (1, 2, 4)
                                       for size in ("256K", "512K", "1M")], rotation=45, ha="right")
            axis.grid(axis="y", alpha=.25)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(frameon=False, fontsize="x-small")
            else:
                axis.text(.5, .5, "No completed case at this concurrency",
                          ha="center", va="center", transform=axis.transAxes, color="#66747A")
        for axis in axes[:, 0]:
            axis.set_ylabel("Native IOR IOPS" if metric == "iops" else "Native IOR MiB/s")
        fig.suptitle(f"{workload} · {capacity} · {chooser} · {completion} (size-or-60s)")
        fig.tight_layout()
        name = f"{capacity}_{chooser}_{workload}_{completion}.png"
        save_figure(fig, output, name)
        files.append(name)
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
            raise ValueError("placement plots must stay in the selected raw run")
        rows, counts = load_raw(root)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=output) as directory:
            staged = Path(directory)
            files = plot(rows, counts, staged)
            publish_plots(staged, output, files, {
                "raw_run": str(root), "measurements": len(rows), "plots": files,
                "semantics": "D/S/H compare only at fixed workload, chooser, capacity, geometry, concurrency and completion reason."})
        print(f"Created {len(files)} placement plots in {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot visualize placement results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
