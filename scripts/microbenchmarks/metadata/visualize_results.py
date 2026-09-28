#!/usr/bin/env python3
"""Plot the seven mdtest phases directly from native stdout and run manifests."""

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

from run_mdtest import (RUNS, PHASES, build_command, pilot_units, plan_units,
                        preflight, validate_config, validate_measurement)


COLORS = {"anjuna2": "#16858C", "anjuna3": "#C26D31", "dual": "#42619B"}


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def save_figure(fig, output, name):
    target_path = output / name
    manifest = output / "plot_manifest.json"
    if (target_path.is_symlink() or manifest.is_symlink()
            or target_path.exists() and (not manifest.is_file()
                or name not in json.loads(manifest.read_text()).get("plots", []))):
        raise ValueError("metadata plot filename is not owned by this visualizer")
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
        raise ValueError("metadata plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("raw_run") != metadata["raw_run"]:
        raise ValueError("existing metadata plot manifest belongs to another run")
    previous = set(old.get("plots", []))
    for name in files:
        target = output / name
        if target.is_symlink() or target.exists() and name not in previous:
            raise ValueError("metadata plot destination belongs to another file")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("metadata plot generations may not be a symlink")
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
        raise ValueError("unsafe mdtest attempt path")
    candidate = root / path
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("mdtest evidence escapes the raw run")
    content = candidate.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("mdtest native evidence changed")
    return content.decode("utf-8") if text else json.loads(content)


def load_raw(root):
    owner = json.loads((root / "owner.json").read_text())
    domain = owner["domain"]
    if owner["run_id"] != root.name or domain not in ("metadata", "metadata-pilot"):
        raise ValueError("metadata run owner marker differs")
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = json.loads((root / "inventory.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    config = plan["config"]
    validate_config(config)
    preflight(inventory)
    expected = pilot_units(config) if domain.endswith("-pilot") else plan_units(config)
    if plan["units"] != expected or len(manifest["units"]) != len(expected):
        raise ValueError("metadata canonical plan incomplete")
    plan_hash = fingerprint({"domain": domain, "config": config, "units": expected})
    run_hash = fingerprint({"domain": domain, "plan": plan_hash, "inventory": inventory})
    if plan.get("fingerprint") != plan_hash or manifest.get("fingerprint") != run_hash:
        raise ValueError("metadata scientific fingerprint differs")
    rows = []
    for unit, planned in zip(manifest["units"], expected):
        if any(unit.get(field) != value for field, value in planned.items()):
            raise ValueError("metadata case differs from planned factors")
        attempts = unit.get("attempts", [])
        if (not attempts or attempts[-1].get("state") != "completed"
                or attempts[-1].get("cleanup") != "completed"):
            raise ValueError(f"{unit['id']}: mdtest invocation or cleanup incomplete")
        attempt = attempts[-1]
        evidence = attempt["evidence"]
        prefix = f"attempts/{unit['id']}/{attempt['id']}"
        saved = {name: evidence_file(root, evidence[name], evidence["sha256"][name],
                                     prefix, text=name in ("native", "rank_report"))
                 for name in ("native", "command", "exit", "telemetry", "rank_map", "rank_report")}
        if saved["exit"].get("returncode") != 0 or saved["exit"].get("timeout"):
            raise ValueError("mdtest exit/timeout evidence invalid")
        expected_hosts = (("anjuna2", "anjuna3") if unit["placement"] == "dual"
                          else (unit["placement"],))
        ranks = len(expected_hosts) * unit["ranks_per_client"]
        mapped = saved["rank_map"]
        if (not isinstance(mapped, list) or len(mapped) != ranks
                or {entry.get("rank") for entry in mapped} != set(range(ranks))
                or any(sum(entry["host"] == host for entry in mapped) != unit["ranks_per_client"]
                       for host in expected_hosts)):
            raise ValueError("mdtest MPI rank map differs")
        for host in expected_hosts:
            cores = [entry["core"] for entry in mapped if entry["host"] == host]
            if len(set(cores)) != len(cores) or any(core not in inventory["allowed_cores"][host]
                                                  for core in cores):
                raise ValueError("mdtest rank binding oversubscribed a core")
        pattern = inventory["tools"]["mpi_binding_pattern"]
        matches = list(re.finditer(pattern, saved["rank_report"], re.MULTILINE))
        if (len(matches) != ranks or sorted(mapped, key=lambda row: row["rank"]) != sorted((
                {"rank": int(match.group("rank")), "host": match.group("host"),
                 "core": int(match.group("core"))} for match in matches),
                key=lambda row: row["rank"])):
            raise ValueError("native mdtest MPI binding report differs")
        if saved["command"] != build_command(unit, inventory["tools"]["mpirun"],
                                               inventory["tools"]["mdtest"], evidence["owned_path"]):
            raise ValueError("measured mdtest argv differs from the plan")
        telemetry = saved["telemetry"]
        telemetry["rank_map_verified"] = True
        phases = validate_measurement(unit, saved["native"], telemetry,
                                      schema=inventory["tools"]["mdtest_schema"],
                                      workdir=evidence["owned_path"])
        for phase, metrics in phases.items():
            rows.append({"unit_id": unit["id"], "phase": phase, "placement": unit["placement"],
                         "layout": unit["layout"], "ranks": unit["ranks_per_client"],
                         "rate": metrics["rate"], "repetition": unit["repetition"]})
    if manifest.get("restoration") != "completed":
        raise ValueError("metadata run restoration incomplete")
    restoration = manifest["restoration_evidence"]
    restored = evidence_file(root, restoration["path"], restoration["sha256"],
                             "state-transitions")
    if (restored.get("run_fingerprint") != run_hash
            or restored.get("baseline") != inventory.get("restoration_baseline")
            or restored.get("restored") != inventory.get("restoration_baseline")
            or restored.get("netbench_off") != {"anjuna2": True, "anjuna3": True}
            or not restored.get("watchdog_released")):
        raise ValueError("metadata state restoration not verified")
    return rows


def plots(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for phase in PHASES:
        selected = [row for row in rows if row["phase"] == phase]
        if not selected:
            continue
        fig, axis = plt.subplots(figsize=(9.5, 6))
        for placement in ("anjuna2", "anjuna3", "dual"):
            for layout, marker, linestyle in (("flat", "o", "-"), ("per_rank", "s", "--")):
                grouped = {n: [row["rate"] for row in selected
                               if row["placement"] == placement and row["layout"] == layout
                               and row["ranks"] == n] for n in (1, 4, 16)}
                levels = [n for n, values in grouped.items() if values]
                if not levels:
                    continue
                medians = [statistics.median(grouped[n]) for n in levels]
                positions = [1, 2, 3][:len(levels)] if levels == [1, 4, 16] else [
                    (1, 4, 16).index(n) + 1 for n in levels]
                axis.plot(positions, medians, color=COLORS[placement], marker=marker,
                          linestyle=linestyle, label=f"{placement} · {layout}")
                for x, n, median in zip(positions, levels, medians):
                    values = grouped[n]
                    axis.errorbar(x, median, yerr=[[median - min(values)], [max(values) - median]],
                                  fmt="none", color=COLORS[placement], capsize=3)
                    axis.scatter([x] * len(values), values, marker="_", color=COLORS[placement])
        axis.set_xticks((1, 2, 3), ("1", "4", "16"))
        axis.set_xlabel("MPI ranks per participating client (dual = twice as many total ranks)")
        axis.set_ylabel("Native mdtest aggregate operations/s")
        axis.set_title(f"{phase.replace('_', ' ').title()} · zero-byte namespace")
        axis.grid(axis="y", alpha=.25)
        axis.legend(ncol=2, fontsize="small", frameon=False)
        fig.tight_layout()
        name = f"{phase}.png"
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
            raise ValueError("metadata plots must stay inside the selected raw run")
        rows = load_raw(root)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=output) as directory:
            staged = Path(directory)
            files = plots(rows, staged)
            publish_plots(staged, output, files, {
                "raw_run": str(root), "phase_rows": len(rows), "plots": files,
                "semantics": "Each file/directory operation and layout is separate; rates are whole-path aggregate mdtest rates."})
        print(f"Created {len(files)} metadata plots in {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot visualize mdtest results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
