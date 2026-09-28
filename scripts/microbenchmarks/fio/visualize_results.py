#!/usr/bin/env python3
"""Validate native FIO host runs and plot their performance in one step."""

import argparse
import hashlib
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

from run_fio import artifact_path, plan_cases, validate_config, validate_result

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as error:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error


COLORS = {"hdd": "#B86B25", "nvme": "#16858C"}
WORKLOAD_ORDER = ["seq_read", "seq_write", "rand_read_4k", "rand_write_4k", "rand_read_128k"]
WORKLOAD_LABELS = {
    "seq_read": "Sequential read\n1 MiB",
    "seq_write": "Sequential write\n1 MiB",
    "rand_read_4k": "Random read\n4 KiB",
    "rand_write_4k": "Random write\n4 KiB",
    "rand_read_128k": "Random read\n128 KiB",
}
def parse_args(argv=None):
    """Accept one copied run directory containing raw host manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="raw run directory or single host directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to <run>/plots")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.run / "plots"
    return args


def source_manifests(source):
    """Read only the selected host or its immediate host children, never analysis."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("raw FIO run directory is missing or is a symlink")
    paths = ([source / "manifest.json"] if (source / "manifest.json").is_file()
             else sorted(source.glob("colva*/manifest.json")))
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("no native host manifests found in the raw run")
    return paths


def scientific_config(manifest):
    """Match fixed workloads across hosts without including planning timeouts."""
    config = manifest["config"]
    scientific = {key: value for key, value in config.items()
                  if key not in {"planning", "measurement_timeout_seconds", "prepare"}}
    scientific["prepare"] = {key: value for key, value in config["prepare"].items()
                              if key != "timeout_seconds"}
    return {"mode": manifest.get("mode"), "scientific": scientific}


def load_rows(source):
    """Convert completed native fio.json attempts to plot values in memory."""
    paths = source_manifests(source)
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    baseline = scientific_config(manifests[0])
    if any(scientific_config(item) != baseline for item in manifests[1:]):
        raise ValueError("host manifests have incompatible scientific settings")
    if len({item["host"] for item in manifests}) != len(manifests):
        raise ValueError("duplicate host evidence in the selected FIO run")
    rows = []
    for path, manifest in zip(paths, manifests):
        root = path.parent
        config = manifest["config"]
        validate_config(config)
        if config["repetitions"] != (1 if manifest.get("mode") == "pilot" else 5):
            raise ValueError(f"{manifest['host']}: pilot/full repetition count differs")
        version = manifest["fio_version"]
        if not isinstance(version, str) or not version.startswith("fio-"):
            raise ValueError("unknown FIO binary version")
        scientific = {key: value for key, value in config.items()
                      if key not in {"planning", "measurement_timeout_seconds", "prepare"}}
        scientific["prepare"] = {key: value for key, value in config["prepare"].items()
                                 if key != "timeout_seconds"}
        scientific.update(host=manifest["host"], inventory=manifest["inventory"],
                          fio_version=version,
                          job_policy={"overwrite": 1, "fallocate": "none",
                                      "unique_filename": 0, "region_layout": "disjoint_offsets"})
        if hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest() != manifest.get("fingerprint"):
            raise ValueError(f"{manifest['host']}: scientific fingerprint differs")
        planned = plan_cases(config, manifest["inventory"])
        if len(manifest["cases"]) != len(planned):
            raise ValueError(f"{manifest['host']}: canonical case count differs")
        for saved, canonical in zip(manifest["cases"], planned):
            if any(saved.get(key) != canonical[key] for key in ("id", "target_id", "workload", "repetition")):
                raise ValueError(f"{manifest['host']}: canonical case factors differ")
        inventory = {item["target_id"]: item for item in manifest["inventory"]}
        if (manifest["host"] != root.name and len(paths) > 1):
            raise ValueError("host manifest is in an unexpected directory")
        for target_id, state in manifest["targets"].items():
            preparations = [item for item in state.get("preparations", [])
                            if item.get("state") == "completed"]
            if state.get("cleanup") != "completed" or not preparations:
                raise ValueError(f"{manifest['host']}: target preparation or cleanup incomplete")
            for preparation in preparations:
                prepared = artifact_path(root, preparation["artifacts"]) / "fio.json"
                validate_result(prepared, {
                    "jobname": f"target-{target_id}", "operation": "write",
                    "size": config["fio"]["size"] * config["fio"].get("numjobs", 1),
                    "runtime": 0, "hard_timeout": config["prepare"]["timeout_seconds"],
                    "time_based": False,
                })
        for case in manifest["cases"]:
            attempts = case["attempts"]
            if not attempts or attempts[-1].get("state") != "completed":
                raise ValueError(f"{case['id']}: measurement incomplete")
            attempt = attempts[-1]
            target = inventory[case["target_id"]]
            workload = case["workload"]
            job_count = config["fio"].get("numjobs", 1)
            names = ([f"target-{case['target_id']}-job-{index + 1}" for index in range(job_count)]
                     if job_count > 1 else [f"target-{case['target_id']}"])
            operation = "write" if "write" in workload["rw"] else "read"
            native = artifact_path(root, attempt["artifacts"]) / "fio.json"
            payload, metrics = validate_result(native, {
                "jobname": names[0], "jobnames": names, "operation": operation,
                "size": config["fio"]["size"] * job_count,
                "runtime": config["fio"]["runtime"],
                "hard_timeout": config["measurement_timeout_seconds"],
                "time_based": bool(config["fio"].get("time_based")),
            })
            if (metrics["completion_reason"] != attempt["completion_reason"]
                    or metrics["io_bytes"] != attempt["io_bytes"]):
                raise ValueError(f"{case['id']}: native and manifest completion differ")
            stats = [job[operation] for job in payload["jobs"]]
            for item in stats:
                duration = item["runtime"] / 1000
                if (abs(item["bw_bytes"] - item["io_bytes"] / duration)
                        > max(1024, item["bw_bytes"] * .02)
                        or abs(item["iops"] - item["total_ios"] / duration)
                        > max(.01, item["iops"] * .02)):
                    raise ValueError(f"{case['id']}: native FIO rates contradict bytes or I/O count")
            bandwidth = sum(item["bw_bytes"] for item in stats) / 2**20
            iops = sum(item["iops"] for item in stats)
            if any(not math.isfinite(value) or value <= 0 for value in (bandwidth, iops)):
                raise ValueError(f"{case['id']}: invalid native throughput")
            if target["media"] not in COLORS or workload["name"] not in WORKLOAD_LABELS:
                raise ValueError("unknown FIO media or workload")
            rows.append({"host": manifest["host"], "target_id": case["target_id"],
                         "media": target["media"], "fio_version": version,
                         "workload": workload["name"],
                         "repetition": case["repetition"], "bw_mib_s": bandwidth,
                         "iops": iops, "artifact": str(native.relative_to(source))})
    if not rows:
        raise ValueError("no completed FIO measurements")
    return rows, paths


def groups(rows, keys):
    """Group dictionaries by a tuple of named fields."""
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        result.setdefault(key, []).append(row)
    return result


def style_axis(axis, ylabel, log=False):
    """Apply one restrained treatment to a vertical metric axis."""
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, path):
    """Save a plot deterministically and release its matplotlib resources."""
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("plot output may not follow a symlink")
    if path.exists():
        manifest = path.parent.parent / "plot_manifest.json"
        relative = str(path.relative_to(manifest.parent))
        if (manifest.is_symlink() or not manifest.is_file()
                or relative not in json.loads(manifest.read_text()).get("plots", [])):
            raise ValueError("existing FIO plot is not owned by this visualizer")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as target:
            figure.savefig(target, format="png", dpi=220, bbox_inches="tight",
                           facecolor="white")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def use_log(values):
    """Use a logarithmic axis only when linear scale would hide a category."""
    return min(values) > 0 and max(values) / min(values) >= 20


def plot_workload_by_ost(rows, workload, version, output_dir):
    """Plot one access pattern and FIO version, keeping each OST separate."""
    selected = [row for row in rows if row["workload"] == workload
                and row["fio_version"] == version]
    grouped = groups(selected, ("host", "target_id", "media"))
    targets = sorted(grouped)
    labels = [f"{host}\n{target}" for host, target, media in targets]
    all_values = []
    media_medians = {media: [] for media in COLORS}
    figure, axis = plt.subplots(figsize=(max(14, len(targets) * 0.52), 7))
    for position, target in enumerate(targets):
        values = [row["bw_mib_s"] for row in grouped[target]]
        all_values.extend(values)
        color = COLORS[target[2]]
        median = statistics.median(values)
        media_medians[target[2]].append(median)
        axis.bar(position, median, width=0.68, color=color, edgecolor="#29333A",
                 linewidth=0.6, alpha=0.85, zorder=2)
        axis.errorbar(position, median,
                      yerr=[[median - min(values)], [max(values) - median]],
                      fmt="none", ecolor="#29333A", elinewidth=1.2, capsize=3, zorder=4)
        axis.scatter([position] * len(values), values, marker="_", s=65,
                     color="#29333A", linewidths=1.1, alpha=0.8, zorder=5)
    axis.set_xticks(range(len(targets)), labels, rotation=45, ha="right")
    axis.set_xlabel("OST ID")
    axis.set_title(f"{WORKLOAD_LABELS[workload].replace(chr(10), ' ')} bandwidth by OST · {version}")
    style_axis(axis, "Bandwidth (MiB/s)", use_log(all_values))
    for media in ("hdd", "nvme"):
        if not media_medians[media]:
            continue
        average = statistics.mean(media_medians[media])
        axis.axhline(average, color=COLORS[media], linestyle=":", linewidth=1.8, zorder=1)
        axis.text(0.995, average, f"{media.upper()} average: {average:,.1f} MiB/s",
                  transform=axis.get_yaxis_transform(), color=COLORS[media],
                  ha="right", va="bottom", fontweight="bold")
    legend = [Patch(facecolor=COLORS[media], label=media.upper()) for media in ("hdd", "nvme")]
    legend.append(Line2D([], [], color="#29333A", marker="_", linestyle="-",
                         label="Repetitions / min-max"))
    axis.legend(handles=legend, frameon=False, ncol=4)
    figure.tight_layout()
    slug = re.sub(r"[^a-z0-9]+", "_", version.lower()).strip("_")
    relative = f"by_access_pattern/{workload}_{slug}.png"
    save_figure(figure, output_dir / relative)
    return relative


def remove_obsolete_plots(output_dir, current_files, previous):
    """Remove only exact previously generated plot names after publication."""
    root = output_dir.resolve()
    for relative in previous:
        if (not isinstance(relative, str) or not any(re.fullmatch(
                rf"by_access_pattern/{workload}(?:_[a-z0-9_]+)?\.png", relative)
                for workload in WORKLOAD_ORDER)):
            continue
        candidate = (output_dir / relative).resolve()
        if (candidate.is_relative_to(root) and candidate.suffix == ".png"
                and not (output_dir / relative).is_symlink() and relative not in current_files):
            candidate.unlink(missing_ok=True)


def publish_plots(staging, output, files, metadata):
    """Switch one checksummed generation by a single durable manifest replace."""
    manifest = output / "plot_manifest.json"
    if manifest.is_symlink():
        raise ValueError("plot manifest may not be a symlink")
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if old and old.get("source") != metadata["source"]:
        raise ValueError("existing FIO plot manifest belongs to a different raw run")
    previous = set(old.get("plots", []))
    # Legacy flat outputs belong to an older visualizer, but never overwrite them.
    for relative in files:
        target = output / relative
        if target.is_symlink() or target.parent.is_symlink() or (target.exists() and relative not in previous):
            raise ValueError(f"plot destination is not owned by the visualizer: {relative}")
    generations = output / "generations"
    if generations.is_symlink():
        raise ValueError("FIO plot generations may not be a symlink")
    generations.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    final = generations / generation
    committed = False
    try:
        shutil.copytree(staging, final)
        (final / "plot_owner.json").write_text(json.dumps({"source": metadata["source"],
                                                           "generation": generation}) + "\n")
        for file in final.rglob("*"):
            if file.is_file():
                with file.open("rb") as source:
                    os.fsync(source.fileno())
        directories = [final, *(path for path in final.rglob("*") if path.is_dir())]
        for subdirectory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            descriptor = os.open(subdirectory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(generations, os.O_RDONLY | os.O_DIRECTORY)
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
    remove_obsolete_plots(output, files, previous)
    for directory in generations.iterdir():
        if directory == final or directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / "plot_owner.json"
        if (re.fullmatch(r"[0-9a-f]{32}", directory.name) and marker.is_file()
                and json.loads(marker.read_text()).get("source") == metadata["source"]):
            shutil.rmtree(directory)


def main(argv=None):
    """Generate both grouping views and record exactly what each plot represents."""
    args = parse_args(argv)
    try:
        source = args.run.resolve()
        output = args.output_dir.resolve(strict=False)
        if (output != source / "plots"
                or args.run.is_symlink() or args.output_dir.is_symlink()):
            raise ValueError("plots must be saved in the selected raw run's plots directory")
        rows, manifests = load_rows(source)
        versions = sorted({row["fio_version"] for row in rows})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=args.output_dir) as temporary_dir:
            staging = Path(temporary_dir)
            files = [plot_workload_by_ost(rows, workload, version, staging)
                     for workload in WORKLOAD_ORDER for version in versions
                     if any(row["workload"] == workload and row["fio_version"] == version for row in rows)]
            metadata = {
                "source": str(source), "manifests": [str(path) for path in manifests],
                "measurements": len(rows), "fio_versions": versions, "plots": files,
                "semantics": {
                    "sampling_unit": "each row is one OST; OST measurements are never combined",
                    "figures": "one bandwidth figure per access pattern and FIO version",
                    "marks": "bars show OST medians, ticks show repetitions, and whiskers show min-max",
                    "media": "HDD and NVMe are identified by color only, not aggregated",
                    "axes": "OSTs are horizontal; bandwidth in MiB/s is vertical",
                    "references": "dotted lines average per-OST medians within one FIO version and media only",
                },
            }
            publish_plots(staging, args.output_dir, files, metadata)
        print(f"Created {len(files)} plots under {args.output_dir}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot visualize results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
