#!/usr/bin/env python3
"""Validate completed mdtest raw output and plot native phase summaries."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import run_mdtest


OPERATIONS = (
    "Directory creation", "Directory stat", "Directory rename",
    "Directory removal", "File creation", "File stat", "File read",
    "File removal", "Tree creation", "Tree removal",
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ROW = re.compile(r"^\s*(.*?)\s+(" + NUMBER + r")\s+(" + NUMBER
                 + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s*$")
COLORS = {"anjuna2": "#315A7D", "anjuna3": "#D08B32", "dual": "#4E7D5B"}
LAYOUT_MARKERS = {"flat": "o", "per_rank": "^"}


def load(path):
    """Load runner JSON evidence without following symlinks."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_table(output, title):
    """Parse mdtest's Max/Min/Mean/Std Dev rows below one summary heading."""
    headings = [m.start() for m in re.finditer(r"^SUMMARY " + re.escape(title),
                                                output, re.MULTILINE)]
    if len(headings) != 1:
        raise ValueError(f"expected one SUMMARY {title} table")
    start = headings[0]
    end = re.search(r"^SUMMARY ", output[start + 1:], re.MULTILINE)
    block = output[start:] if end is None else output[start:start + 1 + end.start()]
    rows = {}
    for line in block.splitlines()[1:]:
        match = ROW.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        if name in OPERATIONS:
            if name in rows:
                raise ValueError(f"duplicate {title} row for {name}")
            values = tuple(float(value) for value in match.groups()[1:])
            if (not all(math.isfinite(value) for value in values)
                    or min(values[:3]) < 0 or values[3] < 0
                    or (title.startswith("rate") and min(values[:3]) <= 0)
                    or not values[1] <= values[2] <= values[0]):
                raise ValueError(f"invalid {title} values for {name}")
            rows[name] = {"max": values[0], "min": values[1],
                          "mean": values[2], "stddev": values[3]}
    if set(rows) != set(OPERATIONS):
        missing = sorted(set(OPERATIONS) - set(rows))
        raise ValueError(f"{title} table is missing phases: {missing}")
    return rows


def validate_case(run, unit, items, storage_pools=None):
    """Validate one successful, cleaned case and return its native summaries."""
    if (run.joinpath("cases").is_symlink()
            or not re.fullmatch(r"(?:pilot-\d{2}|r\d{2}-(?:anjuna2|anjuna3|dual)-n(?:1|4|16)-(?:flat|per_rank))(?:-(?:hdd|ssd))?",
                                unit.get("id", ""))):
        raise ValueError("case path is symlinked or has an unexpected ID")
    case = run / "cases" / unit["id"]
    if case.is_symlink():
        raise ValueError(f"{unit['id']}: case directory is a symlink")
    result = load(case / "result.json")
    if (result.get("returncode") != 0 or result.get("interrupted")
            or result.get("cleanup") != "completed"):
        raise ValueError(f"{unit['id']}: case did not succeed and clean up")
    command = load(case / "command.json")
    work = (run_mdtest.NAMESPACE / run.name / unit["id"] / "work")
    if command != run_mdtest.command(unit, work, items):
        raise ValueError(f"{unit['id']}: saved command differs from planned host/rank/workload")
    if storage_pools is not None:
        placement = load(case / "placement.json")
        run_mdtest.validate_placement(placement, unit, work, storage_pools)
        after_pools_path = case / "after_storage_pools.json"
        if (after_pools_path.is_symlink() or not after_pools_path.is_file()
                or result.get("storage_pools_unchanged") is not True):
            raise ValueError(f"{unit['id']}: missing or failed post-case pool verification")
        after_pools = load(after_pools_path)
        after_pools.pop("at", None)
        if after_pools != storage_pools:
            raise ValueError(f"{unit['id']}: storage-pool membership changed during case")
    stdout = case / "stdout.txt"
    if stdout.is_symlink() or not stdout.is_file():
        raise ValueError(f"{unit['id']}: missing or symlinked stdout")
    output = stdout.read_text(encoding="utf-8")
    if hashlib.sha256(stdout.read_bytes()).hexdigest() != result.get("stdout_sha256"):
        raise ValueError(f"{unit['id']}: stdout checksum differs from result.json")
    stderr = case / "stderr.txt"
    if (stderr.is_symlink() or not stderr.is_file()
            or hashlib.sha256(stderr.read_bytes()).hexdigest() != result.get("stderr_sha256")):
        raise ValueError(f"{unit['id']}: stderr checksum differs from result.json")
    return {"unit": unit, "items_per_rank": items,
            "stderr_nonempty": stderr.stat().st_size > 0,
            "rate": parse_table(output, "rate (in ops/sec):"),
            "time": parse_table(output, "time (in ms/op):")}


def load_run(run):
    """Require every planned case to have intact native output before plotting."""
    run = Path(run)
    if run.is_symlink() or not run.is_dir():
        raise ValueError(f"missing or symlinked run directory: {run}")
    plan = load(run / "plan.json")
    if (not isinstance(plan, dict) or plan.get("mode") not in ("pilot", "full")
            or not isinstance(plan.get("units"), list)):
        raise ValueError("unrecognized metadata run plan")
    targeted = "storage_pools" in plan
    expected = ({"pilot": (8, 1000), "full": (180, 10000)} if targeted
                else {"pilot": (4, 1000), "full": (90, 10000)})[plan["mode"]]
    planned_units = (run_mdtest.units(plan["mode"] == "pilot") if targeted
                     else run_mdtest.legacy_units(plan["mode"] == "pilot"))
    if targeted:
        run_mdtest.validate_storage_pools(plan["storage_pools"])
    if (plan["units"] != planned_units or len(plan["units"]) != expected[0]
            or plan.get("items_per_rank") != expected[1]
            or not isinstance(plan.get("mpirun"), dict)
            or not isinstance(plan.get("mdtest"), dict)
            or plan["mpirun"].get("path") != str(run_mdtest.MPIRUN)
            or plan["mdtest"].get("path") != str(run_mdtest.MDTEST)):
        raise ValueError("plan does not match the documented pilot/full matrix")
    cases = [validate_case(run, unit, plan["items_per_rank"],
                           plan["storage_pools"] if targeted else None)
             for unit in plan["units"]]
    return run, plan, cases


def compact_config_label(place, ranks, layout):
    """Keep the 18 full-matrix x-axis labels distinct without crowding."""
    placement = {"anjuna2": "a2", "anjuna3": "a3", "dual": "dual"}[place]
    layout_code = "R" if layout == "per_rank" else "F"
    return f"{placement}-{ranks}{layout_code}"


def time_per_operation_ms(rate_ops_per_second):
    """Convert one invocation's aggregate phase rate into milliseconds/op."""
    if not math.isfinite(rate_ops_per_second) or rate_ops_per_second <= 0:
        raise ValueError("operation rate must be finite and positive")
    return 1000.0 / rate_ops_per_second


def mean_time_per_operation_ms(rates_ops_per_second):
    """Average each invocation's throughput-normalized time per operation."""
    return statistics.fmean(time_per_operation_ms(rate)
                            for rate in rates_ops_per_second)


def plot_metric(cases, metric, field, ylabel, path, run_id, storage_pools=None):
    """Plot one mean per configuration, averaging the independent invocations."""
    configs = {}
    for case in cases:
        unit = case["unit"]
        key = (unit["placement"], unit["ranks"], unit["layout"])
        target_class = unit.get("target_class")
        configs.setdefault((key, target_class), []).append(case)
    ordered = [(place, ranks, layout) for place in ("anjuna2", "anjuna3", "dual")
               for ranks in (1, 4, 16) for layout in ("flat", "per_rank")
               if any((place, ranks, layout) == key for key, _ in configs)]
    target_classes = (("hdd", "ssd") if storage_pools is not None else (None,))
    fig, axes = plt.subplots(5, 2, figsize=(19, 17))
    for index, operation in enumerate(OPERATIONS):
        ax = axes.flat[index]
        plotted = 0
        for position, key in enumerate(ordered, 1):
            place, _, layout = key
            for target_class in target_classes:
                samples = configs.get((key, target_class), [])
                if not samples:
                    continue
                # One point per config and target: mean across separate runs.
                if metric == "time":
                    # The installed mdtest `time (in ms/op)` table is phase
                    # duration with misleading units. Use reciprocal ops/s;
                    # do not divide by -n again.
                    rates = [case["rate"][operation]["mean"] for case in samples]
                    value = mean_time_per_operation_ms(rates)
                else:
                    values = [case[metric][operation][field] for case in samples]
                    value = statistics.fmean(values)
                if value <= 0:
                    continue
                offset = (-.12 if target_class == "hdd" else .12) if target_class else 0
                face = "none" if target_class == "hdd" else COLORS[place]
                ax.plot(position + offset, value, marker=LAYOUT_MARKERS[layout],
                        color=COLORS[place], markerfacecolor=face, markersize=5,
                        linestyle="none")
                plotted += 1
        ax.set_title(operation)
        ax.set_yscale("log")
        if not plotted:
            ax.text(.5, .5, "no positive values",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks(range(1, len(ordered) + 1))
        ax.set_xticklabels([compact_config_label(p, r, layout)
                            for p, r, layout in ordered], fontsize=5.5,
                           rotation=35, ha="right")
        ax.grid(axis="y", alpha=.25)
    handles = [Line2D([0], [0], color=color, marker="o", linestyle="none", label=place)
               for place, color in COLORS.items()]
    handles += [Line2D([0], [0], color="black", marker=marker, linestyle="none",
                       label=layout.replace("_", " "))
                for layout, marker in LAYOUT_MARKERS.items()]
    if storage_pools is not None:
        handles += [Line2D([0], [0], color="black", marker="o", linestyle="none",
                           markerfacecolor="none", label=f"HDD target {storage_pools['classes']['hdd']['target_id']}"),
                    Line2D([0], [0], color="black", marker="o", linestyle="none",
                           markerfacecolor="black", label=f"SSD target {storage_pools['classes']['ssd']['target_id']}")]
    metric_title = "mean time per operation" if metric == "time" else "rate"
    metric_note = ("time/op = mean per-invocation reciprocal of mdtest aggregate ops/s"
                   if metric == "time" else "rate = mean mdtest aggregate operations/s")
    pool_note = (f"HDD/SSD singleton pools: {storage_pools['classes']['hdd']['name']} / "
                 f"{storage_pools['classes']['ssd']['name']}" if storage_pools else "")
    fig.suptitle(f"BeeGFS mdtest {metric_title}: {run_id}\n"
                 f"one point per configuration, averaged across invocations; {metric_note}; log scale\n"
                 f"x: a2/a3/dual, rank-count + F(flat)/R(per-rank); {pool_note}; file read is open/close",
                 fontsize=13)
    fig.supylabel(ylabel)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945),
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .02, 1, .91))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    """Validate a completed run and plot rate and per-operation time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run, plan, cases = load_run(args.run_dir)
        plots = run / "plots"
        if plots.is_symlink():
            raise ValueError(f"refusing symlinked plot directory: {plots}")
        plots.mkdir(exist_ok=True)
        rate_plot = plots / "mean_rate.png"
        time_plot = plots / "mean_time_per_op.png"
        obsolete_plot = plots / "mean_phase_duration_seconds.png"
        if any(path.is_symlink() for path in (rate_plot, time_plot, obsolete_plot)):
            raise ValueError("refusing symlinked plot output")
        plot_metric(cases, "rate", "mean", "Mean rate (ops/s, log scale)",
                    rate_plot, run.name, plan.get("storage_pools"))
        plot_metric(cases, "time", "mean", "Mean time per operation (ms/op, log scale)",
                    time_plot, run.name, plan.get("storage_pools"))
        # Remove only the interim, visualizer-generated phase-duration plot.
        if obsolete_plot.exists():
            if not obsolete_plot.is_file():
                raise ValueError(f"refusing non-file obsolete plot output: {obsolete_plot}")
            obsolete_plot.unlink()
        print(f"{plan['mode']}: plotted {len(cases)}/{len(plan['units'])} validated cases")
        stderr_cases = sum(case["stderr_nonempty"] for case in cases)
        if stderr_cases:
            print(f"warning: {stderr_cases} case(s) had nonempty stderr; inspect raw stderr.txt")
        print(f"plots: {plots}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Metadata results not plottable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
