#!/usr/bin/env python3
"""Plot one completed run_cache_read.py run from its own raw artifacts.

Read-only with respect to the measurement: this program opens results.json and
each case directory, revalidates the native IOR summary and the recorded
traffic counters, and writes PNG files below <run>/plots/.
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZE = 8 * 1024**3
STATES = ("backend", "server_ram", "client_ram")
COLORS = {"backend": "#315A7D", "server_ram": "#D08B32", "client_ram": "#4E7D5B"}
MEASURED = ("client_network", "server_network", "backend_ratio")
VERIFICATION = "traffic_counters_v1"


def load(path):
    """Return the JSON object at path, refusing symbolic links."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_native(ior, result):
    """Return the IOR read rate after checking it matches the cache protocol.

    Requires one POSIX read of the whole 8-GiB file by one rank with 1-MiB
    transfers, and the same rate the case recorded as its result.
    """
    if ior.get("Version") != "4.1.0+dev":
        raise ValueError(f"unexpected IOR version {ior.get('Version')!r}")
    rows = ior.get("summary")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected exactly one IOR summary phase")
    row = rows[0]
    if (row.get("API") != "POSIX" or row.get("operation") != "read"
            or row.get("numTasks") != 1 or row.get("blockSize") != SIZE
            or row.get("transferSize") != 1024**2 or row.get("xsizeMiB") != 8192):
        raise ValueError("IOR read does not match the 8-GiB POSIX protocol")
    rate = row.get("bwMeanMIB")
    seconds = row.get("MeanTime")
    if (not isinstance(rate, (int, float)) or not 0 < rate < float("inf")
            or not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf")):
        raise ValueError("IOR reported an invalid rate or duration")
    if abs(rate * seconds - 8192) > 8192 * .02:
        raise ValueError("IOR rate and duration disagree with 8192 MiB")
    if abs(rate - result["MiB_per_second"]) > max(.01, rate * .001):
        raise ValueError("case result rate differs from the native IOR summary")
    return rate


def check_command(argv):
    """Require the documented single-rank 8-GiB read without O_DIRECT."""
    if not isinstance(argv, list) or "--posix.odirect" in argv:
        raise ValueError("recorded command is not a list or used O_DIRECT")
    for option, value in (("-a", "POSIX"), ("-b", "8g"), ("-t", "1m"),
                          ("-s", "1"), ("-i", "1"), ("-np", "1")):
        if option not in argv or argv[argv.index(option) + 1] != value:
            raise ValueError(f"recorded command lacks {option} {value}")
    for flag in ("-r", "-E", "-k", "-g"):
        if flag not in argv:
            raise ValueError(f"recorded command lacks {flag}")


def ratios(counters):
    """Return traffic ratios over IOR's 8-GiB logical read from raw counters."""
    before, after = counters["before"], counters["after"]
    return {"network_ratio": (after["client_network"] - before["client_network"]) / SIZE,
            "server_network_ratio": (after["server_network"] - before["server_network"]) / SIZE,
            "backend_ratio": (after["device_read"] - before["device_read"]) / SIZE}


def expected_label(mode, state, network, server, backend, cached, quiet, verification):
    """Return the state supported by current or preserved legacy evidence."""
    if not quiet:
        return "unverified"
    traffic_rules = {"backend": network >= .8 and server >= .8 and backend >= .8,
                     "server_ram": network >= .8 and server >= .8 and backend <= .2,
                     "client_ram": network <= .2 and server <= .2 and backend <= .2}
    if verification == VERIFICATION or (verification is None and mode == "buffered"):
        rules = traffic_rules
    elif verification is None and mode == "native" and cached is None:
        # Transitional native pilots recorded traffic and labels without a
        # verification marker or a raw pre-read idle sample. Validate their
        # measured traffic, but retain that evidence limit in documentation.
        rules = traffic_rules
    elif verification is None and mode == "native" and isinstance(cached, (int, float)):
        # Preserve the labels attached to investigation pilots made with the
        # destructive residency probe. It is not accepted for new results.
        rules = {"backend": (network >= .8 and server >= .8 and backend >= .8 and cached <= .1),
                 "server_ram": (network >= .8 and server >= .8 and backend <= .2 and cached <= .1),
                 "client_ram": (network <= .2 and server <= .2 and backend <= .2 and cached >= .95)}
    else:
        raise ValueError("native result lacks a recognized verification method")
    if state not in rules or (mode == "buffered" and state == "client_ram"):
        raise ValueError(f"state {state!r} is invalid for mode {mode!r}")
    return state if rules[state] else "unverified"


def load_cases(run):
    """Return validated case records for run, ordered as the runner wrote them."""
    run = Path(run)
    if run.is_symlink() or not run.is_dir():
        raise ValueError(f"run directory not found: {run}")
    owner = load(run / "owner.json")
    if owner.get("run_id") != run.name:
        raise ValueError("owner.json belongs to another run")
    results = load(run / "results.json")
    if not isinstance(results, list) or not results:
        raise ValueError("results.json is empty or malformed")
    cases = []
    for index, result in enumerate(results, 1):
        name = f"{index:02d}-{result['medium'].lower()}-{result['intended']}"
        case = run / name
        record = {"name": name, "medium": result["medium"], "state": result["intended"],
                  "mode": result["mode"], "recorded": result["achieved"],
                  "verification": result.get("verification")}
        if result["mode"] != owner.get("mode"):
            raise ValueError(f"{name}: case mode differs from owner.json")
        record["rate"] = check_native(load(case / "ior.json"), result)
        check_command(load(case / "command.json"))
        raw_counters = load(case / "counters.json")
        observed = ratios(raw_counters)
        for key, value in observed.items():
            if abs(value - result[key]) > 1e-6:
                raise ValueError(f"{name}: {key} differs from the raw counters")
        cached = result.get("client_residency")
        if record["verification"] == VERIFICATION:
            if "client_residency" in result:
                raise ValueError(f"{name}: current traffic evidence includes a legacy residency field")
            idle, before = raw_counters["idle"], raw_counters["before"]
            quiet_from_raw = all(0 <= before[key] - idle[key] < SIZE // 100
                                 for key in before)
            if result.get("quiet_before") is not quiet_from_raw:
                raise ValueError(f"{name}: quiet_before differs from the raw counters")
        record.update(observed, cached=cached, quiet=result.get("quiet_before") is True)
        record["label"] = expected_label(result["mode"], record["state"], observed["network_ratio"],
                                          observed["server_network_ratio"], observed["backend_ratio"],
                                          cached, record["quiet"], record["verification"])
        if record["label"] != record["recorded"]:
            raise ValueError(f"{name}: recorded label {record['recorded']!r} "
                             f"does not follow from the evidence ({record['label']!r})")
        cases.append(record)
    return owner, cases


def groups(cases):
    """Return [(medium, state, [case records])] in first-measured order."""
    ordered = []
    for case in cases:
        key = (case["medium"], case["state"])
        if key not in ordered:
            ordered.append(key)
    return [(medium, state, [case for case in cases
                             if (case["medium"], case["state"]) == (medium, state)])
            for medium, state in ordered]


def save(fig, path, manifest_path, owned):
    """Write fig to path below the run's own plots directory, then update the manifest."""
    if path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("refusing to write through a symlink")
    if path.exists() and path.name not in owned.get("plots", []):
        raise ValueError(f"{path} was not written by this visualizer")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as target:
            fig.savefig(target, format="png", dpi=150)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    plt.close(fig)
    names = sorted(set(owned.get("plots", [])) | {path.name})
    temporary = manifest_path.with_name(f".plot_manifest.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as target:
        json.dump({"run_id": manifest_path.parent.parent.name, "plots": names}, target, indent=2)
        target.write("\n")
    os.replace(temporary, manifest_path)
    return {**owned, "plots": names}


def plot_throughput(cases, path, manifest_path, owned):
    """Draw every measured read rate as a dot, with its group's min-max and median.

    A box plot is not used: each group holds only five repetitions, so its
    quartiles and whiskers would be interpolated from almost no data and would
    hide the worst repetition. Each dot is one IOR read.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for position, (_, state, members) in enumerate(groups(cases), 1):
        rates = sorted(case["rate"] for case in members)
        color = COLORS[state]
        ax.plot([position, position], [rates[0], rates[-1]], color=color, linewidth=2,
                solid_capstyle="butt", zorder=1)
        middle = median(rates)
        ax.plot([position - .16, position + .16], [middle, middle], color="black",
                linewidth=2.5, zorder=3)
        for offset, rate in zip((-.16, -.08, 0, .08, .16) * 4, rates):
            ax.plot(position + offset, rate, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=.8, zorder=2)
        ax.annotate(f"{rates[0]:.0f}-{rates[-1]:.0f}", (position + .34, rates[0]),
                    fontsize=8, color=color, va="center")
    ax.set_xticks(range(1, len(groups(cases)) + 1))
    ax.set_xticklabels([f"{medium}\n{state}  (n={len(members)})"
                        for medium, state, members in groups(cases)])
    ax.set_xlim(.5, len(groups(cases)) + .5)
    ax.set_ylabel("IOR read throughput (MiB/s)")
    ax.set_title("Cache run: one dot per measured IOR read\n"
                 "vertical line = min to max, black bar = median", fontsize=11)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, path, manifest_path, owned)


def plot_evidence(cases, path, manifest_path, owned):
    """Draw the three traffic ratios for every case, one panel per configuration.

    One panel per medium/state group keeps each case's three ratios adjacent,
    which a single 20-case axis cannot do. The 0.8 and 0.2 lines are the
    thresholds run_cache_read.py applies before it accepts a path.
    """
    layout = groups(cases)
    columns = 2 if len(layout) > 1 else 1
    rows = -(-len(layout) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.4 * rows),
                             squeeze=False, sharey=True)
    series = (("network_ratio", "client received"), ("server_network_ratio", "server sent"),
              ("backend_ratio", "device read"))
    for panel, ((medium, state, members), ax) in enumerate(zip(layout, axes.flat)):
        for slot, (key, title) in enumerate(series):
            values = [case[key] for case in members]
            ax.bar([index + (slot - 1) * .3 for index in range(len(members))], values,
                   width=.28, label=title)
        ax.axhline(.8, color="black", linestyle="--", linewidth=1)
        ax.axhline(.2, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(range(len(members)))
        ax.set_xticklabels([case["name"].split("-", 1)[0] for case in members])
        ax.set_ylim(0, 1.25)
        ax.set_title(f"{medium} {state}: every case {'/'.join(sorted({case['label'] for case in members}))}",
                     fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
    for panel in range(len(layout), rows * columns):
        axes.flat[panel].set_visible(False)
    axes.flat[0].set_ylabel("traffic / 8-GiB logical read")
    axes.flat[0].legend(fontsize=8, loc="upper left", ncol=3)
    fig.suptitle("Path evidence per case: dashed 0.8 and dotted 0.2 are the protocol thresholds",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, path, manifest_path, owned)


def plot_residency(cases, path, manifest_path, owned):
    """Draw the failed legacy probe values as investigation evidence only."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for position, (_, _, members) in enumerate(groups(cases), 1):
        values = [case["cached"] for case in members if case["cached"] is not None]
        if values:
            ax.plot([position] * len(values), values, "o", color=COLORS[members[0]["state"]])
    ax.axhline(.95, color="grey", linestyle="--", linewidth=1)
    ax.axhline(.1, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(range(1, len(groups(cases)) + 1))
    ax.set_xticklabels([f"{medium} {state}" for medium, state, _ in groups(cases)],
                       rotation=20, ha="right")
    ax.set_ylabel("legacy mincore probe result")
    ax.set_ylim(-.05, 1.05)
    ax.set_title("Legacy destructive probe diagnostic; not valid residency evidence")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return save(fig, path, manifest_path, owned)


def summarize(owner, cases):
    """Return printed lines describing the run and each measured group."""
    labels = [case["label"] for case in cases]
    lines = [f"run {owner['run_id']}: mode={owner['mode']} pilot={owner['pilot']} "
             f"cases={len(cases)} verified={labels.count('server_ram') + labels.count('backend') + labels.count('client_ram')}"
             f" unverified={labels.count('unverified')}"]
    for medium, state, members in groups(cases):
        rates = sorted(case["rate"] for case in members)
        middle = median(rates)
        lines.append(f"  {medium:3} {state:10} n={len(rates)} "
                     f"min={rates[0]:.1f} median={middle:.1f} max={rates[-1]:.1f} MiB/s")
    return lines


def main(argv=None):
    """Plot the run directory named on the command line; return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="results/microbenchmarks/runs/<run-id>")
    args = parser.parse_args(argv)
    try:
        owner, cases = load_cases(args.run)
        plots = args.run / "plots"
        plots.mkdir(exist_ok=True)
        manifest = plots / "plot_manifest.json"
        current = load(manifest) if manifest.is_file() else {}
        if current and current.get("run_id") != owner["run_id"]:
            raise ValueError("plots/plot_manifest.json belongs to another run")
        current = plot_throughput(cases, plots / "throughput.png", manifest, current)
        current = plot_evidence(cases, plots / "traffic_evidence.png", manifest, current)
        if any(case["cached"] is not None for case in cases):
            plot_residency(cases, plots / "client_residency.png", manifest, current)
        print("\n".join(summarize(owner, cases)))
        print(f"\nplots written to {plots}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Cache results not plottable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
