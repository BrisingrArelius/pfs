#!/usr/bin/env python3
"""Validate and plot completed cache read or write runs from raw artifacts.

Read-only with respect to the measurement: this program opens results.json and
each case directory, revalidates its protocol and recorded traffic counters,
and writes per-run PNG files below <run>/plots/. Multiple runs of one
operation can be passed together to produce a combined comparison.
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
TARGET_IDS = {"HDD": 101, "SSD": 104}
STATES = ("backend", "server_ram", "client_ram")
COLORS = {"backend": "#315A7D", "server_ram": "#D08B32", "client_ram": "#4E7D5B"}
WRITE_COLORS = {"server_disk": "#315A7D", "server_ram": "#D08B32",
                "client_ram": "#4E7D5B"}
MEASURED = ("client_network", "server_network", "backend_ratio")
VERIFICATION = "traffic_counters_v1"
WRITE_VERIFICATION = "write_traffic_v1"
CLIENT_SIZE = 1024**3
NETWORK_PEAK_MIB_S = 2_500_000_000 / 8 / 1024**2


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


def check_write_command(argv):
    """Require the runner's single-rank 8-GiB POSIX write command."""
    if not isinstance(argv, list) or "--posix.odirect" in argv:
        raise ValueError("recorded write command is not a list or used O_DIRECT")
    for option, value in (("-a", "POSIX"), ("-b", "8g"), ("-t", "1m"),
                          ("-s", "1"), ("-i", "1"), ("-np", "1")):
        if option not in argv or argv[argv.index(option) + 1] != value:
            raise ValueError(f"recorded write command lacks {option} {value}")
    for flag in ("-w", "-E", "-k", "-g", "-e"):
        if flag not in argv:
            raise ValueError(f"recorded write command lacks {flag}")


def check_write_ior(ior, result):
    """Validate the 8-GiB write IOR summary and return its rate and duration."""
    rows = ior.get("summary")
    if ior.get("Version") != "4.1.0+dev" or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected one IOR 4.1.0+dev write summary")
    row = rows[0]
    if (row.get("API") != "POSIX" or row.get("operation") != "write"
            or row.get("numTasks") != 1 or row.get("blockSize") != SIZE
            or row.get("transferSize") != 1024**2 or row.get("xsizeMiB") != 8192):
        raise ValueError("IOR write does not match the 8-GiB POSIX protocol")
    rate, seconds = row.get("bwMeanMIB"), row.get("MeanTime")
    if (not isinstance(rate, (int, float)) or not 0 < rate < float("inf")
            or not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf")):
        raise ValueError("IOR reported an invalid write rate or duration")
    if (abs(rate * seconds - 8192) > 8192 * .02
            or abs(rate - result["MiB_per_second"]) > max(.01, rate * .001)
            or abs(seconds - result["seconds"]) > max(.01, seconds * .001)):
        raise ValueError("IOR write rate or duration differs from the recorded result")
    return rate, seconds


def close_enough(actual, expected):
    """Compare stored ratios while allowing JSON floating-point roundoff."""
    return isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual - expected) <= 1e-6


def load_write_cases(run):
    """Validate write protocol, counters and labels; return plot-ready cases."""
    run = Path(run)
    if run.is_symlink() or not run.is_dir():
        raise ValueError(f"run directory not found: {run}")
    owner = load(run / "owner.json")
    if owner.get("run_id") != run.name or owner.get("operation") != "write":
        raise ValueError("owner.json is not a write run for this directory")
    mode, remote_fsync = owner.get("mode"), owner.get("remote_fsync")
    if mode not in ("buffered", "native") or remote_fsync not in ("true", "false"):
        raise ValueError("write owner has an invalid mode or fsync policy")
    results = load(run / "results.json")
    if not isinstance(results, list) or not results:
        raise ValueError("results.json is empty or malformed")
    pilot = owner.get("pilot") is True
    states = (["server_disk"] if remote_fsync == "true" else ["server_ram"])
    if mode == "native" and remote_fsync == "true":
        states.append("client_ram")
    expected_count = len(states) * (1 if pilot else 5) * 2
    if len(results) != expected_count:
        raise ValueError(f"expected {expected_count} write cases, found {len(results)}")

    cases = []
    for index, result in enumerate(results, 1):
        medium, state = result.get("medium"), result.get("intended")
        if medium not in TARGET_IDS or state not in states:
            raise ValueError(f"case {index}: unexpected medium or write state")
        name = f"{index:02d}-{medium.lower()}-{state}"
        folder = run / name
        recorded = load(folder / "result.json")
        if recorded != result:
            raise ValueError(f"{name}: case result differs from results.json")
        if (result.get("operation") != "write" or result.get("mode") != mode
                or result.get("remote_fsync") != remote_fsync
                or result.get("target") != TARGET_IDS[medium]
                or result.get("verification") != WRITE_VERIFICATION):
            raise ValueError(f"{name}: result metadata differs from owner or target")

        size = CLIENT_SIZE if state == "client_ram" else SIZE
        if result.get("bytes") != size:
            raise ValueError(f"{name}: unexpected logical write size")
        idle = load(folder / "idle_counters.json")
        before_idle, before = idle.get("idle", {}), idle.get("before", {})
        counter_names = ("client_sent", "server_received", "device_written")
        if any(key not in before_idle or key not in before for key in counter_names):
            raise ValueError(f"{name}: idle counter snapshot is incomplete")
        quiet = all(0 <= before[key] - before_idle[key] < size // 100
                    for key in counter_names)
        if result.get("quiet_before") is not quiet:
            raise ValueError(f"{name}: quiet_before differs from raw counters")

        raw = load(folder / "counters.json")
        after_key = "after_sync" if state == "client_ram" else "after"
        after = raw.get(after_key, {})
        if any(key not in raw.get("before", {}) or key not in after for key in counter_names):
            raise ValueError(f"{name}: measurement counter snapshot is incomplete")
        observed = {f"{key}_ratio": (after[key] - raw["before"][key]) / size
                    for key in counter_names}
        for key, value in observed.items():
            if value < 0 or not close_enough(result.get(key), value):
                raise ValueError(f"{name}: {key} differs from raw counters")

        if state == "client_ram":
            workload = load(folder / "workload.json")
            timing = load(folder / "posix.json")
            if (workload.get("API") != "POSIX" or workload.get("operation") != "write"
                    or workload.get("timed_region") != "write calls only; fsync/close excluded"
                    or workload.get("bytes") != CLIENT_SIZE
                    or workload.get("transferSize") != 1024**2
                    or timing.get("bytes") != CLIENT_SIZE):
                raise ValueError(f"{name}: client write workload differs from protocol")
            seconds = timing.get("write_seconds")
            sync_seconds = timing.get("fsync_seconds")
            if (not isinstance(seconds, (int, float)) or seconds <= 0
                    or not isinstance(sync_seconds, (int, float)) or sync_seconds < 0
                    or not close_enough(result.get("seconds"), seconds)
                    or not close_enough(result.get("MiB_per_second"), 1024 / seconds)):
                raise ValueError(f"{name}: client write timing differs from raw timing")
            timed = (raw.get("timed_end_tx", -1) - raw.get("timed_start_tx", -1)) / size
            if timed < 0 or not close_enough(result.get("timed_client_sent_ratio"), timed):
                raise ValueError(f"{name}: timed client traffic differs from raw counters")
            rate = result["MiB_per_second"]
        else:
            rate, _ = check_write_ior(load(folder / "ior.json"), result)
            check_write_command(load(folder / "command.json"))

        network_ok = (observed["client_sent_ratio"] >= .8
                      and observed["server_received_ratio"] >= .8)
        device_ok = observed["device_written_ratio"] >= .8
        verified = quiet and network_ok and (
            (state == "server_ram" and remote_fsync == "false")
            or (state == "server_disk" and remote_fsync == "true" and device_ok)
            or (state == "client_ram" and mode == "native" and remote_fsync == "true"
                and device_ok and result["timed_client_sent_ratio"] <= .2))
        label = state if verified else "unverified"
        if result.get("achieved") != label:
            raise ValueError(f"{name}: achieved label disagrees with raw write evidence")
        cases.append({"name": name, "medium": medium, "state": state, "rate": rate,
                      "mode": mode, "remote_fsync": remote_fsync,
                      "label": label, **observed,
                      "timed_client_sent_ratio": result.get("timed_client_sent_ratio")})
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


def add_network_peak(ax):
    """Show the captured 2.5-Gbit/s link's nominal line-rate ceiling."""
    return ax.axhline(NETWORK_PEAK_MIB_S, color="#555555", linestyle=":",
                      linewidth=1.4, label="Nominal 2.5-Gbit/s line rate (~298 MiB/s)",
                      zorder=0)


def save(fig, path, manifest_path, owned, manifest_id=None):
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
        json.dump({"run_id": manifest_id or manifest_path.parent.parent.name,
                   "plots": names}, target, indent=2)
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
    add_network_peak(ax)
    ax.set_title("Cache run: one dot per measured IOR read\n"
                 "vertical line = min to max, black bar = median", fontsize=11)
    ax.legend(fontsize=8, loc="best")
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


def write_state_name(state):
    """Explain the write benchmark's result labels without implying pure RAM speed."""
    return {"server_disk": "disk-policy fsync", "server_ram": "server-cache ack",
            "client_ram": "client write-call acceptance"}[state]


def plot_write_throughput(cases, owner, path, manifest_path, owned):
    """Plot each IOR write rate or timed native client write-call rate."""
    layout = groups(cases)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for position, (_, state, members) in enumerate(layout, 1):
        rates = sorted(case["rate"] for case in members)
        color = WRITE_COLORS[state]
        ax.plot([position, position], [rates[0], rates[-1]], color=color, linewidth=2,
                solid_capstyle="butt", zorder=1)
        middle = median(rates)
        ax.plot([position - .16, position + .16], [middle, middle], color="black",
                linewidth=2.5, zorder=3)
        offsets = [(index - (len(rates) - 1) / 2) * .07 for index in range(len(rates))]
        for offset, rate in zip(offsets, rates):
            ax.plot(position + offset, rate, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=.8, zorder=2)
    ax.set_xticks(range(1, len(layout) + 1))
    short_state = {"server_disk": "disk fsync", "server_ram": "cache ack",
                   "client_ram": "timed client writes"}
    ax.set_xticklabels([f"{medium}\n{short_state[state]}\n(n={len(members)})"
                        for medium, state, members in layout], fontsize=9)
    ax.set_xlim(.5, len(layout) + .5)
    ax.set_yscale("log")
    ax.set_ylabel("MiB/s (log scale)")
    add_network_peak(ax)
    ax.set_title(f"Cache write: mode={owner['mode']}, tuneRemoteFSync={owner['remote_fsync']}\n"
                 "8-GiB IOR includes fsync; client case times 1-GiB write calls, not fsync\n"
                 "dots = cases; vertical line = min–max; black bar = median", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, path, manifest_path, owned)


def plot_write_evidence(cases, owner, path, manifest_path, owned):
    """Plot recorded network/device ratios and timed client traffic evidence."""
    layout = groups(cases)
    columns = 2 if len(layout) > 1 else 1
    rows = -(-len(layout) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.5 * rows),
                             squeeze=False, sharey=True)
    for (medium, state, members), ax in zip(layout, axes.flat):
        series = [("client_sent_ratio", "client sent by completion"),
                  ("server_received_ratio", "server received"),
                  ("device_written_ratio", "device written")]
        if state == "client_ram":
            series.append(("timed_client_sent_ratio", "client sent during timed writes"))
        for slot, (key, label) in enumerate(series):
            values = [case.get(key) for case in members]
            positions = [index + (slot - (len(series) - 1) / 2) * .18
                         for index in range(len(members))]
            ax.bar(positions, [value if value is not None else 0 for value in values],
                   width=.17, label=label)
        ax.axhline(.8, color="black", linestyle="--", linewidth=1)
        ax.axhline(.2, color="black", linestyle=":", linewidth=1)
        maximum = max([value for case in members for value in
                       (case.get(key) for key, _ in series) if value is not None] + [1.0])
        ax.set_ylim(0, max(1.25, maximum * 1.12))
        ax.set_xticks(range(len(members)))
        ax.set_xticklabels([case["name"].split("-", 1)[0] for case in members])
        ax.set_title(f"{medium} {write_state_name(state)}: "
                     f"{len([case for case in members if case['label'] != 'unverified'])}/"
                     f"{len(members)} verified", fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
    for panel in range(len(layout), rows * columns):
        axes.flat[panel].set_visible(False)
    axes.flat[0].set_ylabel("counter delta / logical write bytes")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Write-path traffic evidence; dashed 0.8 and dotted 0.2\n"
                 "Client-RAM completion counters include fsync; timed client TX excludes it",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, path, manifest_path, owned)


def combined_groups(cases, operation):
    """Group combined cases by run configuration, medium, and measured state."""
    configs = list(dict.fromkeys(case["config"] for case in cases))
    states = (("backend", "server_ram", "client_ram") if operation == "read"
              else ("server_disk", "server_ram", "client_ram"))
    keys = [(config, medium, state) for config in configs for medium in ("HDD", "SSD")
            for state in states
            if any(case["config"] == config and case["medium"] == medium
                   and case["state"] == state for case in cases)]
    return [(config, medium, state, [case for case in cases
             if (case["config"], case["medium"], case["state"]) == (config, medium, state)])
            for config, medium, state in keys]


def combined_state_name(operation, state):
    """Return concise display labels for the benchmark state."""
    if operation == "read":
        return state.replace("_", " ")
    return {"server_disk": "disk fsync", "server_ram": "cache ack",
            "client_ram": "client write calls"}[state]


def plot_combined_throughput(cases, operation, path, manifest, owned):
    """Plot all supplied read or write throughput groups on one log-scale axis."""
    layout = combined_groups(cases, operation)
    colors = COLORS if operation == "read" else WRITE_COLORS
    fig, ax = plt.subplots(figsize=(max(13, 1.55 * len(layout)), 6))
    for position, (_, _, state, members) in enumerate(layout, 1):
        rates = sorted(case["rate"] for case in members)
        color = colors[state]
        ax.plot([position, position], [rates[0], rates[-1]], color=color,
                linewidth=2, solid_capstyle="butt", zorder=1)
        ax.plot([position - .16, position + .16], [median(rates)] * 2,
                color="black", linewidth=2.5, zorder=3)
        offsets = [(index - (len(rates) - 1) / 2) * .07 for index in range(len(rates))]
        for offset, rate in zip(offsets, rates):
            ax.plot(position + offset, rate, "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=.7, zorder=2)
    ax.set_xticks(range(1, len(layout) + 1))
    ax.set_xticklabels([f"{config}\n{medium} {combined_state_name(operation, state)}\n(n={len(members)})"
                        for config, medium, state, members in layout], fontsize=8)
    ax.set_xlim(.5, len(layout) + .5)
    ax.set_yscale("log")
    ax.set_ylabel("Throughput (MiB/s, log scale)")
    add_network_peak(ax)
    operation_name = "read" if operation == "read" else "write"
    note = ("IOR 8-GiB read" if operation == "read" else
            "8-GiB IOR includes fsync; client case times 1-GiB write calls, not fsync")
    ax.set_title(f"BeeGFS cache {operation_name} throughput: combined runs\n{note}\n"
                 "dots = cases; vertical line = min–max; black bar = median", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, path, manifest, owned, f"combined-cache-{operation}")


def plot_combined_evidence(cases, operation, path, manifest, owned):
    """Plot traffic evidence for every supplied run in one linear-scale image."""
    layout = combined_groups(cases, operation)
    columns = 2
    rows = -(-len(layout) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.1 * rows),
                             squeeze=False, sharey=True)
    series_by_state = {
        "read": (("network_ratio", "client received"),
                 ("server_network_ratio", "server sent"),
                 ("backend_ratio", "device read")),
        "write": (("client_sent_ratio", "client sent by completion"),
                  ("server_received_ratio", "server received"),
                  ("device_written_ratio", "device written")),
    }
    maximum = 1.0
    for _, _, state, members in layout:
        series = list(series_by_state[operation])
        if operation == "write" and state == "client_ram":
            series.append(("timed_client_sent_ratio", "client TX during timed writes"))
        maximum = max(maximum, *(value for case in members for key, _ in series
                                 if (value := case.get(key)) is not None))
    y_top = max(1.25, maximum * 1.1)

    for (config, medium, state, members), ax in zip(layout, axes.flat):
        series = list(series_by_state[operation])
        if operation == "write" and state == "client_ram":
            series.append(("timed_client_sent_ratio", "client TX during timed writes"))
        for slot, (key, label) in enumerate(series):
            positions = [index + (slot - (len(series) - 1) / 2) * .22
                         for index in range(len(members))]
            ax.bar(positions, [case[key] for case in members], width=.2, label=label)
        ax.axhline(.8, color="black", linestyle="--", linewidth=1)
        ax.axhline(.2, color="black", linestyle=":", linewidth=1)
        ax.set_ylim(0, y_top)
        ax.set_xticks(range(len(members)))
        ax.set_xticklabels([case["name"].split("-", 1)[0] for case in members])
        verified = sum(case["label"] != "unverified" for case in members)
        ax.set_title(f"{config} · {medium} {combined_state_name(operation, state)} "
                     f"({verified}/{len(members)} verified)", fontsize=9)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
    for panel in range(len(layout), rows * columns):
        axes.flat[panel].set_visible(False)
    axes.flat[0].set_ylabel("Counter delta / logical bytes (linear scale)")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    if operation == "read":
        title = "Cache read traffic evidence: combined runs; dashed 0.8 and dotted 0.2 thresholds"
    else:
        title = ("Cache write traffic evidence: combined runs; dashed 0.8 and dotted 0.2 thresholds\n"
                 "Client-RAM completion counters include fsync; timed client TX excludes it")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    return save(fig, path, manifest, owned, f"combined-cache-{operation}")


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


def summarize_write(owner, cases):
    """Return write-run throughput and evidence totals for stdout."""
    labels = [case["label"] for case in cases]
    lines = [f"run {owner['run_id']}: operation=write mode={owner['mode']} "
             f"remote_fsync={owner['remote_fsync']} pilot={owner['pilot']} "
             f"cases={len(cases)} verified={sum(label != 'unverified' for label in labels)} "
             f"unverified={labels.count('unverified')}"]
    for medium, state, members in groups(cases):
        rates = sorted(case["rate"] for case in members)
        lines.append(f"  {medium:3} {write_state_name(state):28} n={len(rates)} "
                     f"min={rates[0]:.1f} median={median(rates):.1f} "
                     f"max={rates[-1]:.1f} MiB/s")
    return lines


def main(argv=None):
    """Plot one run, or combine multiple runs of the same operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+",
                        help="one run, or several runs of the same operation to combine")
    args = parser.parse_args(argv)
    try:
        loaded = []
        for run in args.runs:
            raw_results = load(run / "results.json")
            is_write = (isinstance(raw_results, list) and bool(raw_results)
                        and isinstance(raw_results[0], dict)
                        and raw_results[0].get("operation") == "write")
            owner, cases = load_write_cases(run) if is_write else load_cases(run)
            operation = "write" if is_write else "read"
            if len(args.runs) > 1:
                config = (f"{owner['mode']} fsync={owner['remote_fsync']}" if is_write
                          else owner["mode"])
                for case in cases:
                    case["config"] = config
                    case["run_id"] = owner["run_id"]
            loaded.append((run, owner, cases, operation))

        operations = {item[3] for item in loaded}
        if len(operations) != 1:
            raise ValueError("combined plots require runs of the same operation")
        operation = operations.pop()
        if len(loaded) == 1:
            run, owner, cases, _ = loaded[0]
            plots = run / "plots"
            plots.mkdir(exist_ok=True)
            manifest = plots / "plot_manifest.json"
            current = load(manifest) if manifest.is_file() else {}
            if current and current.get("run_id") != owner["run_id"]:
                raise ValueError("plots/plot_manifest.json belongs to another run")
            if operation == "write":
                current = plot_write_throughput(cases, owner, plots / "write_throughput.png",
                                                manifest, current)
                plot_write_evidence(cases, owner, plots / "write_traffic_evidence.png",
                                    manifest, current)
                lines = summarize_write(owner, cases)
            else:
                current = plot_throughput(cases, plots / "throughput.png", manifest, current)
                current = plot_evidence(cases, plots / "traffic_evidence.png", manifest, current)
                if any(case["cached"] is not None for case in cases):
                    plot_residency(cases, plots / "client_residency.png", manifest, current)
                lines = summarize(owner, cases)
        else:
            roots = {run.absolute().parent for run, _, _, _ in loaded}
            if len(roots) != 1:
                raise ValueError("combined run directories must share one runs/ directory")
            report = next(iter(roots)).parent / "plots" / "cache" / operation
            if report.is_symlink():
                raise ValueError(f"refusing to write through a symlink: {report}")
            report.mkdir(parents=True, exist_ok=True)
            manifest = report / "plot_manifest.json"
            current = load(manifest) if manifest.is_file() else {}
            manifest_id = f"combined-cache-{operation}"
            if current and current.get("run_id") != manifest_id:
                raise ValueError("combined plot manifest belongs to another report")
            cases = [case for _, _, run_cases, _ in loaded for case in run_cases]
            current = plot_combined_throughput(
                cases, operation, report / "throughput.png", manifest, current)
            plot_combined_evidence(
                cases, operation, report / "traffic_evidence.png", manifest, current)
            print(f"combined {operation}: {len(loaded)} runs, {len(cases)} cases")
            print("runs: " + ", ".join(owner["run_id"] for _, owner, _, _ in loaded))
            lines = []
        if lines:
            print("\n".join(lines))
        print(f"\nplots written to {plots if len(loaded) == 1 else report}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Cache results not plottable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
