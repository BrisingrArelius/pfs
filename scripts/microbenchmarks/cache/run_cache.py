#!/usr/bin/env python3
"""Plan the HDD/SSD cache-read experiment and check its recorded evidence.

This is not yet a cluster runner: its CLI writes a plan, never reads a file or
changes a cache. See README.md before trying to run the experiment.

Each case: select one medium -> prepare one of three cache states -> read the
existing 8-GiB file -> compare observed network/disk/residency with the intended
state. The state changes and actual reads still need a live executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
STATES = ("client_miss_server_miss", "client_miss_server_hit", "client_hit")
MEDIA = ("HDD", "SSD")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 1:
        raise ValueError("unknown cache protocol")
    return config


def positive_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")
    return value


def ordered_cases(cases, repetitions, seed):
    """Shuffle once, then rotate the six cases across five repetitions."""
    cases = list(cases)
    random.Random(seed).shuffle(cases)
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) % len(cases)
        for position, case in enumerate(cases[offset:] + cases[:offset]):
            yield repetition, position, case


def finalized_plan(units):
    units = list(units)
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("duplicate unit")
    return units


def require_reviewed_inventory(inventory, domain, fields):
    if inventory.get("domain") != domain or inventory.get("reviewed") is not True:
        raise ValueError("reviewed cache inventory required")
    if any(not inventory.get(field) for field in fields):
        raise ValueError("cache inventory lacks required live fields")


def open_directory_nofollow(path):
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(path).absolute().parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def write_plan(path, domain, config, units):
    path = Path(path).absolute()
    if (path.parent.parent != RUNS or path.name != "plan.json"
            or path.is_symlink() or ".." in path.parts or not path.parent.is_dir()):
        raise ValueError("plan must be inside an existing project run directory")
    if any(component.is_symlink() for component in (path.parent, RUNS)):
        raise ValueError("symlink in result directory")
    record = {"domain": domain, "config": config, "units": units}
    record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    directory = open_directory_nofollow(path.parent)
    temporary = f".plan-{uuid.uuid4().hex}"
    try:
        marker = os.open("owner.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(marker, "r", encoding="utf-8") as source:
            owner = json.load(source)
        if owner != {"run_id": path.parent.name, "domain": domain}:
            raise ValueError("run marker does not match cache protocol")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)
    return record


def validate_ior(native, *, operation, tasks, block_bytes, transfer_bytes,
                 file_per_process, expected_path, required_version, use_existing):
    """Check that saved IOR output really describes our one complete read."""
    if not isinstance(native, dict) or native.get("Version") != required_version:
        raise ValueError("IOR version or JSON schema changed")
    tests, summary = native.get("tests"), native.get("summary")
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(summary, list) or len(summary) != 1:
        raise ValueError("expected one IOR test and phase")
    test, row = tests[0], summary[0]
    params, options = test.get("Parameters"), test.get("Options")
    if not isinstance(params, dict) or not isinstance(options, dict) or not isinstance(row, dict):
        raise ValueError("IOR native parameters missing")
    phases = options.get("Results")
    if not isinstance(phases, list) or len(phases) != 1:
        raise ValueError("expected exactly one IOR read phase")

    # IOR repeats some fields in its Parameters, Options and summary sections.
    # Check every copy: a positive throughput number alone is not enough.
    expected_fields = (
        (phases[0], "access", operation),
        (row, "operation", operation), (row, "API", "POSIX"),
        (params, "api", "POSIX"), (options, "tasks", tasks),
        (row, "numTasks", tasks),
        (row, "blockSize", block_bytes), (params, "blockSize", block_bytes),
        (row, "transferSize", transfer_bytes), (params, "transferSize", transfer_bytes),
        (row, "segmentCount", 1),
        (row, "repetitions", 1), (params, "repetitions", 1),
        (params, "testFileName", str(expected_path)),
        (row, "filePerProc", int(file_per_process)),
        (params, "filePerProc", int(file_per_process)),
        (params, "useExistingTestFile", int(use_existing)),
        (params, "readFile", int(operation == "read")),
        (params, "writeFile", int(operation == "write")),
    )
    if any(source.get(name) != expected for source, name, expected in expected_fields):
        raise ValueError("IOR operation or geometry differs from cache plan")
    phase = phases[0]
    values = (phase.get("bwMiB"), phase.get("iops"), phase.get("totalTime"),
              phase.get("wrRdTime"), row.get("xsizeMiB"), row.get("MeanTime"), row.get("bwMeanMIB"))
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("invalid native IOR performance")
    if (abs(phase["totalTime"] - row["MeanTime"]) > 0.01
            or abs(phase["bwMiB"] - row["bwMeanMIB"]) > max(0.1, phase["bwMiB"] * .01)
            or abs(phase["bwMiB"] * phase["totalTime"] - row["xsizeMiB"]) > max(2, row["xsizeMiB"] * .02)
            or abs(phase["iops"] - row["xsizeMiB"] * 1048576 /
                   transfer_bytes / phase["wrRdTime"]) > max(1, phase["iops"] * .02)):
        raise ValueError("IOR native bytes, duration and rate disagree")
    return {"mib_per_second": phase["bwMiB"], "bytes_approx": row["xsizeMiB"] * 1048576,
            "iops": phase["iops"], "seconds": phase["totalTime"]}


def validate_config(config):
    if (tuple(config["states"]) != STATES or config["client"] != "anjuna2"
            or tuple(config["media"]) != MEDIA
            or config["file_bytes"] != 8 * 1024**3
            or config["safety_reserve_bytes"] != 8 * 1024**3
            or config["transfer_bytes"] != 1024**2
            or config["stripe_count"] != 1 or config["chunk_bytes"] != 512 * 1024):
        raise ValueError("cache protocol differs from the reviewed design")
    if config["repetitions"] != 5:
        raise ValueError("full cache protocol requires five repetitions")
    for state in STATES:
        thresholds = config["thresholds"][state]
        for name, value in thresholds.items():
            positive_number(value, name)
            if value > 1:
                raise ValueError("invalid cache threshold")


def plan_units(config):
    """Five rotated blocks of the six media/cache-state combinations."""
    validate_config(config)
    cases = [(media, state) for media in MEDIA for state in STATES]
    plan = []
    for repetition, position, (media, state) in ordered_cases(
            cases, config["repetitions"], config["order_seed"]):
        plan.append({"id": f"r{repetition:02d}-{media.lower()}-{state}",
                     "repetition": repetition, "media": media,
                     "state": state, "position": position})
    return finalized_plan(plan)


def pilot_units(config):
    """One of each media/state, then one more client hit on each medium."""
    validate_config(config)
    cases = [(media, state) for media in MEDIA for state in STATES]
    cases += [(media, "client_hit") for media in MEDIA]
    plan = []
    for index, (media, state) in enumerate(cases, 1):
        plan.append({"id": f"pilot-{index:02d}-{media.lower()}-{state}",
                     "media": media, "state": state,
                     "repetition": 1 if index <= 6 else 2})
    return finalized_plan(plan)


def build_command(ior, mpirun, owned_file, native_summary):
    """Single rank, normal buffered POSIX read; no O_DIRECT and no write option."""
    return [str(mpirun), "-np", "1", str(ior), "-a", "POSIX", "-r", "-E", "-k",
            "-g", "-t", "1m", "-b", "8g", "-s", "1", "-i", "1",
            "-o", str(owned_file), "-O", "summaryFormat=JSON",
            "-O", f"summaryFile={native_summary}"]


def prepare_cache_state(state, *, evidence):
    """List required steps; no cache-control command runs here."""
    if state not in STATES:
        raise ValueError("unknown cache state")
    if not evidence.get("exclusive_allocation") or not evidence.get("writeback_settled"):
        raise ValueError("cache control requires exclusive allocation and settled writes")
    steps = ["drop_client_and_selected_oss", "wait_settle", "verify_client_residency"]
    if state != STATES[0]:
        steps = ["drop_client_and_selected_oss", "warm_complete_file", "verify_warm_traffic"]
        if state == STATES[1]:
            steps += ["drop_client_only", "verify_client_residency"]
        else:
            steps += ["verify_client_residency_at_least_95_percent"]
    return steps + ["capture_zero_point", "measure_immediately"]


def classify_achieved_state(state, evidence, thresholds):
    """Classify saved traffic and residency; a repeated read alone proves nothing."""
    if state not in STATES:
        raise ValueError("unknown cache state")
    logical = positive_number(evidence["logical_bytes"], "logical bytes")
    if (not evidence.get("quiescent") or not evidence.get("netbench_off")
            or not evidence.get("telemetry_complete") or not evidence.get("file_unchanged")):
        return "cache_influenced_unverified"
    network = evidence["network_bytes"] / logical
    backend = evidence["backend_read_bytes"] / logical
    residency = evidence["client_residency"]
    if any(type(value) not in (int, float) or not 0 <= value < float("inf")
           for value in (network, backend, residency)) or residency > 1:
        raise ValueError("invalid cache telemetry")
    limits = thresholds[state]
    ratios = {"network": network, "backend": backend, "residency": residency}
    for criterion, limit in limits.items():
        name, bound = criterion.rsplit("_", 1)
        outside_limit = (ratios[name] < limit if bound == "min"
                         else ratios[name] > limit)
        if outside_limit:
            return "cache_influenced_unverified"
    return state


def validate_measurement(unit, native, evidence, config, *, owned_file, ior_version):
    metrics = validate_ior(native, operation="read", tasks=1,
                           block_bytes=config["file_bytes"],
                           transfer_bytes=config["transfer_bytes"],
                           file_per_process=False, expected_path=owned_file,
                           use_existing=True,
                           required_version=ior_version)
    if abs(metrics["bytes_approx"] - config["file_bytes"]) > 1048576:
        raise ValueError("cache read did not transfer the entire 8 GiB file")
    if abs(evidence.get("logical_bytes", 0) - metrics["bytes_approx"]) > 1048576:
        raise ValueError("cache telemetry logical bytes differ from IOR native bytes")
    state = classify_achieved_state(unit["state"], evidence, config["thresholds"])
    return {**metrics, "intended_state": unit["state"], "achieved_state": state}


def preflight(inventory):
    """Validate a *saved* inventory; this does not inspect live cluster state."""
    require_reviewed_inventory(inventory, "cache", ("client_mode", "targets", "devices",
                                                    "mount", "restoration", "namespace"))
    if inventory["client_mode"] != "native" or inventory.get("available_ram_bytes", 0) < 16 * 1024**3:
        raise ValueError("three-state cache study requires native mode and 16 GiB available")
    if inventory.get("server_available_ram_bytes", 0) < 16 * 1024**3:
        raise ValueError("one-OSS cache study requires 16 GiB available on the storage server")
    targets = inventory["targets"]
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("cache study needs exactly two targets")
    if ({target.get("media") for target in targets} != set(MEDIA)
            or len({target.get("oss") for target in targets}) != 1
            or targets[0].get("id") == targets[1].get("id")):
        raise ValueError("cache study needs one reviewed HDD and one SSD target on the same OSS")
    for target in targets:
        if (target.get("oss") not in {"colva1", "colva2", "colva3", "colva4"}
                or target.get("state") != "Online/Good" or not target.get("id")
                or not target.get("device")):
            raise ValueError("cache target identity, device or health is missing")
    if not inventory.get("watchdog_verified") or not inventory.get("privilege_verified"):
        raise ValueError("cache restoration watchdog and authorization must be verified")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(HERE / "cache_config.json")
    units = pilot_units(config) if args.pilot else plan_units(config)
    root = HERE.parents[2].resolve()
    if not args.plan_out.resolve().is_relative_to(root / "results" / "microbenchmarks" / "runs"):
        parser.error("plan output must be below project results/microbenchmarks/runs")
    record = write_plan(args.plan_out, "cache-pilot" if args.pilot else "cache", config, units)
    print(json.dumps({"units": len(units), "fingerprint": record["fingerprint"]}))


if __name__ == "__main__":
    main()
