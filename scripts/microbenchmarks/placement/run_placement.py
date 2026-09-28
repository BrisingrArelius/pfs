#!/usr/bin/env python3
"""Offline placement matrix, IOR command, and state-verification contracts.

Historical configure_pools.sh/reset_pools.sh are not used. Read DESIGN.md and
../IMPLEMENTATION_RULES.md before implementing live cluster transitions.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
CLIENTS = ("anjuna2", "anjuna3")
WORKLOADS = ("seq_read_fpp", "seq_write_fpp", "rand_read_fpp", "rand_write_fpp",
             "seq_read_shared", "seq_write_shared")
CONCURRENCY = {"a2_r1": (("anjuna2",), 1), "a2_r4": (("anjuna2",), 4),
               "dual_r1": (CLIENTS, 1), "dual_r4": (CLIENTS, 4)}


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 1:
        raise ValueError("unknown placement protocol")
    return config


def finalized_plan(units):
    units = list(units)
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("duplicate unit")
    return units


def require_reviewed_inventory(inventory, domain, fields):
    if inventory.get("domain") != domain or inventory.get("reviewed") is not True:
        raise ValueError("reviewed placement inventory required")
    if any(not inventory.get(field) for field in fields):
        raise ValueError("placement inventory lacks required live fields")


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
            raise ValueError("run marker does not match placement protocol")
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
                 stonewall_seconds, file_per_process, fsync_required, random_offsets,
                 use_existing, expected_path, required_version):
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
    if (not isinstance(phases, list) or len(phases) != 1 or phases[0].get("access") != operation
            or row.get("operation") != operation or row.get("API") != "POSIX"
            or params.get("api") != "POSIX" or options.get("tasks") != tasks
            or row.get("numTasks") != tasks or row.get("blockSize") != block_bytes
            or params.get("blockSize") != block_bytes or row.get("transferSize") != transfer_bytes
            or params.get("transferSize") != transfer_bytes or row.get("segmentCount") != 1
            or row.get("repetitions") != 1 or params.get("repetitions") != 1
            or params.get("testFileName") != str(expected_path)
            or row.get("filePerProc") != int(file_per_process)
            or params.get("filePerProc") != int(file_per_process)
            or params.get("useExistingTestFile") != int(use_existing)
            or params.get("fsync") != int(fsync_required)
            or params.get("randomOffset") != int(random_offsets)
            or params.get("deadlineForStonewall") != stonewall_seconds
            or params.get("readFile") != int(operation == "read")
            or params.get("writeFile") != int(operation == "write")):
        raise ValueError("IOR operation or geometry differs from placement plan")
    phase = phases[0]
    values = (phase.get("bwMiB"), phase.get("iops"), phase.get("totalTime"),
              phase.get("wrRdTime"), row.get("xsizeMiB"), row.get("MeanTime"), row.get("bwMeanMIB"))
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("invalid native IOR performance")
    if (abs(phase["totalTime"] - row["MeanTime"]) > .01
            or abs(phase["bwMiB"] - row["bwMeanMIB"]) > max(.1, phase["bwMiB"] * .01)
            or abs(phase["bwMiB"] * phase["totalTime"] - row["xsizeMiB"]) > max(2, row["xsizeMiB"] * .02)
            or abs(phase["iops"] - row["xsizeMiB"] * 1048576 /
                   transfer_bytes / phase["wrRdTime"]) > max(1, phase["iops"] * .02)):
        raise ValueError("IOR native bytes, duration and rate disagree")
    return {"mib_per_second": phase["bwMiB"], "bytes_approx": row["xsizeMiB"] * 1048576,
            "iops": phase["iops"], "seconds": phase["totalTime"],
            "transfer_seconds": phase["wrRdTime"]}


def validate_config(config):
    fixed = {"storage_configurations": ["D", "S", "H"],
             "choosers": ["randomized", "roundrobin"], "stripe_counts": [1, 2, 4],
             "stripe_sizes_bytes": [262144, 524288, 1048576], "workloads": list(WORKLOADS),
             "concurrency": list(CONCURRENCY), "block_bytes": 16 * 1024**3,
             "stonewall_seconds": 60, "repetitions": 5}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("placement settings differ from the baseline-capacity protocol")
    if config.get("capacity_states") == ["baseline"]:
        return
    provisioning = config.get("capacity_provisioning")
    if (config.get("capacity_states") != ["high_free", "low_free"]
            or config.get("capacity_approved") is not True
            or not isinstance(config.get("capacity_bands"), dict)
            or not isinstance(provisioning, dict)
            or not all(provisioning.get(key) for key in
                       ("filler_root", "restore_plan", "inode_reserve", "headroom_bytes"))):
        raise ValueError("capacity comparison needs approved high/low site bands and restoration")
    fields = ("free_bytes_min", "free_bytes_max", "free_percent_min", "free_percent_max",
              "free_inodes_min", "byte_class", "inode_class")
    for state in config["capacity_states"]:
        band = config["capacity_bands"].get(state)
        if not isinstance(band, dict) or len(band) != 28:
            raise ValueError("capacity bands must cover all 28 reviewed targets")
        for target, limits in band.items():
            if not str(target).isdigit() or not all(name in limits for name in fields):
                raise ValueError("capacity band has incomplete target limits")
            if (type(limits["free_bytes_min"]) is not int or
                    type(limits["free_bytes_max"]) is not int or
                    type(limits["free_inodes_min"]) is not int or
                    not 0 < limits["free_bytes_min"] < limits["free_bytes_max"] or
                    limits["free_inodes_min"] <= 0 or
                    not 0 <= limits["free_percent_min"] < limits["free_percent_max"] <= 100 or
                    not limits["byte_class"] or not limits["inode_class"]):
                raise ValueError("invalid per-target capacity band")
    if set(config["capacity_bands"]["high_free"]) != set(config["capacity_bands"]["low_free"]):
        raise ValueError("high/low states must use the same physical targets")
    for target, high in config["capacity_bands"]["high_free"].items():
        low = config["capacity_bands"]["low_free"][target]
        if (high["free_bytes_min"] <= low["free_bytes_max"]
                or high["free_percent_min"] <= low["free_percent_max"]):
            raise ValueError("high/low free-space bands must be distinct and ordered")


def validate_capacity_inventory(config, inventory):
    if config["capacity_states"] != ["baseline"]:
        reviewed_ids = {str(item["id"]) for item in inventory["targets"]}
        for state in config["capacity_states"]:
            if set(config["capacity_bands"][state]) != reviewed_ids:
                raise ValueError("capacity bands do not match the reviewed target inventory")


def plan_units(config, inventory=None):
    validate_config(config)
    if inventory is not None:
        validate_capacity_inventory(config, inventory)
    result = []
    configurations = [dict(zip(("stripes", "chunk_bytes", "workload", "concurrency"), values))
                      for values in product(config["stripe_counts"], config["stripe_sizes_bytes"],
                                            config["workloads"], config["concurrency"])]
    for capacity in config["capacity_states"]:
        for repetition in range(1, config["repetitions"] + 1):
            storages = config["storage_configurations"]
            storages = storages[(repetition - 1) % 3:] + storages[:(repetition - 1) % 3]
            choosers = config["choosers"] if repetition % 2 else list(reversed(config["choosers"]))
            for block_position, (chooser, storage) in enumerate(product(choosers, storages)):
                ordered = list(configurations)
                random.Random(config["order_seed"] + 10 * config["choosers"].index(chooser)
                              + config["storage_configurations"].index(storage)).shuffle(ordered)
                offset = (repetition - 1) * 11 % len(ordered)
                ordered = ordered[offset:] + ordered[:offset]
                for position, case in enumerate(ordered):
                    unit = {"id": f"{capacity}-{storage}-{chooser}-r{repetition:02d}-"
                                  f"s{case['stripes']}-c{case['chunk_bytes']}-"
                                  f"{case['workload']}-{case['concurrency']}",
                            "capacity": capacity, "storage": storage, "chooser": chooser,
                            "repetition": repetition, "block_position": block_position,
                            "position": position, **case}
                    if inventory is not None:
                        eligible = eligible_targets(inventory, storage)
                        if len(eligible) < case["stripes"]:
                            unit["infeasible_reason"] = "insufficient eligible Online/Good targets"
                    result.append(unit)
    return finalized_plan(result)


def pilot_units(config):
    """Pairwise-style coverage proposal, subject to measured 20-minute admission."""
    validate_config(config)
    cases = (
        ("D", "randomized", 1, 262144, "seq_read_fpp", "a2_r1"),
        ("S", "roundrobin", 4, 1048576, "seq_write_fpp", "dual_r4"),
        ("H", "randomized", 1, 1048576, "rand_read_fpp", "a2_r4"),
        ("D", "roundrobin", 4, 262144, "rand_write_fpp", "dual_r1"),
        ("S", "randomized", 4, 262144, "seq_read_shared", "dual_r4"),
        ("H", "roundrobin", 1, 1048576, "seq_write_shared", "a2_r1"),
    )
    return finalized_plan({"id": f"pilot-{index:02d}", "capacity": "baseline",
                           "storage": storage, "chooser": chooser, "stripes": stripes,
                           "chunk_bytes": chunk, "workload": workload, "concurrency": concurrency,
                           "repetition": 1}
                          for index, (storage, chooser, stripes, chunk, workload, concurrency)
                          in enumerate(cases, 1))


def eligible_targets(inventory, storage):
    require_reviewed_inventory(inventory, "placement", ("targets", "pool_baseline", "namespace"))
    targets = inventory["targets"]
    if not isinstance(targets, list) or len({item["id"] for item in targets}) != len(targets):
        raise ValueError("target identity is ambiguous")
    if (len(targets) != 28 or
            sum(item["media"] == "HDD" for item in targets) != 14 or
            sum(item["media"] == "NVMe" for item in targets) != 14 or
            {item["oss"] for item in targets} != {"colva1", "colva2", "colva3", "colva4"}):
        raise ValueError("live target geometry differs from the 28-target protocol")
    if storage not in ("D", "S", "H"):
        raise ValueError("unknown placement configuration")
    media = {"D": {"HDD", "NVMe"}, "S": {"NVMe"}, "H": {"HDD"}}[storage]
    return [item["id"] for item in targets
            if item["media"] in media and item["state"] == "Online/Good"]


def build_command(unit, mpirun, ior, owned_path, summary):
    workload = unit["workload"]
    if workload not in WORKLOADS or unit["concurrency"] not in CONCURRENCY:
        raise ValueError("unknown workload or concurrency")
    hosts, ranks_per_client = CONCURRENCY[unit["concurrency"]]
    command = [str(mpirun), "-np", str(len(hosts) * ranks_per_client), "--host",
               ",".join(f"{host}:{ranks_per_client}" for host in hosts),
               "--map-by", f"ppr:{ranks_per_client}:node", "--bind-to", "core", str(ior),
               "-a", "POSIX", "-t", "4k" if workload.startswith("rand") else "1m",
               "-b", "16g", "-s", "1", "-i", "1", "-g", "-D", "60", "-k",
               "--posix.odirect"]
    if "read" in workload:
        command += ["-r", "-E"]
    else:
        command += ["-w", "-e"]
        if workload == "rand_write_fpp":
            command.append("-E")
    if workload.endswith("fpp"):
        command += ["-F"]
    if workload.startswith("rand"):
        command += ["-z"]
    return command + ["-o", str(owned_path), "-O", "summaryFormat=JSON",
                      "-O", f"summaryFile={summary}"]


def apply_block_state(unit, *, observed, inventory):
    """Verify saved state evidence; this function never changes pool/chooser."""
    eligible = eligible_targets(inventory, unit["storage"])
    if (unit.get("infeasible_reason") or len(eligible) < unit["stripes"]
            or observed.get("chooser") != unit["chooser"]
            or set(observed.get("eligible_targets", ())) != set(eligible)
            or observed.get("netbench") != {client: 0 for client in CLIENTS}
            or not observed.get("restoration_watchdog_active")):
        raise ValueError("cluster state does not match the placement block")
    if unit["capacity"] != "baseline" and not observed.get("approved_capacity_verified"):
        raise ValueError("high/low capacity band is not verified on every target")
    return True


def validate_layout(unit, actual_targets, inventory):
    eligible = set(eligible_targets(inventory, unit["storage"]))
    if (len(actual_targets) != unit["stripes"] or len(set(actual_targets)) != len(actual_targets)
            or not set(actual_targets).issubset(eligible)):
        raise ValueError("actual file layout differs from the desired eligible layout")
    return True


def classify_completion(rank_results, *, tasks, block_bytes, stonewall_seconds,
                        native_bytes, native_transfer_seconds):
    if (not isinstance(rank_results, list) or len(rank_results) != tasks
            or {entry.get("rank") for entry in rank_results if isinstance(entry, dict)} != set(range(tasks))):
        raise ValueError("rank-level size-or-stonewall evidence is incomplete")
    for entry in rank_results:
        bytes_done, seconds = entry.get("bytes"), entry.get("transfer_seconds")
        if (type(bytes_done) is not int or not 0 < bytes_done <= block_bytes
                or type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds <= 0 or
                bytes_done < block_bytes
                and not stonewall_seconds - 2 <= seconds <= stonewall_seconds + 5):
            raise ValueError("a rank stopped before its byte limit and before stonewall")
    if abs(sum(entry["bytes"] for entry in rank_results) - native_bytes) > 1048576:
        raise ValueError("per-rank bytes contradict the native aggregate")
    if (any(entry["bytes"] < block_bytes for entry in rank_results)
            and native_transfer_seconds < stonewall_seconds - 2):
        raise ValueError("native transfer phase ended before the stonewall")
    return ("byte_limit" if all(entry["bytes"] == block_bytes for entry in rank_results)
            else "stonewall")


def validate_measurement(unit, native, evidence, config, inventory, *, owned_path,
                         ior_version):
    if (not evidence.get("netbench_off") or not evidence.get("telemetry_complete")
            or not evidence.get("synchronized_write", False) and "write" in unit["workload"]
            or not evidence.get("pool_restored") or not evidence.get("chooser_restored")):
        raise ValueError("placement state or telemetry incomplete")
    if unit["workload"] == "rand_write_fpp" and (
            not evidence.get("random_existing_file_verified")
            or evidence.get("identity_before") != evidence.get("identity_after")
            or evidence.get("size_before") != evidence.get("size_after")):
        raise ValueError("random write may have recreated or truncated its prepared files")
    validate_capacity_inventory(config, inventory)
    for window in ("before", "after"):
        values = evidence.get(f"target_capacity_{window}")
        if not isinstance(values, dict) or set(values) != {str(item["id"]) for item in inventory["targets"]}:
            raise ValueError("capacity telemetry does not cover all targets")
        for target, observed in values.items():
            if (type(observed.get("free_bytes")) is not int or observed["free_bytes"] < 0
                    or type(observed.get("free_inodes")) is not int or observed["free_inodes"] < 0
                    or type(observed.get("free_percent")) not in (int, float)
                    or not math.isfinite(observed["free_percent"])
                    or not 0 <= observed["free_percent"] <= 100
                    or not observed.get("byte_class") or not observed.get("inode_class")):
                raise ValueError("invalid target capacity evidence")
            if unit["capacity"] != "baseline":
                limits = config["capacity_bands"][unit["capacity"]][target]
                if (not limits["free_bytes_min"] <= observed["free_bytes"] <= limits["free_bytes_max"]
                        or not limits["free_percent_min"] <= observed["free_percent"] <= limits["free_percent_max"]
                        or observed["free_inodes"] < limits["free_inodes_min"]
                        or observed["byte_class"] != limits["byte_class"]
                        or observed["inode_class"] != limits["inode_class"]):
                    raise ValueError("target left the approved capacity band")
    layouts = evidence.get("actual_layouts", [])
    hosts, ranks = CONCURRENCY[unit["concurrency"]]
    if len(layouts) != (len(hosts) * ranks if unit["workload"].endswith("fpp") else 1):
        raise ValueError("not every rank file was checked")
    for targets in layouts:
        validate_layout(unit, targets, inventory)
    operation = "read" if "read" in unit["workload"] else "write"
    metrics = validate_ior(native, operation=operation, tasks=len(hosts) * ranks,
                           block_bytes=config["block_bytes"],
                           transfer_bytes=4096 if unit["workload"].startswith("rand") else 1048576,
                           stonewall_seconds=config["stonewall_seconds"],
                           file_per_process=unit["workload"].endswith("fpp"),
                           fsync_required=operation == "write",
                           random_offsets=unit["workload"].startswith("rand"),
                           use_existing=operation == "read" or unit["workload"] == "rand_write_fpp",
                           expected_path=owned_path, required_version=ior_version)
    metrics["completion_reason"] = classify_completion(
        evidence.get("rank_results"), tasks=len(hosts) * ranks,
        block_bytes=config["block_bytes"], stonewall_seconds=config["stonewall_seconds"],
        native_bytes=metrics["bytes_approx"], native_transfer_seconds=metrics["transfer_seconds"])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(HERE / "placement_config.json")
    units = pilot_units(config) if args.pilot else plan_units(config)
    root = HERE.parents[2].resolve() / "results" / "microbenchmarks" / "runs"
    if not args.plan_out.resolve().is_relative_to(root):
        parser.error("plan output must be below project results/microbenchmarks/runs")
    record = write_plan(args.plan_out, "placement-pilot" if args.pilot else "placement",
                        config, units)
    print(json.dumps({"units": len(units), "fingerprint": record["fingerprint"]}))


if __name__ == "__main__":
    main()
