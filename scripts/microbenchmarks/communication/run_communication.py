#!/usr/bin/env python3
"""Offline NetBench protocol plan and exact IOR command builders.

No NetBench control, pool mutation or IOR execution occurs in this module.
See DESIGN.md and ../IMPLEMENTATION_RULES.md for execution requirements.
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


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 1:
        raise ValueError("unknown communication protocol")
    return config


def shuffled_blocks(configurations, repetitions, seed, rotation=0):
    configurations = list(configurations)
    random.Random(seed).shuffle(configurations)
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) * rotation % len(configurations)
        for position, case in enumerate(configurations[offset:] + configurations[:offset]):
            yield repetition, position, case


def finalized_plan(units):
    units = list(units)
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("duplicate unit")
    return units


def require_reviewed_inventory(inventory, domain, fields):
    if inventory.get("domain") != domain or inventory.get("reviewed") is not True:
        raise ValueError("reviewed communication inventory required")
    if any(not inventory.get(field) for field in fields):
        raise ValueError("communication inventory lacks required live fields")


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
            raise ValueError("run marker does not match communication protocol")
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
        raise ValueError("IOR operation or geometry differs from communication plan")
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
    fixed = {"placements": ["anjuna2", "anjuna3", "dual"], "ranks_per_client": [1, 4],
             "directions": ["write", "read"], "organizations": ["fpp", "shared"],
             "stripe_counts": [1, 4], "chunk_bytes": 524288,
             "transfer_bytes": 1048576, "block_bytes": 16 * 1024**3,
             "stonewall_seconds": 30, "chooser": "roundrobin", "repetitions": 5}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("communication settings differ from the fixed protocol")
    limit = config.get("backend_data_ratio_max")
    if limit is not None and (type(limit) not in (int, float) or not 0 <= limit <= 1):
        raise ValueError("invalid pilot-derived backend limit")
    if limit is not None and (not isinstance(config.get("calibrated_from"), str)
                              or len(config["calibrated_from"]) != 64
                              or any(character not in "0123456789abcdef"
                                     for character in config["calibrated_from"])):
        raise ValueError("approved backend limit must reference a completed pilot fingerprint")


def plan_units(config):
    validate_config(config)
    configurations = product(config["placements"], config["ranks_per_client"],
                             config["directions"], config["organizations"], config["stripe_counts"])
    configurations = [{"placement": placement, "ranks_per_client": ranks,
                       "direction": direction, "organization": organization, "stripes": stripes}
                      for placement, ranks, direction, organization, stripes in configurations]
    return finalized_plan({"id": f"r{rep:02d}-{case['placement']}-n{case['ranks_per_client']}-"
                                 f"{case['direction']}-{case['organization']}-s{case['stripes']}",
                           "repetition": rep, "position": position, **case}
                          for rep, position, case in shuffled_blocks(
                              configurations, config["repetitions"], config["order_seed"], rotation=7))


def pilot_units(config):
    """Eight-case coverage of both directions, organizations, stripes and max ranks."""
    validate_config(config)
    cases = (
        ("anjuna2", 1, "write", "fpp", 1),
        ("anjuna3", 1, "read", "shared", 4),
        ("dual", 1, "write", "shared", 1),
        ("anjuna2", 4, "read", "fpp", 4),
        ("anjuna3", 4, "write", "shared", 4),
        ("dual", 1, "read", "fpp", 1),
        ("dual", 4, "write", "fpp", 4),
        ("dual", 4, "read", "shared", 1),
    )
    return finalized_plan({"id": f"pilot-{index:02d}", "repetition": 1,
                           "placement": placement, "ranks_per_client": ranks,
                           "direction": direction, "organization": organization,
                           "stripes": stripes}
                          for index, (placement, ranks, direction, organization, stripes)
                          in enumerate(cases, 1))


def build_command(unit, mpirun, ior, owned_path, summary):
    clients = CLIENTS if unit["placement"] == "dual" else (unit["placement"],)
    ranks = unit["ranks_per_client"]
    if not clients or any(host not in CLIENTS for host in clients) or ranks not in (1, 4):
        raise ValueError("invalid MPI placement")
    hosts = ",".join(f"{host}:{ranks}" for host in clients)
    command = [str(mpirun), "-np", str(len(clients) * ranks), "--host", hosts,
               "--map-by", f"ppr:{ranks}:node", "--bind-to", "core", str(ior),
               "-a", "POSIX", "-t", "1m", "-b", "16g", "-s", "1", "-i", "1",
               "-g", "-D", "30", "-k", "--posix.odirect"]
    command += ["-w"] if unit["direction"] == "write" else ["-r", "-E"]
    if unit["organization"] == "fpp":
        command.append("-F")
    elif unit["organization"] != "shared":
        raise ValueError("unknown file organization")
    command += ["-o", str(owned_path), "-O", "summaryFormat=JSON",
                "-O", f"summaryFile={summary}"]
    return command


def prepare_read_dataset(unit, *, netbench_off, complete_files, layout_verified, synchronized):
    if unit["direction"] != "read" or not all((netbench_off, complete_files,
                                               layout_verified, synchronized)):
        raise ValueError("read preparation requires a complete normal BeeGFS dataset")
    return {"direction": "read", "status": "prepared", "unit": unit["id"]}


def netbench_transaction(participants, *, before, during, after, watchdog):
    """Validate recorded transition evidence; no client state is changed here."""
    if (not set(participants).issubset(CLIENTS) or not participants
            or any(before.get(client) != 0 or after.get(client) != 0 for client in CLIENTS)
            or any(during.get(client) != (1 if client in participants else 0) for client in CLIENTS)
            or not watchdog):
        raise ValueError("NetBench was not verified enabled only during owned traffic")
    return True


def validate_measurement(unit, native, evidence, config, *, owned_path, ior_version):
    limit = config.get("backend_data_ratio_max")
    calibration = limit is None and str(unit.get("id", "")).startswith("pilot-")
    if limit is None and not calibration:
        raise ValueError("NetBench backend tolerance requires a pilot and a new fingerprint")
    clients = CLIENTS if unit["placement"] == "dual" else (unit["placement"],)
    tasks = len(clients) * unit["ranks_per_client"]
    rank_times = evidence.get("rank_completion_seconds")
    if (not isinstance(rank_times, list) or len(rank_times) != tasks
            or any(type(value) not in (float, int) or not math.isfinite(value)
                   or not config["stonewall_seconds"] - 2 <= value <= config["stonewall_seconds"] + 5
                   for value in rank_times)):
        raise ValueError("not every rank reached the 30-second stonewall")
    netbench_transaction(clients, before=evidence["before_mode"],
                         during=evidence["during_mode"], after=evidence["after_mode"],
                         watchdog=evidence.get("watchdog_verified"))
    if (not evidence.get("layout_verified") or not evidence.get("telemetry_complete")
            or not evidence.get("pool_restored") or not evidence.get("chooser_restored")):
        raise ValueError("incomplete NetBench or cluster restoration evidence")
    result = validate_ior(native, operation=unit["direction"], tasks=tasks,
                          block_bytes=config["block_bytes"],
                          transfer_bytes=config["transfer_bytes"],
                          stonewall_seconds=config["stonewall_seconds"],
                          file_per_process=unit["organization"] == "fpp",
                          fsync_required=False, random_offsets=False,
                          use_existing=unit["direction"] == "read",
                          expected_path=owned_path, required_version=ior_version)
    transfer = result["transfer_seconds"]
    if (not config["stonewall_seconds"] - 2 <= transfer <= config["stonewall_seconds"] + 5
            or abs(transfer - max(rank_times)) > 2):
        raise ValueError("native transfer interval contradicts per-rank stonewall evidence")
    rank_bytes = evidence.get("rank_bytes")
    if (not isinstance(rank_bytes, list) or len(rank_bytes) != tasks
            or any(type(value) is not int or not 0 < value < config["block_bytes"]
                   for value in rank_bytes)
            or abs(sum(rank_bytes) - result["bytes_approx"]) > 1048576):
        raise ValueError("ranks exhausted the byte limit or native bytes contradict per-rank evidence")
    byte_field = "backend_read_bytes" if unit["direction"] == "read" else "backend_write_bytes"
    backend_bytes = evidence.get(byte_field)
    if type(backend_bytes) not in (int, float) or not math.isfinite(backend_bytes) or backend_bytes < 0:
        raise ValueError("missing direction-specific backend traffic evidence")
    ratio = backend_bytes / result["bytes_approx"]
    if not calibration and ratio > limit:
        raise ValueError("substantial backend traffic invalidates NetBench evidence")
    result["backend_ratio"] = ratio
    result["evidence_status"] = "calibration_only" if calibration else "verified_netbench"
    return result


def preflight(inventory):
    require_reviewed_inventory(inventory, "communication", ("clients", "targets", "mount",
                                                           "netbench_paths", "pool_baseline",
                                                           "chooser_baseline", "namespace"))
    if not inventory.get("watchdog_verified") or not inventory.get("restoration_tested"):
        raise ValueError("NetBench, pool and chooser restoration must be tested")
    targets = inventory["targets"]
    if (not isinstance(targets, list) or len(targets) != 4
            or {target.get("oss") for target in targets}
            != {"colva1", "colva2", "colva3", "colva4"}
            or len({target.get("id") for target in targets}) != 4
            or any(target.get("state") != "Online/Good" for target in targets)):
        raise ValueError("communication pool requires one healthy reviewed target per OSS")
    if set(inventory["netbench_paths"]) != {"anjuna2", "anjuna3"}:
        raise ValueError("both client NetBench proc paths must be reviewed")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(HERE / "communication_config.json")
    units = pilot_units(config) if args.pilot else plan_units(config)
    root = HERE.parents[2].resolve() / "results" / "microbenchmarks" / "runs"
    if not args.plan_out.resolve().is_relative_to(root):
        parser.error("plan output must be below project results/microbenchmarks/runs")
    record = write_plan(args.plan_out, "communication-pilot" if args.pilot else "communication",
                        config, units)
    print(json.dumps({"units": len(units), "fingerprint": record["fingerprint"]}))


if __name__ == "__main__":
    main()
