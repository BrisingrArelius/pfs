#!/usr/bin/env python3
"""Offline mdtest plan, command and namespace safety contracts.

No MPI launch or BeeGFS mutation is performed here. Read DESIGN.md and
../IMPLEMENTATION_RULES.md before implementing cluster execution.
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
import re
import stat
import sys
import uuid

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
CLIENTS = ("anjuna2", "anjuna3")
PHASES = ("directory_create", "directory_stat", "directory_remove",
          "file_create", "file_stat", "file_read", "file_remove")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 1:
        raise ValueError("unknown metadata protocol")
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
        raise ValueError("reviewed metadata inventory required")
    if any(not inventory.get(field) for field in fields):
        raise ValueError("metadata inventory lacks required live fields")


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
            raise ValueError("run marker does not match metadata protocol")
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


def open_directory_nofollow(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("directory traversal")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def admit_pilot(observed_step_seconds, cleanup_reserve_seconds):
    required = {"preparation", "measurement", "validation", "restoration", "cleanup"}
    if not isinstance(observed_step_seconds, dict) or set(observed_step_seconds) != required:
        raise ValueError("pilot requires measured timing for every step")
    values = (*observed_step_seconds.values(), cleanup_reserve_seconds)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in values) or sum(values) >= 1200:
        raise ValueError("pilot plus cleanup must fit under 20 minutes")
    return True


def validate_mdtest(stdout, *, schema, tasks, items_per_rank, workdir):
    """Accept seven phase rows only under the pilot-pinned native output schema."""
    if (not isinstance(stdout, str) or len(stdout) > 4 * 1024 * 1024
            or not isinstance(schema, dict) or set(schema.get("phases", {})) != set(PHASES)
            or not all(schema.get(key) for key in ("version_signature", "task_signature",
                                                    "item_signature", "path_signature"))):
        raise ValueError("missing pinned mdtest native output schema")
    for key, value in (("version_signature", None), ("task_signature", tasks),
                       ("item_signature", items_per_rank), ("path_signature", workdir)):
        signature = schema[key] if value is None else schema[key].format(value=re.escape(str(value)))
        if not re.search(signature, stdout, re.MULTILINE):
            raise ValueError(f"mdtest {key} missing")
    results = {}
    for phase in PHASES:
        pattern = schema["phases"][phase]
        if not isinstance(pattern, str) or not pattern.startswith("^") or not pattern.endswith("$"):
            raise ValueError("mdtest phase expression must cover a whole line")
        matches = list(re.finditer(pattern, stdout, re.MULTILINE))
        if len(matches) != 1 or set(matches[0].groupdict()) != {"rate", "elapsed", "operations"}:
            raise ValueError(f"mdtest {phase} missing, duplicated or unknown")
        groups = matches[0].groupdict()
        rate, elapsed, operations = float(groups["rate"]), float(groups["elapsed"]), int(groups["operations"])
        if (not math.isfinite(rate) or not math.isfinite(elapsed) or rate <= 0 or elapsed <= 0
                or operations != tasks * items_per_rank
                or abs(rate - operations / elapsed) > max(1, rate * .02)):
            raise ValueError(f"invalid mdtest {phase} native metrics")
        results[phase] = {"rate": rate, "seconds": elapsed, "operations": operations}
    if re.search(r"(?im)^\s*(?:error|fatal|failed|stonewall)\b", stdout):
        raise ValueError("mdtest reported failure")
    return results


def validate_config(config):
    fixed = {"placements": ["anjuna2", "anjuna3", "dual"],
             "ranks_per_client": [1, 4, 16], "layouts": ["flat", "per_rank"],
             "items_per_rank": 100000, "bytes_written_per_file": 0,
             "bytes_read_per_file": 0, "rotation": 7, "repetitions": 5}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("mdtest settings differ from the reviewed protocol")


def plan_units(config):
    validate_config(config)
    configurations = [{"placement": placement, "ranks_per_client": ranks, "layout": layout}
                      for placement, ranks, layout in product(config["placements"],
                                                               config["ranks_per_client"],
                                                               config["layouts"])]
    return finalized_plan({"id": f"r{rep:02d}-{case['placement']}-"
                                 f"n{case['ranks_per_client']}-{case['layout']}",
                           "repetition": rep, "position": position, **case}
                          for rep, position, case in shuffled_blocks(
                              configurations, config["repetitions"], config["order_seed"], rotation=7))


def build_command(unit, mpirun, mdtest, owned_workdir):
    clients = CLIENTS if unit["placement"] == "dual" else (unit["placement"],)
    ranks = unit["ranks_per_client"]
    if any(client not in CLIENTS for client in clients) or ranks not in (1, 4, 16):
        raise ValueError("invalid MPI placement")
    if unit["layout"] not in ("flat", "per_rank"):
        raise ValueError("invalid directory layout")
    command = [str(mpirun), "-np", str(len(clients) * ranks), "--host",
               ",".join(f"{client}:{ranks}" for client in clients),
               "--map-by", f"ppr:{ranks}:node", "--bind-to", "core", str(mdtest),
               "-d", str(owned_workdir), "-n", "100000", "-i", "1", "-w", "0",
               "-e", "0", "-N", "0", "-P"]
    if unit["layout"] == "per_rank":
        command.append("-u")
    return command


def pilot_units(config):
    """Bounded four-case production-geometry pilot; time admission needs live estimates."""
    plan_units(config)
    cases = (("anjuna2", 1, "flat"), ("anjuna3", 1, "flat"),
             ("dual", 1, "per_rank"), ("dual", 16, "flat"))
    return finalized_plan({"id": f"pilot-{index:02d}", "placement": placement,
                           "ranks_per_client": ranks, "layout": layout, "repetition": 1}
                          for index, (placement, ranks, layout) in enumerate(cases, 1))


def owned_attempt_path(base, run_id, unit_id, attempt_number):
    """Construct a non-traversing relative namespace path from canonical IDs."""
    for value in (run_id, unit_id):
        if not value or not all(character.isalnum() or character in "_-" for character in value):
            raise ValueError("unsafe run or unit identifier")
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("invalid attempt number")
    base = Path(base)
    if base.is_symlink() or any(part.is_symlink() for part in base.parents):
        raise ValueError("symlink in namespace root")
    path = base / run_id / "attempts" / unit_id / f"attempt-{attempt_number}"
    if path.exists() or path.is_symlink():
        raise FileExistsError("attempt path already exists")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("attempt path escapes reviewed base")
    return path


def cleanup_attempt(path, *, base, run_id, unit_id, attempt_number,
                    fingerprint, lock_held, mount_verified, owned_entries):
    """Remove exactly one owned BeeGFS attempt, never a sibling or run root."""
    if not lock_held or not mount_verified:
        raise ValueError("cleanup needs held lock and verified BeeGFS mount")
    path, base = Path(path), Path(base)
    expected = base / run_id / "attempts" / unit_id / f"attempt-{attempt_number}"
    if path != expected or not path.is_dir():
        raise ValueError("not the exact generated attempt path")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink in attempt path")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("attempt path escapes reviewed namespace")
    if not isinstance(owned_entries, dict) or not owned_entries:
        raise ValueError("cleanup requires a saved inventory of owned entries")
    descriptor = open_directory_nofollow(path)
    try:
        marker_fd = os.open("owner.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        with os.fdopen(marker_fd, "r", encoding="utf-8") as source:
            owner = json.load(source)
        expected_owner = {"run_id": run_id, "unit_id": unit_id,
                          "attempt_id": f"attempt-{attempt_number}", "fingerprint": fingerprint}
        if any(owner.get(key) != value for key, value in expected_owner.items()):
            raise ValueError("attempt marker does not match the manifest")
        seen = {}

        def walk(current, prefix, *, remove):
            for name in os.listdir(current):
                relative = f"{prefix}/{name}" if prefix else name
                identity = os.stat(name, dir_fd=current, follow_symlinks=False)
                if (not (stat.S_ISDIR(identity.st_mode) or stat.S_ISREG(identity.st_mode))
                        or identity.st_dev != os.fstat(descriptor).st_dev
                        or owned_entries.get(relative) != (identity.st_dev, identity.st_ino)):
                    raise ValueError("attempt contains a foreign or changed entry")
                seen[relative] = (identity.st_dev, identity.st_ino)
                if stat.S_ISDIR(identity.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=current)
                    try:
                        if os.fstat(child).st_ino != identity.st_ino:
                            raise ValueError("attempt directory changed during cleanup")
                        walk(child, relative, remove=remove)
                    finally:
                        os.close(child)
                    if remove:
                        again = os.stat(name, dir_fd=current, follow_symlinks=False)
                        if (again.st_dev, again.st_ino) != (identity.st_dev, identity.st_ino):
                            raise ValueError("attempt directory changed during cleanup")
                        os.rmdir(name, dir_fd=current)
                elif remove:
                    again = os.stat(name, dir_fd=current, follow_symlinks=False)
                    if (again.st_dev, again.st_ino) != (identity.st_dev, identity.st_ino):
                        raise ValueError("attempt file changed during cleanup")
                    os.unlink(name, dir_fd=current)

        walk(descriptor, "", remove=False)
        if seen != owned_entries:
            raise ValueError("attempt inventory has missing or unknown entries")
        seen.clear()
        walk(descriptor, "", remove=True)
        parent = open_directory_nofollow(path.parent)
        try:
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (os.fstat(descriptor).st_dev,
                                                    os.fstat(descriptor).st_ino):
                raise ValueError("attempt root changed during cleanup")
            os.rmdir(path.name, dir_fd=parent)
        finally:
            os.close(parent)
    finally:
        os.close(descriptor)


def preflight(inventory):
    require_reviewed_inventory(inventory, "metadata", ("clients", "meta_target", "mount",
                                                      "tools", "namespace", "allowed_cores"))
    if any(len(inventory["allowed_cores"].get(host, [])) < 16 for host in CLIENTS):
        raise ValueError("each client needs at least 16 allowed distinct CPU cores")
    if not inventory.get("netbench_off") or not inventory.get("shared_mount_verified"):
        raise ValueError("NetBench-off and identical BeeGFS mount must be verified")
    return True


def validate_measurement(unit, stdout, evidence, *, schema, workdir):
    if (not evidence.get("rank_map_verified") or not evidence.get("telemetry_complete")
            or not evidence.get("netbench_off") or not evidence.get("mount_unchanged")):
        raise ValueError("metadata run lacks verified rank/telemetry/mount evidence")
    tasks = unit["ranks_per_client"] * (2 if unit["placement"] == "dual" else 1)
    return validate_mdtest(stdout, schema=schema, tasks=tasks,
                           items_per_rank=100000, workdir=workdir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(HERE / "metadata_config.json")
    units = pilot_units(config) if args.pilot else plan_units(config)
    root = HERE.parents[2].resolve() / "results" / "microbenchmarks" / "runs"
    if not args.plan_out.resolve().is_relative_to(root):
        parser.error("plan output must be below project results/microbenchmarks/runs")
    record = write_plan(args.plan_out, "metadata-pilot" if args.pilot else "metadata", config, units)
    print(json.dumps({"units": len(units), "fingerprint": record["fingerprint"]}))


if __name__ == "__main__":
    main()
