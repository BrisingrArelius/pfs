#!/usr/bin/env python3
"""Per-target FIO: prepare once, checkpoint each measurement, clean up last."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

from run_support import (
    BudgetExpired, atomic_json, atomic_text, duration, make_directory, now, read_json,
    remaining_seconds, run_command, set_deadline, sync_directory,
)

HERE = Path(__file__).resolve().parent
OPERATIONS = {"read", "write", "randread", "randwrite"}
BLOCK_BYTES = {"4k": 4096, "128k": 131072, "1m": 1048576}


def parse_args(argv=None):
    """Validate the small CLI before creating output or probing targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--targets", help="Comma-separated local target IDs; default: all")
    parser.add_argument("--pilot", action="store_true",
                        help="Run one repetition per workload instead of the full five")
    parser.add_argument("--resume", action="store_true")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--time-limit", type=duration)
    timing.add_argument("--deadline", help="ISO-8601 time with timezone")
    timing.add_argument("--extend-deadline", type=duration)
    parser.add_argument("--cleanup-buffer", type=duration, default=300)
    args = parser.parse_args(argv)
    try:
        if args.targets is not None:
            args.targets = [int(value) for value in args.targets.split(",")]
            if len(set(args.targets)) != len(args.targets):
                raise ValueError("duplicate target IDs")
        if args.extend_deadline is not None and (args.resume or args.pilot or args.targets is not None):
            raise ValueError("extension mode does not accept --resume, --pilot or --targets")
        if args.deadline:
            deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                raise ValueError("deadline needs a timezone")
            args.deadline = deadline.timestamp()
            if args.deadline <= time.time() + args.cleanup_buffer:
                raise ValueError("deadline is already inside the cleanup window")
        if args.time_limit is not None and args.time_limit <= args.cleanup_buffer:
            raise ValueError("time limit must exceed cleanup buffer")
    except ValueError as error:
        parser.error(str(error))
    args.results_dir = args.results_dir.absolute()
    return args


def validate_config(config):
    """Reject unsupported geometry and unsafe job overrides before any I/O."""
    fio, prep, planning = config["fio"], config["prepare"], config["planning"]
    fixed = {"numjobs": 4, "nrfiles": 1, "direct": 1, "time_based": 1,
             "ramp_time": 5, "allow_file_create": 0, "end_fsync": 0}
    allowed = set(fixed) | {"size", "runtime", "ioengine", "iodepth",
                            "clat_percentiles", "percentile_list"}
    if set(fio) - allowed or any(fio.get(key) != value for key, value in fixed.items()):
        raise ValueError("unsupported FIO options: require four sustained direct-I/O jobs")
    if fio["ioengine"] != "libaio" or config["repetitions"] not in (1, 5):
        raise ValueError("require libaio and either one pilot or five full repetitions")
    if set(prep) != {"rw", "bs", "end_fsync", "timeout_seconds"} or (
            prep["rw"], prep["bs"], prep["end_fsync"]) != ("write", "1m", 1):
        raise ValueError("preparation must be a full sequential 1-MiB write with final sync")
    positive = [fio["size"], fio["runtime"], fio["iodepth"], config["protocol_version"],
                config["free_space_reserve_bytes"], prep["timeout_seconds"],
                config["measurement_timeout_seconds"], planning["margin_multiplier"],
                planning["overhead_seconds"]]
    positive += [value for value in planning["prepare_seconds"].values() if value is not None]
    positive += [value for media in planning["measurement_seconds"].values()
                 for value in media.values()]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in positive):
        raise ValueError("sizes, durations and planning values must be positive finite numbers")
    if type(fio["size"]) is not int or fio["size"] % BLOCK_BYTES["1m"]:
        raise ValueError("file size must be an integer multiple of 1 MiB")
    if planning["margin_multiplier"] < 1 or config["measurement_timeout_seconds"] <= fio["runtime"]:
        raise ValueError("planning margin must be >= 1; hard timeout must exceed FIO runtime")
    names = []
    for workload in config["workloads"]:
        if (set(workload) != {"name", "rw", "bs"}
                or not re.fullmatch(r"[a-z][a-z0-9_]*", workload["name"])
                or workload["rw"] not in OPERATIONS or workload["bs"] not in BLOCK_BYTES):
            raise ValueError(f"invalid workload: {workload}")
        names.append(workload["name"])
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("require five uniquely named workloads")


def dataset_size(config):
    """Return the owned file size for four disjoint per-job regions."""
    return config["fio"]["size"] * config["fio"]["numjobs"]


def plan_cases(config, targets):
    """Make a seeded target order and position-balanced five-round schedule."""
    ordered = sorted(targets, key=lambda target: target["target_id"])
    if len({target["target_id"] for target in ordered}) != len(ordered):
        raise ValueError("duplicate inventory target IDs")
    random.Random(config["order_seed"]).shuffle(ordered)
    workloads = list(config["workloads"])
    random.Random(config["order_seed"]).shuffle(workloads)
    cases = []
    for target in ordered:
        for index in range(config["repetitions"]):
            for workload in workloads[index:] + workloads[:index]:
                cases.append({"id": f"{target['target_id']}__{workload['name']}__r{index + 1}",
                              "target_id": target["target_id"], "workload": workload,
                              "repetition": index + 1, "attempts": []})
    return cases


def build_job(config, target, workload, file_path, phase):
    """Render setup or four measured jobs over disjoint regions of one file."""
    options = dict(config["fio"], overwrite=1, fallocate="none", unique_filename=0)
    filename = str(file_path)
    if any(char in filename for char in "\n\r$\\"):
        raise ValueError("unsupported characters in FIO file path")
    filename = filename.replace(":", r"\:")
    if phase == "prepare":
        options.pop("runtime")
        options.pop("ramp_time")
        options.update(numjobs=1, time_based=0, size=dataset_size(config))
        options.update({key: config["prepare"][key] for key in ("rw", "bs", "end_fsync")})
        options["allow_file_create"] = 1
        options["filename"] = filename
        return f"[target-{target['target_id']}]\n" + "".join(
            f"{key}={value}\n" for key, value in options.items())
    sections = []
    for index in range(config["fio"]["numjobs"]):
        job = dict(options, numjobs=1, rw=workload["rw"], bs=workload["bs"],
                   offset=index * config["fio"]["size"],
                   randseed=workload["repetition"] * config["fio"]["numjobs"] + index,
                   filename=filename)
        sections.append(f"[target-{target['target_id']}-job-{index + 1}]\n" + "".join(
            f"{key}={value}\n" for key, value in job.items()))
    return "\n".join(sections)


def validate_result(path, expected):
    """Validate all native jobs and aggregate one target-level measurement."""
    result = read_json(path)
    jobs = result.get("jobs", [])
    names = expected.get("jobnames", [expected["jobname"]])
    if (len(jobs) != len(names) or [job.get("jobname") for job in jobs] != names
            or any(job.get("error") != 0 for job in jobs)):
        raise ValueError(f"missing, unexpected or failed FIO job in {path}")
    opposite = "write" if expected["operation"] == "read" else "read"
    transferred = total_ios = 0
    elapsed = 0
    for job in jobs:
        stats = job[expected["operation"]]
        job_bytes, job_elapsed = stats["io_bytes"], stats["runtime"]
        if (type(job_bytes) is not int or job_bytes <= 0
                or type(job_elapsed) not in (int, float) or not math.isfinite(job_elapsed)
                or job_elapsed <= 0 or stats.get("total_ios", 0) <= 0):
            raise ValueError(f"invalid I/O accounting in {path}")
        if job.get(opposite, {}).get("io_bytes", 0):
            raise ValueError(f"unexpected {opposite} traffic in {path}")
        if expected.get("time_based") and job_elapsed < expected["runtime"] * 1000 - 1000:
            raise ValueError(f"FIO stopped before the runtime limit in {path}")
        transferred += job_bytes
        total_ios += stats["total_ios"]
        elapsed = max(elapsed, job_elapsed)
    if elapsed > expected["hard_timeout"] * 1000:
        raise ValueError(f"FIO duration exceeds the command timeout in {path}")
    reason = "time_limit" if expected.get("time_based") else (
        "byte_limit" if transferred == expected["size"] else "time_limit")
    if not expected.get("time_based") and transferred > expected["size"]:
        raise ValueError(f"invalid I/O byte limit in {path}")
    if reason == "time_limit" and not expected.get("time_based") and (
            not expected["runtime"] or elapsed < expected["runtime"] * 1000 - 1000):
        raise ValueError(f"FIO stopped before either completion limit in {path}")
    return result, {"completion_reason": reason, "fio_runtime_ms": elapsed,
                    "io_bytes": transferred, "total_ios": total_ios}


def completed(attempts):
    """Only the latest validated attempt determines whether work is finished."""
    return bool(attempts and attempts[-1]["state"] == "completed")


def target_data_path(target, run_id):
    """Return the only data pathname this run may create, write or unlink."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("invalid run ID in manifest")
    mount = Path(target["mount"])
    if not mount.is_absolute() or target["target_id"] <= 0:
        raise ValueError("target mount and ID cannot form a safe data path")
    return mount / ".local-fio" / run_id / f"target-{target['target_id']}" / "data"


def require_data_path(target, run_id, path):
    """Reject tampered/traversing paths before inspection, execution or deletion."""
    path, expected = Path(path), target_data_path(target, run_id)
    if path != expected or "beegfs_storage" in path.parts or path == Path(target["device"]):
        raise ValueError(f"unsafe benchmark data path: {path}")
    return path


def artifact_path(root, relative):
    """Confine raw evidence to the selected results directory, including on resume."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"artifact escapes results directory: {relative}")
    return path


def run_markdown(manifest):
    """Render a readable, non-authoritative description of the exact run."""
    config = manifest["config"]
    fio = config["fio"]
    total_gib = dataset_size(config) / 2**30
    region_gib = fio["size"] / 2**30
    lines = [
        f"# Local FIO run on {manifest['host']}", "",
        "> Generated from `manifest.json`; the JSON manifest and native FIO files are authoritative.", "",
        "## Identity", "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Protocol version: `{config['protocol_version']}`",
        f"- FIO version: `{manifest['fio_version']}`", "",
        "## Measurement protocol", "",
        f"- Parallel jobs per OST: **{fio['numjobs']}**",
        f"- Per-job region: **{region_gib:g} GiB**",
        f"- Prepared file per OST: **{total_gib:g} GiB**",
        f"- I/O engine: `{fio['ioengine']}` with `direct={fio['direct']}`",
        f"- Queue depth: **{fio['iodepth']} per job**, up to **{fio['iodepth'] * fio['numjobs']} aggregate**",
        f"- Timing: **{fio['ramp_time']} s ramp + {fio['runtime']} measured s**, `time_based={fio['time_based']}`",
        f"- Repetitions: **{config['repetitions']}** per workload",
        "- Each job uses a separate non-overlapping region of the same prepared file.",
        "- Reported bandwidth and IOPS are sums across jobs for one OST.", "",
        "## Workloads", "", "| Name | Pattern | Block size |", "|---|---|---:|",
    ]
    lines.extend(f"| `{item['name']}` | `{item['rw']}` | `{item['bs']}` |"
                 for item in config["workloads"])
    lines.extend(["", "## Targets", "", "| OST ID | Media | Mount | Device |",
                  "|---:|---|---|---|"])
    lines.extend(f"| {target['target_id']} | {target['media']} | `{target['mount']}` | `{target['device']}` |"
                 for target in manifest["inventory"])
    lines.extend(["", "## Evidence", "",
                  "Raw `job.fio`, `fio.json`, stdout and stderr are under `raw/`. ",
                  "Session progress, completion state, file identity and device snapshots are in `manifest.json`.", ""])
    return "\n".join(lines)


def save(run):
    """Checkpoint the sole progress record, including the observed live deadline."""
    if run["deadline"].exists():
        try:
            observed = read_json(run["deadline"])
            json.dumps(observed, allow_nan=False)
            run["manifest"]["sessions"][-1]["deadline"] = observed
        except (OSError, ValueError):
            pass  # Still persist a failure caused by malformed deadline state.
    atomic_json(run["root"] / "manifest.json", run["manifest"])
    atomic_text(run["root"] / "RUN.md", run_markdown(run["manifest"]))


def load_run(args):
    """Select local targets, validate compatibility and restore durable progress."""
    config = read_json(HERE / "fio_config.json")
    validate_config(config)
    host = socket.gethostname().split(".")[0]
    inventory = read_json(HERE / "target_inventory.json")
    if host not in inventory:
        raise ValueError(f"host {host!r} is not in target_inventory.json")
    manifest_path = args.results_dir / "manifest.json"
    if manifest_path.exists() != args.resume:
        raise ValueError("use --resume for an existing manifest; a new run needs a new directory")
    old = read_json(manifest_path) if args.resume else None
    mode = old.get("mode", "full") if old else ("pilot" if args.pilot else "full")
    if old and args.pilot and mode != "pilot":
        raise ValueError("--pilot cannot change an existing full run")
    config["repetitions"] = 1 if mode == "pilot" else 5
    ids = args.targets if args.targets is not None else (
        [target["target_id"] for target in old["inventory"]] if old else
        [target["target_id"] for target in inventory[host]])
    targets = [target for target in inventory[host] if target["target_id"] in ids]
    if len(targets) != len(ids) or not targets:
        raise ValueError(f"targets must belong to {host}: {ids}")
    version = subprocess.run(["fio", "--version"], capture_output=True, text=True,
                             check=True, timeout=10).stdout.strip()
    if not re.fullmatch(r"fio-\d+.*", version):
        raise ValueError(f"expected Flexible I/O Tester, found {version!r}")
    scientific = {key: value for key, value in config.items()
                  if key not in {"planning", "measurement_timeout_seconds", "prepare"}}
    scientific["prepare"] = {key: value for key, value in config["prepare"].items()
                              if key != "timeout_seconds"}
    scientific.update(host=host, inventory=targets, fio_version=version,
                      job_policy={"overwrite": 1, "fallocate": "none", "unique_filename": 0,
                                  "region_layout": "disjoint_offsets"})
    fingerprint = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    if old and old["fingerprint"] != fingerprint:
        raise ValueError("resume rejected: scientific settings, targets or FIO version changed")
    manifest = old or {"run_id": uuid.uuid4().hex, "mode": mode, "fingerprint": fingerprint,
                       "config": config, "inventory": targets, "fio_version": version,
                       "host": host, "targets": {}, "cases": plan_cases(config, targets),
                       "sessions": []}
    for target in targets:
        tid = str(target["target_id"])
        path = target_data_path(target, manifest["run_id"])
        state = manifest["targets"].setdefault(tid, {
            "path": str(path), "preparations": [], "file_identity": None,
            "file_owned": False, "cleanup": "pending"})
        if state["path"] != str(path):
            raise ValueError("recorded data path does not match this run and target")
    for owner in list(manifest["targets"].values()) + manifest["cases"]:
        for attempt in owner.get("preparations", owner.get("attempts", [])):
            if attempt["state"] == "running":
                attempt.update(state="interrupted", error="previous session ended without checkpoint")
            if attempt["state"] == "completed":
                folder = artifact_path(args.results_dir, attempt["artifacts"])
                try:
                    if not all((folder / name).is_file() for name in ("job.fio", "fio.json", "stdout", "stderr")):
                        raise ValueError("required raw artifact missing")
                    validate_result(folder / "fio.json", attempt["expected"])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    attempt.update(state="failed", error=f"invalid saved evidence: {error}")
    for previous in manifest["sessions"]:
        if previous["outcome"] == "running":
            previous["outcome"] = "interrupted"
    manifest["sessions"].append({"id": len(manifest["sessions"]) + 1, "started_at": now(),
        "outcome": "running", "kernel": platform.release(),
        "memory": Path("/proc/meminfo").read_text(), "planning": config["planning"],
        "allocation_id": os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID"),
        "device_cache_settings": None, "controller_details": None,
        "execution_assumption": "operator runs one benchmark instance per host"})
    return {"root": args.results_dir, "deadline": args.results_dir / "deadline.json",
            "cleanup_seconds": args.cleanup_buffer, "config": config, "manifest": manifest}


def check_target(target):
    """Verify the exact inventoried XFS mount and snapshot capacity/device activity."""
    path = Path(target["mount"])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"invalid mount path: {path}")
    result = subprocess.run(["findmnt", "--json", "--mountpoint", str(path),
        "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"], check=True, capture_output=True,
        text=True, timeout=10)
    mounts = json.loads(result.stdout).get("filesystems", [])
    if (len(mounts) != 1 or mounts[0]["target"] != str(path)
            or mounts[0]["fstype"] != "xfs"
            or os.path.realpath(mounts[0]["source"]) != os.path.realpath(target["device"])):
        raise ValueError(f"mount identity mismatch for target {target['target_id']}: {mounts}")
    usage = os.statvfs(path)
    total, free = usage.f_blocks * usage.f_frsize, usage.f_bavail * usage.f_frsize
    device = Path(os.path.realpath(target["device"])).name
    counters = next((line for line in Path("/proc/diskstats").read_text().splitlines()
                     if len(line.split()) > 2 and line.split()[2] == device), None)
    return {"captured_at": now(), "mount": mounts[0], "total_bytes": total,
            "free_bytes": free, "free_percent": 100 * free / total if total else 0,
            "free_inodes": usage.f_favail, "diskstats": counters}


def file_identity(path):
    """Reject symlinks/non-regular files and identify the owned file, if present."""
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"symlink in benchmark path: {path}")
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"benchmark file is not a private regular file: {path}")
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size}


def estimate_seconds(run, target, workload, phase):
    """Select observed/pilot wall time and compute the next phase's admission budget."""
    state, config = run["manifest"], run["config"]
    planning, media = config["planning"], target["media"]
    if phase == "prepare":
        attempts = state["targets"][str(target["target_id"])]["preparations"]
        estimate = planning["prepare_seconds"].get(media)
    else:
        attempts = [attempt for case in state["cases"]
                    if case["target_id"] == target["target_id"] and case["workload"] == workload
                    for attempt in case["attempts"]]
        estimate = planning["measurement_seconds"].get(media, {}).get(workload["name"], config["fio"]["runtime"])
    observed = [attempt["command_wall_seconds"] for attempt in attempts if attempt["state"] == "completed"]
    if observed:
        estimate = max(observed)
    if estimate is None:
        raise ValueError(f"set planning.prepare_seconds.{media} from a pilot/provisional estimate")
    return {"estimated_wall_seconds": estimate, "estimate_source": "observed" if observed else "config/fallback",
            "admission_seconds": estimate * planning["margin_multiplier"] + planning["overhead_seconds"]}


def execute_fio(run, target, workload, repetition, phase, attempts, estimate):
    """Checkpoint one FIO invocation; share evidence handling for setup and measurement."""
    config, manifest = run["config"], run["manifest"]
    state = manifest["targets"][str(target["target_id"])]
    path = require_data_path(target, manifest["run_id"], state["path"])
    number = len(attempts) + 1
    relative = (f"raw/target-{target['target_id']}/prepare-{number}" if phase == "prepare" else
                f"raw/{target['target_id']}__{workload['name']}__r{repetition}/attempt-{number}")
    folder = artifact_path(run["root"], relative)
    if folder.exists():
        raise ValueError(f"refusing to overwrite existing artifacts: {folder}")
    timeout = config["prepare"]["timeout_seconds"] if phase == "prepare" else config["measurement_timeout_seconds"]
    jobname = f"target-{target['target_id']}"
    expected = {"jobname": jobname, "size": dataset_size(config),
                 "operation": "write" if phase == "prepare" or "write" in workload["rw"] else "read",
                 "runtime": 0 if phase == "prepare" else config["fio"]["runtime"],
                 "time_based": phase != "prepare" and bool(config["fio"]["time_based"]),
                 "hard_timeout": timeout}
    if phase != "prepare":
        expected["jobnames"] = [f"{jobname}-job-{index + 1}"
                                for index in range(config["fio"]["numjobs"])]
    attempt = dict(estimate, id=number, state="running", phase=phase, started_at=now(),
                   session=manifest["sessions"][-1]["id"], artifacts=relative, expected=expected,
                   hard_timeout_seconds=timeout, preparation_generation=(number if phase == "prepare" else
                   len(state["preparations"])), remaining_seconds=remaining_seconds(run["deadline"], run["cleanup_seconds"]))
    attempts.append(attempt)
    save(run)  # An abandoned attempt remains identifiable even without raw output.
    started = time.monotonic()
    try:
        make_directory(folder)
        attempt["before"] = check_target(target)
        required = config["free_space_reserve_bytes"] + (dataset_size(config) if phase == "prepare" else 0)
        if attempt["before"]["free_bytes"] < required or attempt["before"]["free_inodes"] < 1:
            raise ValueError("insufficient free space/inodes on target")
        if phase == "prepare":
            if file_identity(path) is not None:
                raise ValueError("preparation needs an absent, owned data path")
            make_directory(path.parent)
            state["file_owned"] = True
            save(run)
        before_identity = file_identity(path)
        if phase != "prepare" and (before_identity != state["file_identity"] or before_identity is None):
            raise ValueError("prepared file identity changed")
        job = build_job(config, target, dict(workload or {}, repetition=repetition), path, phase)
        with (folder / "job.fio").open("x") as output:
            output.write(job)
            output.flush()
            os.fsync(output.fileno())
        sync_directory(folder)
        attempt["command"] = ["fio", "--output-format=json", f"--output={folder / 'fio.json'}",
                              f"--aux-path={folder}", "--eta=never", str(folder / "job.fio")]
        attempt["file_before"] = before_identity
        save(run)
        attempt["command_wall_seconds"] = run_command(
            attempt["command"], folder / "stdout", folder / "stderr", run["deadline"],
            run["cleanup_seconds"], timeout, cwd=folder)
        with (folder / "fio.json").open("rb") as output:
            os.fsync(output.fileno())
        sync_directory(folder)
        _, summary = validate_result(folder / "fio.json", expected)
        after_identity = file_identity(path)
        if after_identity is None or after_identity["size"] != dataset_size(config):
            raise ValueError("FIO did not leave a complete target file")
        if phase != "prepare" and after_identity != before_identity:
            raise ValueError("FIO recreated, truncated or lost the target file")
        attempt.update(summary, file_after=after_identity, after=check_target(target), returncode=0)
        if attempt["after"]["free_bytes"] < config["free_space_reserve_bytes"]:
            raise ValueError("target reserve breached during FIO")
        if phase == "prepare":
            state["file_identity"] = after_identity
        attempt["state"] = "completed"
    except BaseException as error:
        attempt.update(state="interrupted" if isinstance(error, (BudgetExpired, KeyboardInterrupt)) else "failed",
                       error=f"{type(error).__name__}: {error}", returncode=getattr(error, "returncode", None))
        if hasattr(error, "command_wall_seconds"):
            attempt["command_wall_seconds"] = error.command_wall_seconds
        raise
    finally:
        attempt.update(ended_at=now(), phase_wall_seconds=time.monotonic() - started)
        save(run)


def run_case(run, case):
    """Admit and checkpoint one measurement without touching the file lifecycle."""
    target = next(target for target in run["manifest"]["inventory"] if target["target_id"] == case["target_id"])
    estimate = estimate_seconds(run, target, case["workload"], "measure")
    if remaining_seconds(run["deadline"], run["cleanup_seconds"]) < estimate["admission_seconds"]:
        raise BudgetExpired(f"not enough estimated time for {case['id']}")
    print(f"{case['id']}: measurement", flush=True)
    execute_fio(run, target, case["workload"], case["repetition"], "measure", case["attempts"], estimate)


def run_target(run, target, cases):
    """Own one prepared file across measurements and allocations; delete it last."""
    state = run["manifest"]["targets"][str(target["target_id"])]
    path = require_data_path(target, run["manifest"]["run_id"], state["path"])
    state["last_check"] = check_target(target)
    actual = file_identity(path)
    known = state["file_identity"]
    if actual is not None and not state.get("file_owned", False):
        raise ValueError(f"unexpected file at {path}; refusing to overwrite/delete it")
    if actual is not None and known is not None and any(
            actual[key] != known[key] for key in ("device", "inode")):
        raise ValueError(f"unexpected file at {path}; refusing to overwrite/delete it")
    pending = [case for case in cases if not completed(case["attempts"])]
    if pending:
        state["cleanup"] = "pending"
        ready = completed(state["preparations"])
        if ready and actual is not None and actual != known:
            raise ValueError(f"prepared file size changed: {path}")
        if not ready or actual is None:
            estimate = estimate_seconds(run, target, None, "prepare")
            first = estimate_seconds(run, target, pending[0]["workload"], "measure")
            if remaining_seconds(run["deadline"], run["cleanup_seconds"]) < estimate["admission_seconds"] + first["admission_seconds"]:
                raise BudgetExpired("not enough estimated time for preparation and first measurement")
            if actual is not None:  # Only a known incomplete/invalid preparation reaches here.
                path.unlink()
                sync_directory(path.parent)
            state["file_identity"] = None
            state["file_owned"] = True
            save(run)
            print(f"target {target['target_id']}: prepare once", flush=True)
            execute_fio(run, target, None, 0, "prepare", state["preparations"], estimate)
        for case in pending:
            run_case(run, case)
    # Cleanup is independent of measurement completion and safe to retry alone.
    started = time.monotonic()
    try:
        check_target(target)
        actual = file_identity(path)
        if actual is not None:
            if actual != state["file_identity"]:
                raise ValueError("file identity changed before cleanup")
            path.unlink()
            sync_directory(path.parent)
        state["cleanup"] = "completed"
        state["file_owned"] = False
        state.pop("cleanup_error", None)
    except BaseException as error:
        state.update(cleanup="failed", cleanup_error=str(error))
        raise
    finally:
        state["cleanup_wall_seconds"] = time.monotonic() - started
        save(run)


def main(argv=None):
    """Coordinate targets and sessions; stop immediately on unexpected failure."""
    args = parse_args(argv)
    run = None
    started = time.monotonic()
    try:
        if args.extend_deadline is not None:
            set_deadline(args.results_dir / "deadline.json", extend=args.extend_deadline)
            print("Reservation deadline extended.")
            return 0
        if not args.resume and args.results_dir.exists() and any(
                path.name != ".lock" for path in args.results_dir.iterdir()):
            raise ValueError("a new run needs an empty results directory")
        if args.resume and not args.results_dir.is_dir():
            raise ValueError("resume directory does not exist")
        make_directory(args.results_dir)
        with (args.results_dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            run = load_run(args)
            set_deadline(run["deadline"], time_limit=args.time_limit, deadline=args.deadline)
            save(run)
            try:
                cases = run["manifest"]["cases"]
                for tid in dict.fromkeys(case["target_id"] for case in cases):
                    target = next(item for item in run["manifest"]["inventory"] if item["target_id"] == tid)
                    run_target(run, target, [case for case in cases if case["target_id"] == tid])
                outcome, code = "completed", 0
            except BudgetExpired as error:
                outcome, code = "budget_stop", 0
                run["manifest"]["sessions"][-1]["message"] = str(error)
            except KeyboardInterrupt:
                outcome, code = "interrupted", 130
            except Exception as error:
                outcome, code = "failed", 1
                run["manifest"]["sessions"][-1]["error"] = f"{type(error).__name__}: {error}"
                print(f"Stopped: {error}", file=sys.stderr)
            run["manifest"]["sessions"][-1].update(
                outcome=outcome, ended_at=now(), wall_seconds=time.monotonic() - started)
            save(run)
            print(f"{outcome}: {args.results_dir / 'manifest.json'}")
            return code
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Cannot run: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
