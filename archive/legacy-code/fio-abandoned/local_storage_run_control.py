#!/usr/bin/env python3
"""Reservation, safety, and durable-state helpers for local-storage FIO runs."""

import argparse
from datetime import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from local_storage_fio import write_json_atomic


class BudgetExpired(RuntimeError):
    """Indicate that the reservation entered its cleanup window."""


def load_json(path):
    """Load one JSON file and return its decoded value."""
    with open(path) as source:
        return json.load(source)


def parse_duration(value):
    """Convert values such as 90m, 5h, or 1d to seconds."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("use a duration such as 90m, 5h, or 1d")
    amount = float(match.group(1))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    return amount * multiplier


def parse_deadline(value):
    """Convert a timezone-qualified ISO-8601 deadline to an epoch timestamp."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use an ISO-8601 deadline with a timezone") from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("deadline must include a timezone")
    return parsed.timestamp()


def parse_fio_size(value):
    """Convert the simple binary size suffixes accepted by this FIO configuration."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?", str(value).lower())
    if not match:
        raise ValueError(f"unsupported FIO size: {value}")
    multiplier = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    return int(float(match.group(1)) * multiplier[match.group(2)])


def read_active_deadline(deadline_file):
    """Read the mutable deadline used by the active runner."""
    return load_json(deadline_file).get("deadline_epoch")


def initialize_deadline(deadline_file, deadline):
    """Create or reset deadline state while preserving its extension history."""
    previous = load_json(deadline_file) if deadline_file.exists() else {}
    state = {
        "deadline_epoch": deadline,
        "deadline": datetime.fromtimestamp(deadline).astimezone().isoformat() if deadline else None,
        "updated_at": datetime.now().astimezone().isoformat(),
        "extensions": previous.get("extensions", []),
    }
    write_json_atomic(deadline_file, state)
    return state


def extend_deadline(deadline_file, duration):
    """Atomically add seconds to a live deadline and preserve an audit entry."""
    state = load_json(deadline_file)
    previous = state.get("deadline_epoch")
    if previous is None:
        raise ValueError("the active run has no deadline to extend")
    extended = max(previous, time.time()) + duration
    state["deadline_epoch"] = extended
    state["deadline"] = datetime.fromtimestamp(extended).astimezone().isoformat()
    state.setdefault("extensions", []).append({
        "extended_at": datetime.now().astimezone().isoformat(),
        "added_seconds": duration,
        "previous_deadline_epoch": previous,
    })
    write_json_atomic(deadline_file, state)
    return state


def terminate_process_group(process):
    """Terminate a benchmark process group, escalating to SIGKILL after ten seconds."""
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()


def run_with_deadline(command, deadline_file, cleanup_buffer):
    """Run a command while polling the mutable deadline and capturing its output."""
    process = subprocess.Popen(
        command,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        deadline = read_active_deadline(deadline_file)
        timeout = None
        if deadline is not None:
            remaining = deadline - cleanup_buffer - time.time()
            if remaining <= 0:
                terminate_process_group(process)
                raise BudgetExpired("reservation deadline reached")
            timeout = min(remaining, 5)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            terminate_process_group(process)
            raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def capacity_snapshot(path):
    """Capture total, used, and free bytes for one mounted target filesystem."""
    usage = shutil.disk_usage(path)
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def verify_inventory_mount(path, target):
    """Require an active XFS mount whose source device matches the inventory."""
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        capture_output=True,
        text=True,
        check=True,
    )
    filesystems = json.loads(result.stdout).get("filesystems", [])
    if len(filesystems) != 1:
        raise RuntimeError(f"could not identify one mounted filesystem for {path}")
    mounted = filesystems[0]
    actual_source = mounted["source"].split("[", 1)[0]
    if os.path.realpath(actual_source) != os.path.realpath(target["device"]):
        raise RuntimeError(
            f"mount source mismatch for {path}: expected {target['device']}, found {mounted['source']}"
        )
    if mounted["fstype"].lower() != "xfs":
        raise RuntimeError(f"filesystem mismatch for {path}: expected xfs, found {mounted['fstype']}")
    return mounted


def check_cache_drop_permission():
    """Fail before benchmarking unless root or non-interactive sudo is available."""
    if os.geteuid() == 0:
        return
    try:
        subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cache dropping requires root or a valid non-interactive sudo credential; "
            "run the runner with sudo or configure narrowly scoped NOPASSWD access"
        ) from error


def drop_caches(deadline_file, cleanup_buffer):
    """Flush dirty data, then request page-cache, dentry, and inode eviction."""
    run_with_deadline(["sync"], deadline_file, cleanup_buffer)
    if os.geteuid() == 0:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
    else:
        run_with_deadline(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            deadline_file,
            cleanup_buffer,
        )


def error_record(error, phase, context=None):
    """Convert an exception and benchmark context into manifest-safe JSON."""
    record = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "phase": phase,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if context:
        record["context"] = context
    if getattr(error, "stdout", None):
        record["stdout"] = error.stdout
    if getattr(error, "stderr", None):
        record["stderr"] = error.stderr
    return record


def fio_version():
    """Return the installed FIO version, or null when FIO is unavailable."""
    try:
        result = subprocess.run(["fio", "--version"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
