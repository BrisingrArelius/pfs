"""Deadline enforcement and durable files; no FIO or experiment policy."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


class BudgetExpired(Exception):
    """The reservation has no usable time left for this operation."""


def now():
    """Return a UTC timestamp for human-readable provenance."""
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    """Read one complete JSON artifact."""
    return json.loads(Path(path).read_text())


def sync_directory(path):
    """Persist directory entries after creation, rename or unlink."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_directory(path):
    """Create missing directories, persisting each new entry."""
    path = Path(path)
    if not path.exists():
        make_directory(path.parent)
        path.mkdir(exist_ok=True)
        sync_directory(path.parent)


def atomic_json(path, value):
    """Durably replace JSON without exposing a partially written checkpoint."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def duration(value):
    """Parse positive seconds or a duration suffixed with s, m or h."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", value)
    if not match:
        raise ValueError("use a positive duration such as 30s, 5m or 2h")
    seconds = float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be positive and finite")
    return seconds


def remaining_seconds(path, cleanup_seconds):
    """Reread the live deadline; invalid state never means unlimited time."""
    deadline = read_json(path)["deadline_epoch"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("deadline_epoch must be a finite number")
    return deadline - time.time() - cleanup_seconds


def set_deadline(path, *, time_limit=None, deadline=None, extend=None):
    """Create an allocation deadline or extend an existing unexpired one."""
    if sum(value is not None for value in (time_limit, deadline, extend)) != 1:
        raise ValueError("specify exactly one time limit, deadline or extension")
    if extend is not None:
        remaining = remaining_seconds(path, 0)
        if remaining <= 0:
            raise ValueError("cannot extend an expired deadline")
        epoch = read_json(path)["deadline_epoch"] + extend
    else:
        epoch = deadline if deadline is not None else time.time() + time_limit
    if not math.isfinite(epoch) or epoch <= time.time():
        raise ValueError("deadline must be in the future")
    atomic_json(path, {"deadline_epoch": epoch, "updated_at": now()})


def run_command(argv, stdout_path, stderr_path, deadline_path, cleanup_seconds,
                timeout_seconds):
    """Run/reap one process group, enforcing allocation and hard time limits."""
    if remaining_seconds(deadline_path, cleanup_seconds) <= 0:
        raise BudgetExpired("reservation cleanup window reached")
    started = time.monotonic()
    process = None
    with open(stdout_path, "xb") as stdout, open(stderr_path, "xb") as stderr:
        try:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            while True:
                available = remaining_seconds(deadline_path, cleanup_seconds)
                hard_remaining = timeout_seconds - (time.monotonic() - started)
                if available <= 0:
                    raise BudgetExpired("reservation cleanup window reached")
                if hard_remaining <= 0:
                    raise TimeoutError(f"command exceeded {timeout_seconds:g}s")
                try:
                    process.wait(timeout=min(1, available, hard_remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            elapsed = time.monotonic() - started
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, argv)
            return elapsed
        except BaseException as error:
            # This also covers a malformed live deadline and a user interrupt.
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # Reap descendants too, even if the group leader exited on TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            error.command_wall_seconds = time.monotonic() - started
            raise
        finally:
            for output in (stdout, stderr):
                output.flush()
                os.fsync(output.fileno())
            sync_directory(Path(stdout_path).parent)
