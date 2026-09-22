"""Durable files, deadlines, and safe path helpers for network benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time


class BudgetExpired(Exception):
    """The reservation has no usable time left for another atomic unit."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text())


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        make_directory(path.parent)
        path.mkdir(exist_ok=True)
        sync_directory(path.parent)


def atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def duration(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", value)
    if not match:
        raise ValueError("use a positive duration such as 30s, 5m or 3h")
    seconds = float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be positive and finite")
    return seconds


def set_deadline(path: Path, *, time_limit: float | None = None,
                 deadline: float | None = None, extend: float | None = None) -> None:
    if sum(value is not None for value in (time_limit, deadline, extend)) != 1:
        raise ValueError("specify exactly one time limit, deadline or extension")
    if extend is not None:
        current = read_json(path)
        epoch = current["deadline_epoch"] + extend
    else:
        epoch = deadline if deadline is not None else time.time() + time_limit
    if type(epoch) not in (int, float) or not math.isfinite(epoch) or epoch <= time.time():
        raise ValueError("deadline must be a finite future time")
    atomic_json(path, {"deadline_epoch": epoch, "updated_at": now()})


def remaining_seconds(path: Path, cleanup_seconds: float) -> float:
    value = read_json(path)["deadline_epoch"]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("deadline_epoch must be finite")
    return value - time.time() - cleanup_seconds


def safe_artifact(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"artifact escapes results directory: {relative}")
    return path
