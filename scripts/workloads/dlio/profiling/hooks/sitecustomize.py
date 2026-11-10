"""Instrumentation only: record rank/worker identity and DXT clock anchors.

Loaded only when run_profiles.py adds this directory to the rank PYTHONPATH.
Per-process recording gives rank processes and reader children separate logs;
the preflight verifies this with the selected existing Darshan runtime.
"""

import atexit
import json
import os
from pathlib import Path
import importlib.machinery
import runpy
import sys
import time


def install():
    case = Path(os.environ["PFS_DXT_CASE"])
    rank = int(os.environ.get("PFS_DXT_RANK", os.environ.get("PMI_RANK",
                   os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))))
    if rank < 0:
        raise RuntimeError("No MPI rank or inherited PFS_DXT_RANK")
    owner = int(os.environ.get("PFS_DXT_OWNER_PID", os.getpid()))
    os.environ["PFS_DXT_RANK"] = str(rank)
    os.environ["PFS_DXT_OWNER_PID"] = str(owner)
    state = {}

    def save():
        path = case / "anchors" / f"pid-{os.getpid()}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state) + "\n")
        temporary.replace(path)

    def marker(kind):
        path = case / "anchors" / f"pid-{os.getpid()}.anchor-{kind}"
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            before = time.time_ns()
            os.write(fd, b"PFS-DXT-clock-anchor\n")
            after = time.time_ns()
        finally:
            os.close(fd)
        state[kind] = {"path": str(path), "wall_before_ns": before,
                       "wall_after_ns": after, "monotonic_ns": time.monotonic_ns()}
        save()

    def begin():
        state.clear()
        state.update(pid=os.getpid(), ppid=os.getppid(), owner_pid=owner, rank=rank,
                     role="rank" if os.getpid() == owner else "worker",
                     hostname=os.uname().nodename)
        # Let Darshan include the actual PID in its native filename. An
        # inherited fixed filename lets a utility child claim the parent's
        # log before the parent exits, losing the actual workload trace.
        os.environ.pop("DARSHAN_LOGFILE", None)
        marker("start")

    def finish():
        if "end" not in state:
            marker("end")

    original_exit = os._exit

    def traced_exit(status):
        try:
            finish()
        finally:
            original_exit(status)

    begin()
    os.register_at_fork(after_in_child=begin)
    os._exit = traced_exit
    atexit.register(finish)


if os.environ.get("PFS_DXT_CASE"):
    try:
        original = importlib.machinery.PathFinder.find_spec("sitecustomize", [path for path in sys.path
            if Path(path or ".").resolve() != Path(__file__).resolve().parent])
        if original and original.origin and Path(original.origin).resolve() != Path(__file__).resolve():
            runpy.run_path(original.origin, run_name="_pfs_original_sitecustomize")
        install()
    except BaseException as error:
        # Python otherwise prints a sitecustomize error and keeps running,
        # producing an apparently successful but uninstrumented workload.
        os.write(2, f"PFS DXT instrumentation failed: {error}\n".encode())
        os._exit(87)
