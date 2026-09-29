#!/usr/bin/env python3
"""Shared BeeGFS cache benchmark preparation and safety checks."""

import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
from contextlib import contextmanager

RUNS = Path(__file__).resolve().parents[3] / "results/microbenchmarks/runs"
SHARED = Path("/mnt/beegfs/pfs")
CLIENT_CONFIG = "/etc/beegfs/beegfs-client.conf"
TARGETS = {"HDD": (101, "sdb1"), "SSD": (104, "nvme1n1p1")}
SIZE = 8 * 1024**3
NATIVE_CACHE_THRESHOLD = 2 * 1024**2


def run(argv, *, timeout=600, output=None):
    """Execute argv as the current user and return stdout text.

    If output is set, save both streams there. Kill the process group and raise
    RuntimeError on timeout; raise RuntimeError on a nonzero exit.
    """
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        if output:
            output.mkdir(parents=True, exist_ok=True)
            (output / "stdout.txt").write_bytes(stdout)
            (output / "stderr.txt").write_bytes(stderr)
        raise RuntimeError(f"Timed out: {argv[0]}")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "stdout.txt").write_bytes(stdout)
        (output / "stderr.txt").write_bytes(stderr)
    if process.returncode:
        raise RuntimeError(f"{argv[0]} failed: {stderr.decode(errors='replace')[-800:]}")
    return stdout.decode(errors="replace")


def remote(host, script):
    """Run a command over non-interactive SSH and return its stdout text."""
    return run(["ssh", "-o", "BatchMode=yes", host, script], timeout=90)


def ctl(*args):
    """Run beegfs-ctl with the installed client's root-readable config.

    This host's CLI does not load connDisableAuthentication from that config
    by default. Use its proven --cfgFile form for reads and directory changes.
    """
    return run(["sudo", "-n", "/usr/sbin/beegfs-ctl",
                f"--cfgFile={CLIENT_CONFIG}", *args])


def record(path, data):
    """Replace path with JSON data via a sibling temporary file; return None."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def cache_config_matches(text, mode):
    """Check the effective cache mode and native I/O threshold from procfs."""
    if not re.search(rf"(?m)^tuneFileCacheType\s*=\s*{mode}\s*$", text):
        return False
    if mode == "native":
        match = re.search(r"(?m)^tuneFileCacheBufSize\s*=\s*(\d+)\s*$", text)
        return bool(match and int(match.group(1)) >= NATIVE_CACHE_THRESHOLD)
    return True


def check_cluster(mode):
    """Require the active mode, native threshold and expected inventory.

    Return None when host, mount, targets, NetBench, RAM and sudo checks pass;
    raise on a mismatch before creating BeeGFS benchmark data.
    """
    if socket.gethostname().split(".")[0] != "anjuna2":
        raise RuntimeError("Run on anjuna2")
    if run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(SHARED)]).strip() != "beegfs":
        raise RuntimeError("/mnt/beegfs/pfs is not BeeGFS")
    configs = list(Path("/proc/fs/beegfs").glob("*/config"))
    if not configs or any(not cache_config_matches(config.read_text(), mode)
                          for config in configs):
        raise RuntimeError(f"The active anjuna2 BeeGFS client must be {mode}"
                           + (" with tuneFileCacheBufSize >= 2097152" if mode == "native" else ""))
    states = ctl("--listtargets", "--longnodes", "--state")
    for target, _ in TARGETS.values():
        if not re.search(rf"(?m)^\s*{target}\s+Online\s+Good\s+.*colva1\b", states):
            raise RuntimeError(f"Target {target} is not Online/Good on colva1")
    numbers = remote("colva1", "sudo -n cat /mnt/hdd2/beegfs_storage/targetNumID "
                     "/mnt/nvme0/beegfs_storage/targetNumID")
    if numbers.split() != ["101", "104"]:
        raise RuntimeError("Target-to-device mapping has changed")
    for host in (None, "anjuna3"):
        if host:
            modes = remote(host,
                           'for f in /proc/fs/beegfs/*/netbench_mode; do '
                           'test -r "$f" || exit 1; '
                           'IFS= read -r mode < "$f"; printf "%s\\n" "$mode"; done').splitlines()
        else:
            modes = [p.read_text().partition("\n")[0] for p in
                     Path("/proc/fs/beegfs").glob("*/netbench_mode")]
        if not modes or any(value.strip() != "0" for value in modes):
            raise RuntimeError(f"NetBench status missing or enabled on {host or 'anjuna2'}: {modes!r}")
    for host in (None, "colva1"):
        text = remote(host, "cat /proc/meminfo") if host else Path("/proc/meminfo").read_text()
        match = re.search(r"(?m)^MemAvailable:\s+(\d+) kB$", text)
        if not match or int(match.group(1)) * 1024 < 16 * 1024**3:
            raise RuntimeError(f"Insufficient RAM on {host or 'anjuna2'}")
    run(["sudo", "-n", "true"])
    remote("colva1", "sudo -n true")


def targets(path):
    """Return integer storage target IDs from BeeGFS entry information for path."""
    info = ctl("--getentryinfo", "--verbose", str(path))
    return [int(value) for value in re.findall(r"(?m)^\s*\+\s+(\d+)\s+@", info)]


def target_file(directory, target):
    """Create an empty file on one verified storage target in a run-owned directory."""
    if not directory.exists():
        directory.mkdir()
        ctl("--setpattern", "--numtargets=1", "--chunksize=512k", str(directory))
    elif directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("Target directory is not a real directory")
    for number in range(1, 513):
        path = directory / f"candidate-{number:03d}"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        if targets(path) == [target]:
            return path
        path.unlink()
    raise RuntimeError(f"Could not obtain one-stripe file on target {target}")


def drop(client_only=False, server_only=False):
    """Sync and drop caches on both hosts, or only the selected host.

    Requires non-interactive sudo on anjuna2/colva1. Affects the whole host,
    not just our file. Never set both selector flags.
    """
    if client_only and server_only:
        raise ValueError("Cannot drop both client-only and server-only")
    if not server_only:
        os.sync()
    if not client_only:
        remote("colva1", "sudo -n sh -c 'sync && printf 3 > /proc/sys/vm/drop_caches'")
    if not server_only:
        run(["sudo", "-n", "sh", "-c", "printf 3 > /proc/sys/vm/drop_caches"])


def cleanup(namespace, run_id):
    """Remove only run_id's marker-owned BeeGFS files and directories.

    Reject symlinks, an unexpected marker or unexpected directory contents.
    """
    marker = namespace / "owner.json"
    if (namespace.parent != SHARED or namespace.is_symlink() or marker.is_symlink()
            or json.loads(marker.read_text()) != {"run_id": run_id}):
        raise RuntimeError("Refusing to clean up unowned BeeGFS files")
    for medium in ("hdd", "ssd"):
        directory = namespace / medium
        if not directory.exists():
            continue
        if directory.is_symlink() or any(
                not re.fullmatch(r"candidate-\d{3}", item.name)
                or item.is_symlink() or not item.is_file() for item in directory.iterdir()):
            raise RuntimeError("Unexpected file in benchmark directory")
        for item in directory.iterdir():
            item.unlink()
        directory.rmdir()
    marker.unlink()
    namespace.rmdir()


@contextmanager
def prepared_run(run_id, pilot, mode, results, extra_owner=None, extra_check=None):
    """Lock, preflight, verify ownership, create namespace, and clean up."""
    with (RUNS / ".cache.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_cluster(mode)
        if extra_check is not None:
            extra_check()
        expected = {"run_id": run_id, "user": pwd.getpwuid(os.getuid()).pw_name,
                    "mode": mode, "pilot": pilot, **(extra_owner or {})}
        if results.parent != RUNS or json.loads((results / "owner.json").read_text()) != expected:
            raise RuntimeError("Run directory owner marker differs")
        configs = []
        for path in sorted(Path("/proc/fs/beegfs").glob("*/config")):
            text = path.read_text()
            configs.append({"path": str(path), "settings": {
                key: match.group(1) if (match := re.search(
                    rf"(?m)^{key}\s*=\s*(\S+)", text)) else None
                for key in ("tuneFileCacheType", "tuneFileCacheBufSize", "tuneRemoteFSync")}})
        record(results / "live_config.json", configs)
        namespace = SHARED / (".cache-" + run_id)
        if namespace.exists():
            raise RuntimeError("Benchmark namespace already exists")
        namespace.mkdir()
        record(namespace / "owner.json", {"run_id": run_id})
        try:
            yield namespace
        finally:
            cleanup(namespace, run_id)


def network_interface():
    """Return the client interface routed toward colva1."""
    route = run(["ip", "route", "get", socket.gethostbyname("colva1")])
    match = re.search(r"\bdev\s+(\S+)", route)
    if match is None:
        raise RuntimeError("No client network interface to colva1")
    return match.group(1)


def new_run(run_id, mode, pilot, extra_owner=None):
    """Create a unique local run directory with its ownership marker."""
    if not re.fullmatch(r"cache-[A-Za-z0-9_-]+", run_id):
        raise ValueError("run ID must start with cache- and contain only letters, digits, _ or -")
    if socket.gethostname().split(".")[0] != "anjuna2" or not RUNS.is_dir():
        raise RuntimeError("run from the project checkout on anjuna2")
    if os.geteuid() == 0:
        raise RuntimeError("run as your normal account; sudo is used only to drop caches")
    results = RUNS / run_id
    results.mkdir(mode=0o700)
    record(results / "owner.json", {"run_id": run_id,
                                    "user": pwd.getpwuid(os.getuid()).pw_name,
                                    "mode": mode, "pilot": pilot, **(extra_owner or {})})
    return results
