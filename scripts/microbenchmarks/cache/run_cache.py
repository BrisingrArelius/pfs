#!/usr/bin/env python3
"""Measure HDD/SSD cache paths through anjuna2's existing BeeGFS mount."""

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import pwd
import random
import re
import signal
import socket
import subprocess
import sys
import time


RUNS = Path(__file__).resolve().parents[3] / "results/microbenchmarks/runs"
SHARED = Path("/mnt/beegfs/pfs")
CLIENT_CONFIG = "/etc/beegfs/beegfs-client.conf"
TARGETS = {"HDD": (101, "sdb1"), "SSD": (104, "nvme1n1p1")}
STATES = ("backend", "server_ram", "client_ram")
SIZE = 8 * 1024**3


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


def check_cluster(mode):
    """Require the selected active mode and the expected cluster inventory.

    Return None when host, mount, targets, NetBench, RAM and sudo checks pass;
    raise on a mismatch before creating BeeGFS benchmark data.
    """
    if socket.gethostname().split(".")[0] != "anjuna2":
        raise RuntimeError("Run on anjuna2")
    if run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(SHARED)]).strip() != "beegfs":
        raise RuntimeError("/mnt/beegfs/pfs is not BeeGFS")
    configs = list(Path("/proc/fs/beegfs").glob("*/config"))
    if (not configs or any(not re.search(rf"(?m)^tuneFileCacheType\s*=\s*{mode}\s*$",
                               config.read_text()) for config in configs)):
        raise RuntimeError(f"The active anjuna2 BeeGFS client must be configured as {mode}")
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


def make_file(directory, target):
    """Create and return one 8-GiB file actually assigned to target.

    directory must be a new run-owned path. Set its one-target pattern, discard
    wrong-target empty candidates, then fill and verify the selected file.
    """
    directory.mkdir()
    ctl("--setpattern", "--numtargets=1", "--chunksize=512k", str(directory))
    for number in range(1, 513):
        path = directory / f"candidate-{number:03d}"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        if targets(path) == [target]:
            break
        path.unlink()
    else:
        raise RuntimeError(f"Could not obtain one-stripe file on target {target}")
    run(["dd", "if=/dev/zero", f"of={path}", "bs=1M", "count=8192",
         "oflag=direct", "conv=fsync", "status=none"])
    if path.stat().st_size != SIZE or targets(path) != [target]:
        raise RuntimeError(f"File on target {target} has wrong size or layout")
    return path


def residency(path):
    """Return the fraction (0..1) of path's pages resident in client page cache.

    Used only in native mode; Linux mincore reports page state without reading data.

    The mapping must be read-only. A writable MAP_PRIVATE mapping makes the BeeGFS
    client drop the file's cached pages, which emptied the cache this measures and
    forced every client-RAM case to unverified. mincore needs the raw address, which
    the mmap module cannot expose for a read-only buffer, so mmap(2) is called
    directly.
    """
    length = path.stat().st_size
    pages = (length + os.sysconf("SC_PAGE_SIZE") - 1) // os.sysconf("SC_PAGE_SIZE")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    with path.open("rb", buffering=0) as source:
        address = libc.mmap(None, length, 0x1, 0x2, source.fileno(), 0)
    if address is None or address == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap failed")
    try:
        vector = (ctypes.c_ubyte * pages)()
        if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(length), vector):
            raise OSError(ctypes.get_errno(), "mincore failed")
        return sum(byte & 1 for byte in vector) / pages
    finally:
        libc.munmap(ctypes.c_void_p(address), ctypes.c_size_t(length))


def counters(interface, device):
    """Return cumulative host-wide network and device-read byte counters.

    Call before and after IOR; subtract corresponding values for measured traffic.
    """
    net = int((Path("/sys/class/net") / interface / "statistics/rx_bytes").read_text())
    sent = int(remote("colva1", "cat /sys/class/net/enp7s0/statistics/tx_bytes").strip())
    disk = remote("colva1", f"cat /sys/class/block/{device}/stat").split()
    return {"client_network": net, "server_network": sent,
            "device_read": int(disk[2]) * 512}


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


def cases(pilot, mode):
    """Return ordered (medium, state) pairs: 4/20 buffered or 8/30 native."""
    states = STATES if mode == "native" else STATES[:2]
    combinations = [(medium, state) for medium in TARGETS for state in states]
    if pilot:
        return (combinations + [(medium, "client_ram") for medium in TARGETS]
                if mode == "native" else combinations)
    random.Random(20260924).shuffle(combinations)
    return [case for repetition in range(5)
            for case in combinations[repetition:] + combinations[:repetition]]


def measure(index, medium, state, path, interface, mode, results):
    """Prepare one cache state, run one IOR read, and return its result dict.

    path is an existing verified 8-GiB file. Save native IOR output, counters,
    warm-up evidence and the achieved/unverified label below results.
    """
    folder = results / f"{index:02d}-{medium.lower()}-{state}"
    folder.mkdir()
    drop()
    if state != "backend":
        run(["dd", f"if={path}", "of=/dev/null", "bs=1M", "count=8192",
             "iflag=fullblock", "status=none"], output=folder / "warmup")
        if state == "server_ram":
            drop(client_only=True)
        else:
            drop(server_only=True)
    cached = residency(path) if mode == "native" else None
    identity = path.stat()
    idle = counters(interface, TARGETS[medium][1])
    time.sleep(1)
    before = counters(interface, TARGETS[medium][1])
    quiet = all(0 <= before[key] - idle[key] < SIZE // 100 for key in before)
    summary = folder / "ior.json"
    argv = ["/usr/bin/mpirun", "-np", "1", "/usr/local/bin/ior", "-a", "POSIX",
            "-r", "-E", "-k", "-g", "-t", "1m", "-b", "8g", "-s", "1",
            "-i", "1", "-o", str(path), "-O", "summaryFormat=JSON",
            "-O", f"summaryFile={summary}"]
    record(folder / "command.json", argv)
    started = time.time()
    try:
        run(argv, output=folder)
    finally:
        after = counters(interface, TARGETS[medium][1])
        record(folder / "counters.json", {"before": before, "after": after})
    if (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) != (
            identity.st_ino, identity.st_size, identity.st_mtime_ns):
        raise RuntimeError("IOR modified its input file")
    data = json.loads(summary.read_text())
    rows = data.get("summary")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("Unrecognized native IOR JSON summary; raw files are preserved")
    row = rows[0]
    if (row.get("API") != "POSIX" or row.get("operation") != "read"
            or row.get("numTasks") != 1 or row.get("blockSize") != SIZE
            or row.get("transferSize") != 1024**2 or row.get("xsizeMiB") != 8192):
        raise RuntimeError("IOR read does not match the 8-GiB POSIX protocol")
    rate = row.get("bwMeanMIB")
    if not isinstance(rate, (float, int)) or not 0 < rate < float("inf"):
        raise RuntimeError("IOR did not report valid bandwidth")
    network = (after["client_network"] - before["client_network"]) / SIZE
    server = (after["server_network"] - before["server_network"]) / SIZE
    backend = (after["device_read"] - before["device_read"]) / SIZE
    if mode == "buffered":
        # BeeGFS buffered mode has its own small buffers: mincore is not a
        # measurement of their residency. Require traffic evidence instead.
        verified = quiet and {
            "backend": network >= .8 and server >= .8 and backend >= .8,
            "server_ram": network >= .8 and server >= .8 and backend <= .2,
        }[state]
    else:
        verified = quiet and {
            "backend": network >= .8 and server >= .8 and backend >= .8 and cached <= .1,
            "server_ram": network >= .8 and server >= .8 and backend <= .2 and cached <= .1,
            "client_ram": network <= .2 and server <= .2 and backend <= .2 and cached >= .95,
        }[state]
    result = {"mode": mode, "medium": medium, "target": TARGETS[medium][0], "intended": state,
              "achieved": state if verified else "unverified", "MiB_per_second": rate,
              "client_residency": cached, "network_ratio": network,
              "server_network_ratio": server, "backend_ratio": backend,
              "quiet_before": quiet,
              "seconds": time.time() - started}
    record(folder / "result.json", result)
    return result


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


def benchmark(run_id, pilot, mode, results):
    """Check, prepare, measure sequential cases, and clean up; return 0.

    The local file lock excludes another copy of this script, not other workloads.
    Individual results may still be 'unverified'; command failures raise.
    """
    with (RUNS / ".cache.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_cluster(mode)
        owner = json.loads((results / "owner.json").read_text())
        if (results.parent != RUNS or owner.get("run_id") != run_id
                or owner.get("user") != pwd.getpwuid(os.getuid()).pw_name
                or owner.get("mode") != mode or owner.get("pilot") != pilot):
            raise RuntimeError("Run directory owner marker differs")
        namespace = SHARED / (".cache-" + run_id)
        if namespace.exists():
            raise RuntimeError("Benchmark namespace already exists")
        namespace.mkdir()
        record(namespace / "owner.json", {"run_id": run_id})
        results_so_far = []
        try:
            files = {medium: make_file(namespace / medium.lower(), target)
                     for medium, (target, _) in TARGETS.items()}
            route = run(["ip", "route", "get", socket.gethostbyname("colva1")])
            interface = re.search(r"\bdev\s+(\S+)", route)
            if interface is None:
                raise RuntimeError("No client network interface to colva1")
            planned = cases(pilot, mode)
            for index, (medium, state) in enumerate(planned, 1):
                result = measure(index, medium, state, files[medium],
                                 interface.group(1), mode, results)
                results_so_far.append(result)
                record(results / "results.json", results_so_far)
                print(f"{index}/{len(planned)} {mode} {medium} {state}: "
                      f"{result['MiB_per_second']:.1f} MiB/s ({result['achieved']})", flush=True)
        finally:
            cleanup(namespace, run_id)
        return 0


def main(argv=None):
    """Parse CLI, create a unique run directory and return an exit status.

    Run as a normal user on anjuna2. --mode must match its existing BeeGFS mount;
    the script never changes that mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("buffered", "native"), required=True,
                        help="Must match the existing BeeGFS client's effective mode")
    parser.add_argument("--pilot", action="store_true", help="Four buffered or eight native reads")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cache-[A-Za-z0-9_-]+", args.run_id):
        parser.error("run ID must start with cache- and contain only letters, digits, _ or -")
    results = RUNS / args.run_id
    try:
        if socket.gethostname().split(".")[0] != "anjuna2" or not RUNS.is_dir():
            parser.error("run from the project checkout on anjuna2")
        if os.geteuid() == 0:
            parser.error("run as your normal account; sudo is used only to drop caches")
        results.mkdir(mode=0o700)
        record(results / "owner.json", {"run_id": args.run_id,
                                         "user": pwd.getpwuid(os.getuid()).pw_name,
                                         "mode": args.mode, "pilot": args.pilot})
        return benchmark(args.run_id, args.pilot, args.mode, results)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cache benchmark stopped; saved raw results remain in {results}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
