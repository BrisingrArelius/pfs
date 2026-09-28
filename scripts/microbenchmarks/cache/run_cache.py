#!/usr/bin/env python3
"""Run normal BeeGFS reads from HDD/SSD, server RAM, and client RAM on anjuna2."""

import argparse
import ctypes
import fcntl
import json
import mmap
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
TARGETS = {"HDD": (101, "sdb1"), "SSD": (104, "nvme1n1p1")}
STATES = ("backend", "server_ram", "client_ram")
SIZE = 8 * 1024**3


def run(argv, *, user=None, timeout=600, output=None):
    if user:
        argv = ["runuser", "-u", user, "--", *argv]
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
    account = os.environ.get("SUDO_USER")
    if not account:
        raise RuntimeError("Cannot identify the SSH account used to launch the benchmark")
    return run(["ssh", "-o", "BatchMode=yes", host, script],
               user=account, timeout=90)


def ctl(*args):
    return run(["/usr/sbin/beegfs-ctl", *args])


def record(path, data):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def check_cluster():
    if socket.gethostname().split(".")[0] != "anjuna2":
        raise RuntimeError("Run on anjuna2")
    if run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(SHARED)]).strip() != "beegfs":
        raise RuntimeError("/mnt/beegfs/pfs is not BeeGFS")
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
            modes = remote(host, "for f in /proc/fs/beegfs/*/netbench_mode; do cat \"$f\"; done").split()
        else:
            modes = [p.read_text() for p in Path("/proc/fs/beegfs").glob("*/netbench_mode")]
        if not modes or any(str(value).strip() != "0" for value in modes):
            raise RuntimeError(f"NetBench enabled on {host or 'anjuna2'}")
    for host in (None, "colva1"):
        text = remote(host, "cat /proc/meminfo") if host else Path("/proc/meminfo").read_text()
        match = re.search(r"(?m)^MemAvailable:\s+(\d+) kB$", text)
        if not match or int(match.group(1)) * 1024 < 16 * 1024**3:
            raise RuntimeError(f"Insufficient RAM on {host or 'anjuna2'}")


def native_mount(results):
    original = Path("/etc/beegfs/beegfs-client.conf").read_text()
    regex = re.compile(r"^(\s*tuneFileCacheType\s*=\s*)buffered(\s*(?:#.*)?)$", re.MULTILINE)
    if len(regex.findall(original)) != 1:
        raise RuntimeError("Existing client is not explicitly configured as buffered")
    config, mount = results / "native.conf", results / "native-mount"
    config.write_text(regex.sub(r"\g<1>native\2", original))
    config.chmod(0o600)
    mount.mkdir()
    existing = list(Path("/proc/fs/beegfs").glob("*/config"))
    run(["mount", "-t", "beegfs", "beegfs_nodev", "-o", f"cfgFile={config}", str(mount)])
    configs = [p.read_text() for p in Path("/proc/fs/beegfs").glob("*/config")
               if p not in existing]
    if (not any(re.search(r"(?m)^tuneFileCacheType\s*=\s*native\s*$", text)
                for text in configs)
            or not all(re.search(r"(?m)^tuneFileCacheType\s*=\s*buffered\s*$", p.read_text())
                       for p in existing)):
        raise RuntimeError("Private BeeGFS mount did not create a native-cache client")
    return mount


def targets(path):
    info = ctl("--getentryinfo", "--verbose", str(path))
    return [int(value) for value in re.findall(r"(?m)^\s*\+\s+(\d+)\s+@", info)]


def make_file(directory, target, user):
    directory.mkdir()
    os.chown(directory, user.pw_uid, user.pw_gid)
    ctl("--setpattern", "--numtargets=1", "--chunksize=512k", str(directory))
    for number in range(1, 513):
        path = directory / f"candidate-{number:03d}"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.fchown(descriptor, user.pw_uid, user.pw_gid)
        os.close(descriptor)
        if targets(path) == [target]:
            break
        path.unlink()
    else:
        raise RuntimeError(f"Could not obtain one-stripe file on target {target}")
    run(["dd", "if=/dev/zero", f"of={path}", "bs=1M", "count=8192",
         "oflag=direct", "conv=fsync", "status=none"], user=user.pw_name)
    if path.stat().st_size != SIZE or targets(path) != [target]:
        raise RuntimeError(f"File on target {target} has wrong size or layout")
    return path


def residency(path):
    """Percentage of this file resident in the Linux client page cache."""
    with path.open("rb", buffering=0) as source:
        with mmap.mmap(source.fileno(), 0, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapped:
            pages = (len(mapped) + os.sysconf("SC_PAGE_SIZE") - 1) // os.sysconf("SC_PAGE_SIZE")
            vector = (ctypes.c_ubyte * pages)()
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(len(mapped)), vector):
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(byte & 1 for byte in vector) / pages


def counters(interface, device):
    net = int((Path("/sys/class/net") / interface / "statistics/rx_bytes").read_text())
    sent = int(remote("colva1", "cat /sys/class/net/enp7s0/statistics/tx_bytes").strip())
    disk = remote("colva1", f"cat /sys/class/block/{device}/stat").split()
    return {"client_network": net, "server_network": sent,
            "device_read": int(disk[2]) * 512}


def drop(client_only=False):
    os.sync()
    if not client_only:
        remote("colva1", "sudo -n sh -c 'sync && printf 3 > /proc/sys/vm/drop_caches'")
    Path("/proc/sys/vm/drop_caches").write_text("3\n")


def cases(pilot):
    six = [(medium, state) for medium in TARGETS for state in STATES]
    if pilot:
        return six + [(medium, "client_ram") for medium in TARGETS]
    random.Random(20260924).shuffle(six)
    return [case for repetition in range(5) for case in six[repetition:] + six[:repetition]]


def measure(index, medium, state, path, interface, user, results):
    folder = results / f"{index:02d}-{medium.lower()}-{state}"
    folder.mkdir()
    os.chown(folder, user.pw_uid, user.pw_gid)
    drop()
    if state != "backend":
        run(["dd", f"if={path}", "of=/dev/null", "bs=1M", "count=8192",
             "iflag=fullblock", "status=none"], user=user.pw_name,
            output=folder / "warmup")
        if state == "server_ram":
            drop(client_only=True)
    cached = residency(path)
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
        run(argv, user=user.pw_name, output=folder)
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
    verified = quiet and {
        "backend": network >= .8 and server >= .8 and backend >= .8 and cached <= .1,
        "server_ram": network >= .8 and server >= .8 and backend <= .2 and cached <= .1,
        "client_ram": network <= .2 and server <= .2 and backend <= .2 and cached >= .95,
    }[state]
    result = {"medium": medium, "target": TARGETS[medium][0], "intended": state,
              "achieved": state if verified else "unverified", "MiB_per_second": rate,
              "client_residency": cached, "network_ratio": network,
              "server_network_ratio": server, "backend_ratio": backend,
              "quiet_before": quiet,
              "seconds": time.time() - started}
    record(folder / "result.json", result)
    return result


def cleanup(namespace, run_id):
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


def benchmark(run_id, pilot, results):
    if os.geteuid() != 0 or os.readlink("/proc/self/ns/mnt") == os.readlink("/proc/1/ns/mnt"):
        raise RuntimeError("Root and a private mount namespace are required")
    with (RUNS / ".cache.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_cluster()
        owner = json.loads((results / "owner.json").read_text())
        if (results.parent != RUNS or owner.get("run_id") != run_id
                or owner.get("user") != os.environ.get("SUDO_USER")):
            raise RuntimeError("Run directory owner marker differs")
        user = pwd.getpwnam(owner["user"])
        namespace = SHARED / (".cache-" + run_id)
        if namespace.exists():
            raise RuntimeError("Benchmark namespace already exists")
        namespace.mkdir()
        os.chown(namespace, user.pw_uid, user.pw_gid)
        record(namespace / "owner.json", {"run_id": run_id})
        mount = results / "native-mount"
        mounted = False
        results_so_far = []
        try:
            try:
                native_mount(results)
            finally:
                mounted = subprocess.run(["findmnt", "-n", "--mountpoint", str(mount)],
                                         capture_output=True).returncode == 0
            modes = [p.read_text().strip() for p in Path("/proc/fs/beegfs").glob("*/netbench_mode")]
            if not modes or any(mode != "0" for mode in modes):
                raise RuntimeError("NetBench enabled on a measured client")
            path = mount / "pfs" / namespace.name
            files = {medium: make_file(path / medium.lower(), target, user)
                     for medium, (target, _) in TARGETS.items()}
            route = run(["ip", "route", "get", socket.gethostbyname("colva1")])
            interface = re.search(r"\bdev\s+(\S+)", route)
            if interface is None:
                raise RuntimeError("No client network interface to colva1")
            for index, (medium, state) in enumerate(cases(pilot), 1):
                result = measure(index, medium, state, files[medium],
                                 interface.group(1), user, results)
                results_so_far.append(result)
                record(results / "results.json", results_so_far)
                print(f"{index}/{len(cases(pilot))} {medium} {state}: "
                      f"{result['MiB_per_second']:.1f} MiB/s ({result['achieved']})", flush=True)
        finally:
            try:
                if mounted:
                    run(["umount", str(mount)])
                    mounted = False
            finally:
                if mount.is_dir() and not mounted:
                    mount.rmdir()
                (results / "native.conf").unlink(missing_ok=True)
                cleanup(namespace, run_id)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot", action="store_true", help="Run eight reads rather than thirty")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cache-[A-Za-z0-9_-]+", args.run_id):
        parser.error("run ID must start with cache- and contain only letters, digits, _ or -")
    results = RUNS / args.run_id
    try:
        if args.inside:
            return benchmark(args.run_id, args.pilot, results)
        if socket.gethostname().split(".")[0] != "anjuna2" or not RUNS.is_dir():
            parser.error("run from the project checkout on anjuna2")
        if os.geteuid() == 0:
            parser.error("run as your normal account; the script invokes sudo for its private mount")
        results.mkdir(mode=0o700)
        record(results / "owner.json", {"run_id": args.run_id,
                                         "user": pwd.getpwuid(os.getuid()).pw_name})
        return subprocess.run(["sudo", "-n", "unshare", "--mount", "--propagation", "private",
                               sys.executable, "-B", str(Path(__file__).resolve()),
                               "--run-id", args.run_id, "--inside", *( ["--pilot"] if args.pilot else [])],
                              check=False).returncode
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cache benchmark stopped; saved raw results remain in {results}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
