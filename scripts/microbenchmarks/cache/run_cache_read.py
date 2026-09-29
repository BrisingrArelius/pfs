#!/usr/bin/env python3
"""Measure HDD/SSD cache read paths through anjuna2's existing BeeGFS mount."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from cache_common import (SIZE, TARGETS, cache_config_matches, drop,
                          network_interface, new_run, prepared_run, record, remote,
                          run, target_file, targets)

STATES = ("backend", "server_ram", "client_ram")
VERIFICATION = "traffic_counters_v1"


def make_file(directory, target, placement=None):
    """Create and return one 8-GiB file actually assigned to target.

    Select its target before filling it and verifying size and layout.
    """
    path = target_file(directory, target, placement)
    run(["dd", "if=/dev/zero", f"of={path}", "bs=1M", "count=8192",
         "oflag=direct", "conv=fsync", "status=none"])
    if path.stat().st_size != SIZE or targets(path) != [target]:
        raise RuntimeError(f"File on target {target} has wrong size or layout")
    return path


def counters(interface, device):
    """Return cumulative host-wide network and device-read byte counters.

    Call before and after IOR; subtract corresponding values for measured traffic.
    """
    net = int((Path("/sys/class/net") / interface / "statistics/rx_bytes").read_text())
    sent = int(remote("colva1", "cat /sys/class/net/enp7s0/statistics/tx_bytes").strip())
    disk = remote("colva1", f"cat /sys/class/block/{device}/stat").split()
    return {"client_network": net, "server_network": sent,
            "device_read": int(disk[2]) * 512}


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
        record(folder / "counters.json", {"idle": idle, "before": before, "after": after})
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
        "backend": network >= .8 and server >= .8 and backend >= .8,
        "server_ram": network >= .8 and server >= .8 and backend <= .2,
        "client_ram": network <= .2 and server <= .2 and backend <= .2,
    }[state]
    result = {"mode": mode, "medium": medium, "target": TARGETS[medium][0], "intended": state,
              "achieved": state if verified else "unverified", "MiB_per_second": rate,
              "verification": VERIFICATION,
              "network_ratio": network, "server_network_ratio": server,
              "backend_ratio": backend, "quiet_before": quiet,
              "seconds": time.time() - started}
    record(folder / "result.json", result)
    return result


def benchmark(run_id, pilot, mode, results):
    """Measure read cases in a verified, lock-protected BeeGFS namespace."""
    with prepared_run(run_id, pilot, mode, results) as namespace:
        files = {medium: make_file(namespace / medium.lower(), target,
                                   results / f"{medium.lower()}-placement.json")
                 for medium, (target, _) in TARGETS.items()}
        interface = network_interface()
        planned = cases(pilot, mode)
        results_so_far = []
        for index, (medium, state) in enumerate(planned, 1):
            result = measure(index, medium, state, files[medium], interface, mode, results)
            results_so_far.append(result)
            record(results / "results.json", results_so_far)
            print(f"{index}/{len(planned)} {mode} {medium} {state}: "
                  f"{result['MiB_per_second']:.1f} MiB/s ({result['achieved']})", flush=True)
    return 0


def main(argv=None):
    """Parse the read CLI and run on anjuna2 without changing its mount."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("buffered", "native"), required=True,
                        help="Must match the existing BeeGFS client's effective mode")
    parser.add_argument("--pilot", action="store_true", help="Run one case per state and medium")
    args = parser.parse_args(argv)
    try:
        results = new_run(args.run_id, args.mode, args.pilot)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    try:
        return benchmark(args.run_id, args.pilot, args.mode, results)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cache benchmark stopped; saved raw results remain in {results}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
