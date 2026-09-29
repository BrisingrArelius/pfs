#!/usr/bin/env python3
"""Measure BeeGFS write completion paths on anjuna2's existing mount."""

import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from cache_common import (SIZE, TARGETS, drop, network_interface, new_run,
                          prepared_run, record, remote, run, target_file, targets)

CLIENT_SIZE = 1024**3
VERIFICATION = "write_traffic_v1"


def check_remote_fsync(setting):
    """Match the CLI boolean to each live client's 0/1 or true/false value."""
    configs = list(Path("/proc/fs/beegfs").glob("*/config"))
    accepted = {"true": {"1", "true"}, "false": {"0", "false"}}[setting]
    observed = []
    for path in configs:
        match = re.search(r"(?m)^tuneRemoteFSync[ \t]*=[ \t]*(\S+)[ \t]*$",
                          path.read_text())
        observed.append(match.group(1).lower() if match else None)
    if not observed or any(value not in accepted for value in observed):
        raise RuntimeError(
            f"Active client must have tuneRemoteFSync={setting}; observed {observed}")


def cases(pilot, mode, remote_fsync):
    """Return ordered write cases for this one live client configuration."""
    state = "server_disk" if remote_fsync == "true" else "server_ram"
    choices = [(medium, state) for medium in TARGETS]
    if mode == "native" and remote_fsync == "true":
        choices += [(medium, "client_ram") for medium in TARGETS]
    if pilot:
        return choices
    random.Random(20260929).shuffle(choices)
    return [case for repetition in range(5)
            for case in choices[repetition:] + choices[:repetition]]


def counters(interface, device):
    """Snapshot host-wide client TX, server RX and target sectors written."""
    sent = int((Path("/sys/class/net") / interface / "statistics/tx_bytes").read_text())
    received = int(remote("colva1", "cat /sys/class/net/enp7s0/statistics/rx_bytes").strip())
    sectors = remote("colva1", f"cat /sys/class/block/{device}/stat").split()
    return {"client_sent": sent, "server_received": received,
            "device_written": int(sectors[6]) * 512}


def ratios(before, after, size):
    """Return nonnegative traffic deltas as fractions of the workload size."""
    return {key + "_ratio": (after[key] - before[key]) / size for key in before}


def ior_write(path, folder, interface, device):
    """Time one 8-GiB IOR write with fsync; return rate and raw counters."""
    summary = folder / "ior.json"
    argv = ["/usr/bin/mpirun", "-np", "1", "/usr/local/bin/ior", "-a", "POSIX",
            "-w", "-E", "-k", "-g", "-e", "-t", "1m", "-b", "8g", "-s", "1",
            "-i", "1", "-o", str(path), "-O", "summaryFormat=JSON",
            "-O", f"summaryFile={summary}"]
    record(folder / "command.json", argv)
    before = counters(interface, device)
    try:
        run(argv, output=folder)
    finally:
        after = counters(interface, device)
        record(folder / "counters.json", {"before": before, "after": after})
    rows = json.loads(summary.read_text()).get("summary")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("Unrecognized IOR write summary; raw files preserved")
    row = rows[0]
    if (row.get("API") != "POSIX" or row.get("operation") != "write"
            or row.get("numTasks") != 1 or row.get("blockSize") != SIZE
            or row.get("transferSize") != 1024**2 or row.get("xsizeMiB") != 8192):
        raise RuntimeError("IOR write does not match the 8-GiB POSIX protocol")
    rate = row.get("bwMeanMIB")
    if not isinstance(rate, (int, float)) or not 0 < rate < float("inf"):
        raise RuntimeError("IOR reported invalid write bandwidth")
    return rate, row.get("MeanTime"), ratios(before, after, SIZE)


def client_write(path, folder, interface, device):
    """Time 1-GiB POSIX write calls, then fsync outside the timed interval."""
    record(folder / "workload.json", {"API": "POSIX", "operation": "write",
                                      "bytes": CLIENT_SIZE, "transferSize": 1024**2,
                                      "timed_region": "write calls only; fsync/close excluded"})
    before = counters(interface, device)
    tx_path = Path("/sys/class/net") / interface / "statistics/tx_bytes"
    descriptor = os.open(path, os.O_WRONLY)
    try:
        payload = bytes([0xA5]) * 1024**2
        start_tx = int(tx_path.read_text())
        started = time.monotonic()
        for _ in range(CLIENT_SIZE // len(payload)):
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise RuntimeError("POSIX write made no progress")
                view = view[count:]
        seconds = time.monotonic() - started
        end_tx = int(tx_path.read_text())
        sync_started = time.monotonic()
        os.fsync(descriptor)
        sync_seconds = time.monotonic() - sync_started
    finally:
        os.close(descriptor)
    after = counters(interface, device)
    record(folder / "counters.json", {"before": before, "after_sync": after,
                                      "timed_start_tx": start_tx, "timed_end_tx": end_tx})
    record(folder / "posix.json", {"bytes": CLIENT_SIZE, "write_seconds": seconds,
                                   "fsync_seconds": sync_seconds})
    evidence = ratios(before, after, CLIENT_SIZE)
    evidence["timed_client_sent_ratio"] = (end_tx - start_tx) / CLIENT_SIZE
    return CLIENT_SIZE / 1024**2 / seconds, seconds, evidence


def measure(index, medium, state, mode, remote_fsync, interface, namespace, results):
    """Prepare one target and measure a single write; retain raw evidence."""
    folder = results / f"{index:02d}-{medium.lower()}-{state}"
    folder.mkdir()
    path = target_file(namespace / medium.lower(), TARGETS[medium][0])
    size = CLIENT_SIZE if state == "client_ram" else SIZE
    try:
        drop()
        idle = counters(interface, TARGETS[medium][1])
        time.sleep(1)
        baseline = counters(interface, TARGETS[medium][1])
        quiet = all(0 <= baseline[key] - idle[key] < size // 100 for key in idle)
        record(folder / "idle_counters.json", {"idle": idle, "before": baseline})
        if state == "client_ram":
            rate, seconds, evidence = client_write(
                path, folder, interface, TARGETS[medium][1])
        else:
            rate, seconds, evidence = ior_write(
                path, folder, interface, TARGETS[medium][1])
        if path.stat().st_size != size or targets(path) != [TARGETS[medium][0]]:
            raise RuntimeError("Written file has wrong size or storage target")
        network = (evidence["client_sent_ratio"] >= .8
                   and evidence["server_received_ratio"] >= .8)
        disk = evidence["device_written_ratio"] >= .8
        verified = quiet and network and (
            (state == "server_ram" and remote_fsync == "false")
            or (state == "server_disk" and remote_fsync == "true" and disk)
            or (state == "client_ram" and mode == "native" and remote_fsync == "true"
                and disk and 0 <= evidence["timed_client_sent_ratio"] <= .2))
        result = {"operation": "write", "mode": mode, "medium": medium,
                  "target": TARGETS[medium][0], "intended": state,
                  "achieved": state if verified else "unverified", "bytes": size,
                  "MiB_per_second": rate, "seconds": seconds,
                  "remote_fsync": remote_fsync, "verification": VERIFICATION,
                  "quiet_before": quiet, **evidence}
        record(folder / "result.json", result)
        return result
    finally:
        path.unlink(missing_ok=True)


def benchmark(run_id, pilot, mode, remote_fsync, results):
    """Run this policy's matrix under the shared lock and safety preflight."""
    with prepared_run(run_id, pilot, mode, results,
                      extra_owner={"operation": "write", "remote_fsync": remote_fsync},
                      extra_check=lambda: check_remote_fsync(remote_fsync)) as namespace:
        interface = network_interface()
        planned = cases(pilot, mode, remote_fsync)
        completed = []
        for index, (medium, state) in enumerate(planned, 1):
            result = measure(index, medium, state, mode, remote_fsync,
                             interface, namespace, results)
            completed.append(result)
            record(results / "results.json", completed)
            print(f"{index}/{len(planned)} {mode} fsync={remote_fsync} {medium} {state}: "
                  f"{result['MiB_per_second']:.1f} MiB/s ({result['achieved']})", flush=True)
    return 0


def main(argv=None):
    """Require the current mount policy; never edit BeeGFS configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("buffered", "native"), required=True)
    parser.add_argument("--remote-fsync", choices=("true", "false"), required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    try:
        results = new_run(args.run_id, args.mode, args.pilot,
                          {"operation": "write", "remote_fsync": args.remote_fsync})
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    try:
        return benchmark(args.run_id, args.pilot, args.mode, args.remote_fsync, results)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Write benchmark stopped; raw results remain in {results}: {error}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
