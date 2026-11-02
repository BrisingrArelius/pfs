#!/usr/bin/env python3
"""Check SSD throughput quietly; optionally probe a selected pool through BeeGFS."""

import argparse
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pool_common import DATA_ROOT, DEVICES, FAST, TARGETS, pattern, verify


SAMPLE = r"""
import json, os, pathlib, subprocess, sys, time

devices = sys.argv[1].split(',')
seconds = int(sys.argv[2])
def counters():
    return {device: list(map(int, (pathlib.Path('/sys/class/block') / device / 'stat').read_text().split()))
            for device in devices}

limits = {'IOReadBandwidthMax': {}, 'IOWriteBandwidthMax': {}}
text = subprocess.check_output(['systemctl', 'show', 'beegfs-storage.service',
                                '-p', 'IOReadBandwidthMax', '-p', 'IOWriteBandwidthMax'], text=True)
for line in text.splitlines():
    key, _, value = line.partition('=')
    if key in limits and value:
        path, rate = value.split()
        limits[key][path] = int(rate)

cgroup = subprocess.check_output(['systemctl', 'show', '-P', 'ControlGroup',
                                  'beegfs-storage.service'], text=True).strip()
kernel = {}
if cgroup:
    text = subprocess.check_output(['sudo', '-n', 'cat', '/sys/fs/cgroup' + cgroup + '/io.max'], text=True)
    for line in text.splitlines():
        key, *fields = line.split()
        kernel[key] = dict(field.split('=', 1) for field in fields)

start_time, before = time.monotonic(), counters()
time.sleep(seconds)
after = counters()
elapsed = time.monotonic() - start_time
results = {}
for device in devices:
    parent = '/dev/' + device
    partition = parent + 'p1'
    ids = [os.stat(path).st_rdev for path in (parent, partition)]
    kernel_caps = [kernel.get(f'{os.major(number)}:{os.minor(number)}', {}) for number in ids]
    configured_read = limits['IOReadBandwidthMax'].get(partition)
    configured_write = limits['IOWriteBandwidthMax'].get(partition)
    active = next((entry for entry in kernel_caps
                   if entry.get('rbps') == str(configured_read)
                   and entry.get('wbps') == str(configured_write)), None)
    if active is None:
        active = next((entry for entry in kernel_caps
                       if entry.get('rbps') not in (None, 'max')
                       or entry.get('wbps') not in (None, 'max')), {})
    results[device] = {
        'read_MiB_s': (after[device][2] - before[device][2]) * 512 / 1048576 / elapsed,
        'write_MiB_s': (after[device][6] - before[device][6]) * 512 / 1048576 / elapsed,
        'configured_read': configured_read,
        'configured_write': configured_write,
        'kernel_read': active.get('rbps'), 'kernel_write': active.get('wbps')}
print(json.dumps(results))
"""


COUNTERS = r"""
import json, pathlib, sys
result = {}
for device in sys.argv[1].split(','):
    fields = list(map(int, (pathlib.Path('/sys/class/block') / device / 'stat').read_text().split()))
    result[device] = {'read_bytes': fields[2] * 512, 'write_bytes': fields[6] * 512}
print(json.dumps(result))
"""


def remote_samples(program, seconds=None):
    jobs = {}
    for host in ("colva1", "colva2", "colva3"):
        devices = [Path(device).name.rsplit("p", 1)[0]
                   for node, device in DEVICES.values() if node == host]
        arguments = ["python3", "-c", program, ",".join(devices)]
        if seconds is not None:
            arguments.append(str(seconds))
        jobs[host] = subprocess.Popen(["ssh", host, shlex.join(arguments)], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
    measurements = {}
    for host, process in jobs.items():
        stdout, stderr = process.communicate(timeout=(seconds or 0) + 30)
        if process.returncode:
            raise RuntimeError(f"{host} failed to sample its SSDs: {stderr.strip()}")
        measurements[host] = json.loads(stdout)
    return time.monotonic(), measurements


def limit_status(item):
    if item["configured_read"] is None and item["configured_write"] is None \
            and item["kernel_read"] is None and item["kernel_write"] is None:
        return "none"
    if (item["configured_read"] is not None and item["configured_write"] is not None
            and item["kernel_read"] == str(item["configured_read"])
            and item["kernel_write"] == str(item["configured_write"])):
        return "matched"
    return "MISMATCH"


def show_sample(seconds, measurements):
    print(f"Selected SSDs · {seconds}-second physical-device sample (MiB/s)")
    print("Target  Host    SSD       Read / cap       Write / cap      Kernel limit")
    for target in TARGETS:
        host, path = DEVICES[target]
        device = Path(path).name.rsplit("p", 1)[0]
        item = measurements[host][device]
        def cap(key):
            return f"{item[key] / 1048576:.0f}" if item[key] is not None else "—"
        print(f"{target:<7} {host:<7} {device:<9} {item['read_MiB_s']:>5.1f} / {cap('configured_read'):<5} "
              f"{item['write_MiB_s']:>5.1f} / {cap('configured_write'):<5} "
              f"{limit_status(item)}")
    print("Device-wide counters; zero I/O does not test a cap. Sample during DLIO reads/writes.")


def rates(before, after, direction):
    start, old = before
    end, new = after
    elapsed = end - start
    if elapsed <= 0:
        raise RuntimeError("Invalid device sample interval")
    result = {}
    for target, (host, path) in DEVICES.items():
        device = Path(path).name.rsplit("p", 1)[0]
        key = f"{direction}_bytes"
        amount = new[host][device][key] - old[host][device][key]
        if amount < 0:
            raise RuntimeError(f"Device counter reset on {host}:{device}")
        result[target] = (amount / 1048576, amount / 1048576 / elapsed)
    return result


def fio(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job = json.loads(result.stdout)["jobs"][0]
    if job["error"]:
        raise RuntimeError(f"fio returned error {job['error']}: {result.stderr}")
    return job


def native_check(pool):
    if shutil.which("fio") is None:
        raise RuntimeError("fio is required for --native")
    verify(pool)
    _, configured = remote_samples(SAMPLE, 1)
    mismatches = []
    for target, (host, path) in DEVICES.items():
        device = Path(path).name.rsplit("p", 1)[0]
        item = configured[host][device]
        if limit_status(item) != "matched":
            mismatches.append(f"{target} ({host}/{device}): systemd read/write "
                              f"{item['configured_read']}/{item['configured_write']} B/s, "
                              f"kernel io.max read/write {item['kernel_read']}/{item['kernel_write']} B/s")
    if mismatches:
        raise RuntimeError("Configured limits do not match kernel io.max; probe not started:\n"
                           + "\n".join(mismatches))

    base = DATA_ROOT / "throttled_ssd" / "throughput_checks"
    base.mkdir(parents=True, exist_ok=True)
    directory = base / f"pool{pool}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    pattern(directory, pool)
    filename = directory / "probe.bin"
    print(f"Pool {pool}: probing a temporary four-stripe BeeGFS file ({filename})", flush=True)
    try:
        for operation in ("write", "read"):
            command = ["fio", f"--name=pool{pool}-{operation}", f"--filename={filename}",
                       f"--rw={operation}", "--ioengine=sync", "--direct=1", "--bs=1m",
                       "--size=2G", "--numjobs=1", "--iodepth=1", "--output-format=json"]
            if operation == "write":
                command.append("--end_fsync=1")
            before = remote_samples(COUNTERS)
            job = fio(command)
            after = remote_samples(COUNTERS)
            device_rates = rates(before, after, operation)
            app = job[operation]["bw_bytes"] / 1048576
            print(f"{operation}: fio {app:.1f} MiB/s; physical SSD read/write by target:")
            for target in TARGETS:
                amount, rate = device_rates[target]
                cap = 45 if target in FAST[pool] else 15
                print(f"  {target}: {amount:.0f} MiB, {rate:.1f} MiB/s (configured cap {cap})")
            if operation == "read" and sum(value[0] for value in device_rates.values()) < 1024:
                print("  SSD reads were below half the 2-GiB input; caching masks this cap check.")
    except BaseException:
        print(f"Probe failed; inspect temporary file before removal: {directory}")
        raise
    shutil.rmtree(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Actively write/read a temporary BeeGFS file")
    parser.add_argument("--pool", type=int, choices=FAST, help="Required with --native (14–18)")
    parser.add_argument("--seconds", type=int, default=10, help="Passive sample duration (default: 10)")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if args.native:
        if args.pool is None:
            parser.error("--native requires --pool 14–18")
        native_check(args.pool)
        return
    if args.pool is not None:
        verify(args.pool)
    _, measurements = remote_samples(SAMPLE, args.seconds)
    show_sample(args.seconds, measurements)


if __name__ == "__main__":
    main()
