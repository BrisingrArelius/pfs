#!/usr/bin/env python3
"""Print one quiet read/write throughput sample for the four selected SSDs."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from pool_common import DEVICES, TARGETS


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
    active = next((entry for entry in kernel_caps if 'rbps' in entry or 'wbps' in entry), {})
    results[device] = {
        'read_MiB_s': (after[device][2] - before[device][2]) * 512 / 1048576 / elapsed,
        'write_MiB_s': (after[device][6] - before[device][6]) * 512 / 1048576 / elapsed,
        'configured_read': limits['IOReadBandwidthMax'].get(partition),
        'configured_write': limits['IOWriteBandwidthMax'].get(partition),
        'kernel_read': active.get('rbps'), 'kernel_write': active.get('wbps')}
print(json.dumps(results))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=10, help="Sample duration (default: 10 seconds)")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")

    jobs = {}
    for host in ("colva1", "colva2", "colva3"):
        devices = [Path(device).name.rsplit("p", 1)[0]
                   for node, device in DEVICES.values() if node == host]
        command = shlex.join(("python3", "-c", SAMPLE, ",".join(devices), str(args.seconds)))
        jobs[host] = subprocess.Popen(["ssh", host, command], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
    measurements = {}
    for host, process in jobs.items():
        stdout, stderr = process.communicate(timeout=args.seconds + 30)
        if process.returncode:
            raise RuntimeError(f"{host} failed to sample its SSDs: {stderr.strip()}")
        measurements[host] = json.loads(stdout)

    print(f"Selected SSDs · {args.seconds}-second physical-device sample (MiB/s)")
    print("Target  Host    SSD       Read / cap       Write / cap      Kernel limit")
    for target in TARGETS:
        host, path = DEVICES[target]
        device = Path(path).name.rsplit("p", 1)[0]
        item = measurements[host][device]
        def cap(key):
            return f"{item[key] / 1048576:.0f}" if item[key] is not None else "—"
        if item["configured_read"] is None and item["configured_write"] is None \
                and item["kernel_read"] is None and item["kernel_write"] is None:
            status = "none"
        elif (item["configured_read"] is not None and item["configured_write"] is not None
              and item["kernel_read"] == str(item["configured_read"])
              and item["kernel_write"] == str(item["configured_write"])):
            status = "matched"
        else:
            status = "MISMATCH"
        print(f"{target:<7} {host:<7} {device:<9} {item['read_MiB_s']:>5.1f} / {cap('configured_read'):<5} "
              f"{item['write_MiB_s']:>5.1f} / {cap('configured_write'):<5} "
              f"{status}")
    print("Device-wide counters; zero I/O does not test a cap. Sample during DLIO reads/writes.")


if __name__ == "__main__":
    main()
