#!/usr/bin/env python3
"""Move four SSD targets to the next DLIO pool and set its runtime I/O caps."""

import argparse
import shlex
import subprocess

from pool_common import DEVICES, FAST, TARGETS, beegfs, verify


SERVICE = "beegfs-storage.service"
TARGET_LIST = ",".join(map(str, TARGETS))


def remote(host, *args):
    subprocess.run(["ssh", host, shlex.join(args)], check=True)


def move(source, destination):
    beegfs("--modifystoragepool", f"--id={source}", f"--rmtargets={TARGET_LIST}")
    beegfs("--modifystoragepool", f"--id={destination}", f"--addtargets={TARGET_LIST}")


def apply_limits(pool):
    for host in ("colva1", "colva2", "colva3"):
        settings = [f"{direction}={device} {(45 if target in FAST[pool] else 15) * 1048576}"
                    for direction in ("IOReadBandwidthMax", "IOWriteBandwidthMax")
                    for target, (node, device) in DEVICES.items() if node == host]
        remote(host, "sudo", "systemctl", "set-property", "--runtime", SERVICE, *settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-pool", type=int, required=True, choices=FAST)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--to-pool", type=int, choices=FAST)
    action.add_argument("--restore", action="store_true", help="Return targets to pool 6 and clear runtime caps")
    args = parser.parse_args()

    verify(args.from_pool)
    if args.restore:
        move(args.from_pool, 6)
        for host in ("colva1", "colva2", "colva3"):
            remote(host, "sudo", "systemctl", "set-property", "--runtime", SERVICE,
                   "IOReadBandwidthMax=", "IOWriteBandwidthMax=")
        print("Targets returned to pool 6; runtime limits cleared. Verify the listing below:")
    else:
        if args.to_pool != args.from_pool + 1:
            parser.error("Conditions must progress one pool at a time (14→15→16→17→18)")
        listing = beegfs("--liststoragepools")
        fast = len(FAST[args.to_pool])
        expected = [str(args.to_pool), f"dlio_ssd_f{fast}_s{4 - fast}"]
        if expected not in [line.split() for line in listing.splitlines()]:
            raise RuntimeError(f"Destination pool {args.to_pool} must exist and be empty")
        move(args.from_pool, args.to_pool)
        apply_limits(args.to_pool)
        verify(args.to_pool)
        print(f"Pool {args.to_pool} ready with targets {TARGET_LIST} and its read/write caps.")
    print(beegfs("--liststoragepools"))


if __name__ == "__main__":
    main()
