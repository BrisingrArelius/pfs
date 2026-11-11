#!/usr/bin/env python3
"""Move four SSD targets to the next DLIO pool and set its runtime I/O caps."""

import argparse
import re
import shlex

from pool_common import DEVICES, FAST, TARGETS, beegfs, execute, verify, verify_membership


SERVICE = "beegfs-storage.service"
TARGET_LIST = ",".join(map(str, TARGETS))
NORMAL_TARGETS = {104, 105, 106, 107, 205, 206, 305, 306, 307}


def remote(host, *args):
    return execute(["ssh", host, shlex.join(args)], check=True,
                          capture_output=True, text=True).stdout


def normal_members(listing, pool=6):
    members = set()
    in_pool = False
    for line in listing.splitlines():
        row = re.match(r"^[ \t]*(\d+)[ \t]+([A-Za-z_]\S*)(?:[ \t]+([\d,]+))?", line)
        if row:
            in_pool = row[1] == str(pool)
            if in_pool and row[3]:
                members.update(int(value) for value in row[3].split(",") if value)
        elif in_pool and re.fullmatch(r"[ \t]*[\d,]+[ \t]*", line):
            members.update(int(value) for value in line.strip().split(",") if value)
    return members


def verify_normal():
    listing = beegfs("--liststoragepools")
    if normal_members(listing) != NORMAL_TARGETS:
        raise RuntimeError(f"Pool 6 is not the original SSD pool: {normal_members(listing)}")
    for host in ("colva1", "colva2", "colva3"):
        text = remote(host, "systemctl", "show", SERVICE,
                      "-p", "IOReadBandwidthMax", "-p", "IOWriteBandwidthMax")
        if any(line.split("=", 1)[1].strip() for line in text.splitlines()
               if line.startswith(("IOReadBandwidthMax=", "IOWriteBandwidthMax="))):
            raise RuntimeError(f"{host}: existing storage-service bandwidth limits; restore them first")


def move(source, destination):
    beegfs("--modifystoragepool", f"--id={source}", f"--rmtargets={TARGET_LIST}")
    beegfs("--modifystoragepool", f"--id={destination}", f"--addtargets={TARGET_LIST}")


def apply_limits(pool):
    for host in ("colva1", "colva2", "colva3"):
        # Replacing a per-device property otherwise leaves stale entries on
        # the old path (colva1's partition was configured but not enforced).
        remote(host, "sudo", "systemctl", "set-property", "--runtime", SERVICE,
               "IOReadBandwidthMax=", "IOWriteBandwidthMax=")
        settings = [f"{direction}={device} {(45 if target in FAST[pool] else 15) * 1048576}"
                    for direction in ("IOReadBandwidthMax", "IOWriteBandwidthMax")
                    for target, (node, device) in DEVICES.items() if node == host]
        remote(host, "sudo", "systemctl", "set-property", "--runtime", SERVICE, *settings)


def transition(source, destination):
    if not (source == 6 and destination in FAST) and (source, destination) not in ((14, 16), (16, 18)) and (
            source not in FAST or destination not in FAST or destination != source + 1):
        raise ValueError("Expected pool 6→an experiment pool, the next pool, or CosmoFlow's 14→16→18")
    verify_normal() if source == 6 else verify(source)
    listing = beegfs("--liststoragepools")
    fast = len(FAST[destination])
    expected = [str(destination), f"dlio_ssd_f{fast}_s{4 - fast}"]
    if expected not in [line.split() for line in listing.splitlines()]:
        raise RuntimeError(f"Destination pool {destination} must exist and be empty")
    move(source, destination)
    apply_limits(destination)
    verify(destination)
    print(f"Pool {destination} ready with targets {TARGET_LIST} and its read/write caps.", flush=True)


def restore(source):
    if source not in FAST:
        raise ValueError("Expected an experiment pool 14–18")
    # Recovery must work after cap verification failed during a transition.
    # Confirm ownership, then remove caps rather than requiring them to work.
    verify_membership(source)
    move(source, 6)
    clear_limits()
    verify_normal()
    print("Pool 6 restored with its original targets and no runtime caps.", flush=True)


def clear_limits():
    for host in ("colva1", "colva2", "colva3"):
        remote(host, "sudo", "systemctl", "set-property", "--runtime", SERVICE,
               "IOReadBandwidthMax=", "IOWriteBandwidthMax=")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-pool", type=int, required=True, choices=(6, *FAST))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--to-pool", type=int, choices=FAST)
    action.add_argument("--restore", action="store_true", help="Return targets to pool 6 and clear runtime caps")
    args = parser.parse_args()

    if args.restore:
        if args.from_pool == 6:
            parser.error("--restore requires an experiment pool 14–18")
        restore(args.from_pool)
    else:
        transition(args.from_pool, args.to_pool)
    print(beegfs("--liststoragepools"))


if __name__ == "__main__":
    main()
