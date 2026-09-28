#!/usr/bin/env python3
"""Check whether anjuna2 can mount a private native-cache BeeGFS client.

This is a feasibility probe, not a throughput measurement. The extra mount lives
in a private mount namespace; it vanishes even if the probe is interrupted.
The existing /mnt/beegfs mount and /etc/beegfs configuration are not changed.
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from run_cache import RUNS

CLIENT_CONFIG = Path("/etc/beegfs/beegfs-client.conf")


def native_config(original):
    """Keep the site's settings, changing only its explicit cache-mode line."""
    pattern = re.compile(r"^(\s*tuneFileCacheType\s*=\s*)buffered(\s*(?:#.*)?)$", re.MULTILINE)
    if len(pattern.findall(original)) != 1:
        raise ValueError("expected one explicit buffered setting in the client config")
    return pattern.sub(r"\g<1>native\2", original)


def native_instances():
    """Identify effective native mounts from the kernel, not config-file text."""
    instances = set()
    for path in Path("/proc/fs/beegfs").glob("*/config"):
        if re.search(r"^tuneFileCacheType\s*=\s*native\s*$", path.read_text(), re.MULTILINE):
            instances.add(str(path.parent))
    return instances


def run_command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): "
                           f"{(result.stderr or result.stdout).strip()}")
    return result.stdout


def inside(run_dir):
    if os.geteuid() != 0:
        raise ValueError("the private mount needs root inside unshare")
    if os.readlink("/proc/self/ns/mnt") == os.readlink("/proc/1/ns/mnt"):
        raise ValueError("refusing to mount in the host mount namespace")

    run_dir = Path(run_dir)
    if (run_dir.parent != RUNS or run_dir.is_symlink()
            or json.loads((run_dir / "owner.json").read_text()) !=
            {"run_id": run_dir.name, "domain": "cache-native-probe"}):
        raise ValueError("probe run directory is not owned by this script")
    config = run_dir / "client-native.conf"
    mountpoint = run_dir / "native-mount"
    before = native_instances()
    original_mount = run_command("findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS",
                                 "--target", "/mnt/beegfs")
    if "beegfs" not in original_mount:
        raise ValueError("original /mnt/beegfs mount is missing")
    result = {"native_mount_supported": False, "original_mount_untouched": False}
    mounted = False
    try:
        text = native_config(CLIENT_CONFIG.read_text())
        descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        mountpoint.mkdir(mode=0o700)
        run_command("mount", "-t", "beegfs", "beegfs_nodev", "-o",
                    f"cfgFile={config}", str(mountpoint))
        mounted = True
        actual_mount = run_command("findmnt", "-n", "-o", "FSTYPE,TARGET", "--target", str(mountpoint))
        if actual_mount.split() != ["beegfs", str(mountpoint)]:
            raise ValueError(f"unexpected mount: {actual_mount.strip()}")
        new_native = native_instances() - before
        if not new_native:
            raise ValueError("no new native BeeGFS instance reported by /proc/fs/beegfs")
        result["native_mount_supported"] = True
        result["native_instances"] = sorted(new_native)
    except (OSError, ValueError, RuntimeError) as error:
        result["error"] = str(error)
    finally:
        if mounted:
            try:
                run_command("umount", str(mountpoint))
            except RuntimeError as error:
                result["unmount_error"] = str(error)
        config.unlink(missing_ok=True)
        if mountpoint.exists():
            try:
                mountpoint.rmdir()
            except OSError:
                pass
        try:
            result["original_mount_untouched"] = (run_command(
                "findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS",
                "--target", "/mnt/beegfs") == original_mount)
        except RuntimeError:
            pass
    # The namespace disappears on exit regardless of unmount outcome.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name below project runs/")
    parser.add_argument("--inside", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        result = inside(args.inside)
        print(json.dumps(result))
        return 0 if (result["native_mount_supported"] and result["original_mount_untouched"]
                     and "unmount_error" not in result) else 1

    if not re.fullmatch(r"cache-native-probe-[A-Za-z0-9_-]+", args.run_id):
        parser.error("run ID must begin cache-native-probe- and use letters, digits, _ or -")
    if socket.gethostname().split(".")[0] != "anjuna2":
        parser.error("run this probe from the project checkout on anjuna2")
    if not RUNS.is_dir() or RUNS.is_symlink():
        parser.error("project results/microbenchmarks/runs must exist and not be a symlink")
    run_dir = RUNS / args.run_id
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError:
        parser.error("probe run directory already exists; choose a new run ID")
    (run_dir / "owner.json").write_text(json.dumps({"run_id": args.run_id,
                                                   "domain": "cache-native-probe"}) + "\n")
    process = subprocess.run(["sudo", "-n", "unshare", "--mount", "--propagation", "private",
                              sys.executable, "-B", str(Path(__file__).resolve()),
                              "--run-id", args.run_id, "--inside", str(run_dir)],
                             text=True, capture_output=True, check=False)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        result = {"native_mount_supported": False,
                  "error": f"private mount probe failed ({process.returncode}): "
                           f"{(process.stderr or process.stdout).strip()}"}
    (run_dir / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if process.returncode == 0 and result["native_mount_supported"] else 1


if __name__ == "__main__":
    sys.exit(main())
