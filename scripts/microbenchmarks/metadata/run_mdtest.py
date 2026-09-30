#!/usr/bin/env python3
"""Run the BeeGFS metadata pilot or its 180-case full matrix on anjuna3."""

import argparse
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time


RUNS = Path(__file__).resolve().parents[3] / "results/microbenchmarks/runs"
MOUNT = Path("/mnt/beegfs")
NAMESPACE = MOUNT / "pfs/.metadata-mdtest"
MPIRUN = Path("/mnt/nfs_shared/mpich-install/bin/mpirun")
MDTEST = Path("/home/pfs/ior-main/src/mdtest")
PILOT_ITEMS_PER_RANK = 1000
FULL_ITEMS_PER_RANK = 10000  # Pilot phases were too short at 1,000 for full runs.
PILOT_CASE_TIMEOUT = 300
FULL_CASE_TIMEOUT = 3600  # Revisit with the pilot-derived full item count.
CLIENTS = ("anjuna2", "anjuna3")
HDD_POOL_NAME = "REPLACE_WITH_HDD_SINGLETON_POOL"
SSD_POOL_NAME = "REPLACE_WITH_SSD_SINGLETON_POOL"
TARGET_POOLS = {"hdd": {"name": HDD_POOL_NAME, "target_id": 101},
                "ssd": {"name": SSD_POOL_NAME, "target_id": 104}}
BEEGFS_CTL = Path("/usr/sbin/beegfs-ctl")
CLIENT_CONFIG = Path("/etc/beegfs/beegfs-client.conf")
PILOT = (("anjuna2", 1, "flat"), ("anjuna3", 1, "flat"),
         ("dual", 1, "per_rank"), ("dual", 4, "flat"))


def sha256(path):
    """Hash one binary or raw artifact for run identity and resume checks."""
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    """Atomically replace a runner-owned JSON record in its existing directory."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(path):
    """Read a JSON record previously written by this runner."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_units(pilot):
    """Return the original unpinned matrix for validation of saved results."""
    if pilot:
        return [{"id": f"pilot-{i:02d}", "placement": place, "ranks": ranks,
                 "layout": layout, "repetition": 1}
                for i, (place, ranks, layout) in enumerate(PILOT, 1)]
    cases = list(product((*CLIENTS, "dual"), (1, 4, 16), ("flat", "per_rank")))
    random.Random(20260924).shuffle(cases)
    result = []
    for repetition in range(1, 6):
        shift = (repetition - 1) * 7 % len(cases)
        for place, ranks, layout in cases[shift:] + cases[:shift]:
            result.append({"id": f"r{repetition:02d}-{place}-n{ranks}-{layout}",
                           "placement": place, "ranks": ranks, "layout": layout,
                           "repetition": repetition})
    return result


def units(pilot):
    """Pair target classes and counterbalance which one runs first."""
    expanded = []
    for index, unit in enumerate(legacy_units(pilot)):
        reverse = index % 2 == 1
        target_order = ("ssd", "hdd") if reverse else ("hdd", "ssd")
        expanded.extend({**unit, "target_class": target_class,
                         "id": f"{unit['id']}-{target_class}"}
                        for target_class in target_order)
    return expanded


def beegfs_ctl(argv, timeout=20):
    """Run a local BeeGFS control command with the installed client config."""
    return subprocess.run(["sudo", "-n", str(BEEGFS_CTL),
                           f"--cfgFile={CLIENT_CONFIG}", *argv],
                          capture_output=True, text=True, check=True,
                          timeout=timeout).stdout


def parse_pool_names(output, expected_names):
    """Resolve configured descriptions in both two- and multi-column listings."""
    expected_names = set(expected_names)
    pools = {}
    for line in output.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\S+)(?:\s+.*)?$", line)
        if match and match.group(2) in expected_names:
            pool_id, name = int(match.group(1)), match.group(2)
            if name in pools or pool_id in pools.values():
                raise ValueError("duplicate storage pool ID or description")
            pools[name] = pool_id
    return pools


def parse_target_pools(output):
    """Parse every target row; reject malformed numeric rows rather than miss members."""
    targets = {}
    for line in output.splitlines():
        if not re.match(r"^\s*\d+\b", line):
            continue
        fields = line.split()
        if len(fields) < 3 or any(not field.isdigit() for field in fields[:3]):
            raise ValueError(f"unrecognized target row in pool listing: {line!r}")
        target_id, pool_id = int(fields[0]), int(fields[1])
        if target_id in targets:
            raise ValueError(f"duplicate target {target_id} in pool listing")
        targets[target_id] = pool_id
    if not targets:
        raise ValueError("could not parse target-to-pool listing")
    return targets


def storage_pool_state():
    """Resolve configured pool names and require one expected target per pool."""
    names = {key: value["name"] for key, value in TARGET_POOLS.items()}
    if (set(names) != {"hdd", "ssd"}
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                   or name.startswith("REPLACE_") for name in names.values())
            or len(set(names.values())) != 2):
        raise ValueError("set HDD_POOL_NAME and SSD_POOL_NAME to distinct pool descriptions")
    pool_listing = beegfs_ctl(["--liststoragepools"])
    target_listing = beegfs_ctl(["--listtargets", "--storagepools"])
    name_to_id = parse_pool_names(pool_listing, names.values())
    target_to_pool = parse_target_pools(target_listing)
    result = {}
    for target_class, configured in TARGET_POOLS.items():
        name = configured["name"]
        if name not in name_to_id:
            raise ValueError(f"configured {target_class} pool {name!r} was not found")
        pool_id = name_to_id[name]
        members = sorted(target for target, member_pool in target_to_pool.items()
                         if member_pool == pool_id)
        expected = configured["target_id"]
        if members != [expected]:
            raise ValueError(f"{target_class} pool {name!r} must contain only target "
                             f"{expected}; found {members}")
        result[target_class] = {"name": name, "pool_id": pool_id,
                                "target_id": expected, "members": members}
    if result["hdd"]["pool_id"] == result["ssd"]["pool_id"]:
        raise ValueError("HDD and SSD pool names resolve to the same pool")
    return {"classes": result, "pool_listing": pool_listing,
            "target_listing": target_listing}


def pattern_command(work, pool_id):
    """Set one-stripe RAID0 inheritance on an empty case work directory."""
    return ["sudo", "-n", str(BEEGFS_CTL), f"--cfgFile={CLIENT_CONFIG}",
            "--setpattern", "--pattern=raid0", f"--storagepoolid={pool_id}",
            "--numtargets=1", "--chunksize=512k", str(work)]


def directory_pattern(entryinfo):
    """Return the configured pool and desired target count from entry info."""
    pool = re.search(r"(?m)^\s*\+?\s*Storage Pool:\s*(\d+)", entryinfo)
    targets = re.search(r"(?m)^\s*\+ Number of storage targets: desired:\s*(\d+)",
                        entryinfo)
    if not pool or not targets:
        raise ValueError("BeeGFS entry info lacks storage-pool or stripe-count details")
    return {"pool_id": int(pool.group(1)), "num_targets": int(targets.group(1))}


def validate_placement(record, unit, work, storage_pools):
    """Require raw placement evidence to match the plan and case directory."""
    target_class = unit.get("target_class")
    pool = storage_pools["classes"][target_class]
    if (record.get("target_class") != target_class
            or record.get("pool") != pool
            or record.get("setpattern_command") != pattern_command(work, pool["pool_id"])
            or record.get("verified_pattern") != {"pool_id": pool["pool_id"],
                                                    "num_targets": 1}
            or directory_pattern(record.get("directory_entryinfo", ""))
            != record.get("verified_pattern")):
        raise ValueError(f"{unit['id']}: saved storage-pool placement differs from plan")


def validate_storage_pools(storage_pools):
    """Validate saved singleton-pool membership evidence without cluster access."""
    classes = storage_pools.get("classes", {})
    if set(classes) != {"hdd", "ssd"}:
        raise ValueError("plan lacks HDD/SSD singleton pool configuration")
    if classes["hdd"].get("target_id") != 101 or classes["ssd"].get("target_id") != 104:
        raise ValueError("plan target IDs differ from the documented HDD/SSD pair")
    if (classes["hdd"].get("members") != [101]
            or classes["ssd"].get("members") != [104]
            or classes["hdd"].get("pool_id") == classes["ssd"].get("pool_id")
            or classes["hdd"].get("name") == classes["ssd"].get("name")):
        raise ValueError("plan pools are not distinct single-target pools")
    names = parse_pool_names(storage_pools.get("pool_listing", ""),
                             (pool["name"] for pool in classes.values()))
    target_to_pool = parse_target_pools(storage_pools.get("target_listing", ""))
    for target_class, pool in classes.items():
        if (names.get(pool["name"]) != pool["pool_id"]
                or target_to_pool.get(pool["target_id"]) != pool["pool_id"]
                or sorted(target for target, pool_id in target_to_pool.items()
                          if pool_id == pool["pool_id"]) != pool["members"]):
            raise ValueError("saved BeeGFS pool listings do not verify singleton targets")


def command(unit, work, items):
    """Return one fixed MPICH → mdtest argv; no shell interprets its paths."""
    hosts = CLIENTS if unit["placement"] == "dual" else (unit["placement"],)
    ranks = unit["ranks"]
    argv = [str(MPIRUN), "-wdir", str(work), "-np", str(len(hosts) * ranks), "-hosts",
            ",".join(f"{host}:{ranks}" for host in hosts), "-bind-to", "core",
            str(MDTEST), "-d", str(work), "-n", str(items), "-i", "1",
            "-w", "0", "-e", "0", "-N", "0", "-P"]
    if unit["layout"] == "per_rank":
        argv.append("-u")
    return argv


def no_symlinks(path):
    """Reject links in an owned path before creating or deleting beneath it."""
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink in benchmark path: {part}")


def mount_state():
    """Read the current BeeGFS mount and client configuration identity."""
    no_symlinks(NAMESPACE.parent)
    mount = subprocess.run(["findmnt", "-n", "-o", "FSTYPE,SOURCE,TARGET,OPTIONS",
                            "--target", str(MOUNT)], capture_output=True, text=True,
                           check=True, timeout=10).stdout.strip()
    parts = mount.split(maxsplit=3)
    if len(parts) != 4 or parts[0] != "beegfs" or parts[2] != str(MOUNT):
        raise ValueError(f"expected BeeGFS at {MOUNT}: {mount}")
    configs = list(Path("/proc/fs/beegfs").glob("*/config"))
    if len(configs) != 1:
        raise ValueError("expected one live BeeGFS client config")
    return {"mount": mount, "mount_device": MOUNT.stat().st_dev,
            "client_config_sha256": sha256(configs[0])}


def preflight():
    """Require anjuna3, installed tools and the configured singleton target pools."""
    if socket.gethostname().split(".")[0] != "anjuna3":
        raise ValueError("run this script on anjuna3")
    state = mount_state()
    tools = {}
    for name, path, help_flag, required in (
            ("mpirun", MPIRUN, "-help", ("-wdir", "-hosts", "-bind-to")),
            ("mdtest", MDTEST, "-h", ("-d", "-n", "-i", "-w", "-e", "-N", "-P", "-u"))):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"{name} missing or not executable: {path}")
        help_text = subprocess.run([str(path), help_flag], capture_output=True,
                                   text=True, timeout=20)
        output = help_text.stdout + help_text.stderr
        if any(flag not in output for flag in required):
            raise ValueError(f"{name} help lacks required options")
        tools[name] = {"path": str(path), "sha256": sha256(path), "help": output}
    return {**state, "tools": tools, "storage_pools": storage_pool_state()}


def remote_probe(marker, mdtest_hash):
    """Verify both MPI clients see the marker, NetBench off and the same mdtest."""
    script = ('test "$(findmnt -n -o FSTYPE --target /mnt/beegfs)" = beegfs '
              '&& test -f "$1" && test -x "$2" '
              '&& hash=$(sha256sum "$2") && test "${hash%% *}" = "$3" '
              '&& set -- /proc/fs/beegfs/*/netbench_mode '
              '&& test -r "$1" '
              '&& for f; do value=$(sed -n "1{s/[[:space:]]//g;p;}" "$f") '
              '&& test "$value" = 0 || exit 1; done '
              '&& set -- /proc/fs/beegfs/*/config '
              '&& test "$#" -eq 1 && test -r "$1" '
              '&& hash=$(sha256sum "$1") '
              '&& printf "%s %s\\n" "$(hostname)" "${hash%% *}"')
    argv = [str(MPIRUN), "-wdir", str(marker.parent), "-np", "2",
            "-hosts", "anjuna2:1,anjuna3:1",
            "/bin/sh", "-c", script, "sh", str(marker), str(MDTEST), mdtest_hash]
    probe = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    rows = [line.split() for line in probe.stdout.splitlines()]
    clients = {row[0].split(".")[0]: row[1] for row in rows if len(row) == 2}
    if probe.returncode or len(rows) != 2 or set(clients) != set(CLIENTS):
        raise ValueError(f"MPI/BeeGFS preflight failed (exit {probe.returncode}): "
                         f"stdout={probe.stdout!r} stderr={probe.stderr!r}")
    return {"argv": argv, "stdout": probe.stdout, "stderr": probe.stderr,
            "client_config_sha256": clients}


def owned_run(run_id, plan):
    """Create or verify the local raw directory and marker-owned BeeGFS root."""
    raw = RUNS / run_id
    shared = NAMESPACE / run_id
    no_symlinks(RUNS)
    no_symlinks(NAMESPACE.parent)
    raw.mkdir(mode=0o700, exist_ok=True)
    NAMESPACE.mkdir(mode=0o700, exist_ok=True)
    no_symlinks(raw)
    no_symlinks(NAMESPACE)
    owner = {"run_id": run_id, "plan_sha256": hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest()}
    for root in (raw, shared):
        if root.exists():
            no_symlinks(root)
            marker = root / "owner.json"
            if marker.exists():
                if marker.is_symlink() or load(marker) != owner:
                    raise ValueError(f"run marker differs: {root}")
            elif list(root.iterdir()):
                raise ValueError(f"unowned nonempty run directory: {root}")
            else:
                save(marker, owner)
        else:
            root.mkdir(mode=0o700)
            save(root / "owner.json", owner)
    if (raw / "plan.json").exists():
        if load(raw / "plan.json") != plan:
            raise ValueError("saved plan differs from this script/configuration")
    else:
        save(raw / "plan.json", plan)
    return raw, shared, owner


def cleanup(work_root, shared, unit, owner):
    """Remove only the completed case's exact marker-owned BeeGFS subtree."""
    if work_root != shared / unit["id"] or not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("unsafe cleanup target")
    no_symlinks(work_root)
    if (work_root / "owner.json").is_symlink() or load(work_root / "owner.json") != owner:
        raise ValueError("case marker differs")
    device = work_root.stat().st_dev
    for directory, folders, files in os.walk(work_root, followlinks=False):
        for name in (*folders, *files):
            child = Path(directory) / name
            if child.is_symlink() or child.stat().st_dev != device:
                raise ValueError(f"foreign link or mount in case: {child}")
    shutil.rmtree(work_root)


def launch(argv, raw, timeout):
    """Capture native streams; terminate only this MPI process group on timeout."""
    started = time.time()
    interrupted = False
    with (raw / "stdout.txt").open("xb") as stdout, (raw / "stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            interrupted = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    return {"returncode": process.returncode, "interrupted": interrupted,
            "started_at": started, "ended_at": time.time()}


def measure(unit, raw, shared, owner, items, timeout, baseline):
    """Run one case and checkpoint its unmodified output; stop on any failure."""
    case_raw = raw / "cases" / unit["id"]
    case_shared = shared / unit["id"]
    if case_raw.exists() or case_shared.exists() or case_raw.is_symlink() or case_shared.is_symlink():
        raise ValueError(f"{unit['id']}: incomplete/foreign case exists; inspect before retry")
    probe = remote_probe(shared / "owner.json", baseline["tools"]["mdtest"]["sha256"])
    if probe["client_config_sha256"] != baseline["remote_client_config_sha256"]:
        raise ValueError("BeeGFS client configuration changed during the run")
    before = mount_state()
    if before != {key: baseline[key] for key in before}:
        raise ValueError("BeeGFS mount changed before mdtest")
    no_symlinks(shared)
    case_shared.mkdir(mode=0o700)
    save(case_shared / "owner.json", owner)
    (case_shared / "work").mkdir(mode=0o700)
    case_raw.mkdir(parents=True)
    save(case_raw / "preflight.json", probe)
    live_pools = storage_pool_state()
    if live_pools != baseline["storage_pools"]:
        raise ValueError("BeeGFS storage-pool membership changed during the run")
    pool = live_pools["classes"][unit["target_class"]]
    setpattern_argv = pattern_command(case_shared / "work", pool["pool_id"])
    setpattern = subprocess.run(setpattern_argv, capture_output=True, text=True,
                                timeout=30)
    if setpattern.returncode:
        raise ValueError(f"could not set {unit['target_class']} pool pattern: "
                         f"{setpattern.stderr.strip()}")
    entryinfo = beegfs_ctl(["--getentryinfo", "--verbose", str(case_shared / "work")])
    verified_pattern = directory_pattern(entryinfo)
    if verified_pattern != {"pool_id": pool["pool_id"], "num_targets": 1}:
        raise ValueError(f"{unit['id']}: BeeGFS did not apply the requested singleton pattern")
    save(case_raw / "placement.json", {
        "target_class": unit["target_class"], "pool": pool,
        "setpattern_command": setpattern_argv,
        "setpattern_stdout": setpattern.stdout,
        "directory_entryinfo": entryinfo,
        "verified_pattern": verified_pattern,
    })
    argv = command(unit, case_shared / "work", items)
    save(case_raw / "command.json", argv)
    save(case_raw / "before.json", {"at": time.time(), **before})
    print(f"{unit['id']}: {unit['target_class']} target {pool['target_id']} "
          f"via pool {pool['name']} ({pool['pool_id']}); {' '.join(argv)}", flush=True)
    result = launch(argv, case_raw, timeout)
    after = mount_state()
    save(case_raw / "after.json", {"at": time.time(), **after})
    pool_state_ok = False
    try:
        after_pools = storage_pool_state()
        save(case_raw / "after_storage_pools.json",
             {"at": time.time(), **after_pools})
        pool_state_ok = after_pools == baseline["storage_pools"]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        save(case_raw / "after_storage_pools.json",
             {"at": time.time(), "error": str(error)})
    result["storage_pools_unchanged"] = pool_state_ok
    result["stdout_sha256"] = sha256(case_raw / "stdout.txt")
    result["stderr_sha256"] = sha256(case_raw / "stderr.txt")
    result["cleanup"] = "pending"
    save(case_raw / "result.json", result)
    if result["returncode"] != 0 or result["interrupted"]:
        raise ValueError(f"{unit['id']}: MPI failed/interrupted; shared namespace preserved")
    if not pool_state_ok:
        raise ValueError(f"{unit['id']}: storage-pool state changed or could not be verified; "
                         "shared namespace preserved")
    if after != before:
        raise ValueError("BeeGFS mount changed before cleanup")
    cleanup(case_shared, shared, unit, owner)
    result["cleanup"] = "completed"
    save(case_raw / "result.json", result)


def benchmark(run_id, pilot):
    """Run pending cases serially, skipping only intact completed raw cases."""
    mode = "pilot" if pilot else "full"
    if not re.fullmatch(rf"metadata-{mode}-[A-Za-z0-9_-]+", run_id):
        raise ValueError(f"run ID must start with metadata-{mode}-")
    items = PILOT_ITEMS_PER_RANK if pilot else FULL_ITEMS_PER_RANK
    if type(items) is not int or items < 1:
        raise ValueError("FULL_ITEMS_PER_RANK must be a positive integer before --full")
    if not RUNS.is_dir() or RUNS.is_symlink():
        raise ValueError("project results/microbenchmarks/runs directory missing")
    no_symlinks(RUNS)
    if (RUNS / ".metadata.lock").is_symlink():
        raise ValueError("metadata lock is a symlink")
    with (RUNS / ".metadata.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = preflight()
        plan = {"mode": mode, "items_per_rank": items, "units": units(pilot),
                "storage_pools": baseline["storage_pools"],
                "mpirun": {key: baseline["tools"]["mpirun"][key] for key in ("path", "sha256")},
                "mdtest": {key: baseline["tools"]["mdtest"][key] for key in ("path", "sha256")},
                "mount": baseline["mount"], "client_config_sha256": baseline["client_config_sha256"]}
        raw, shared, owner = owned_run(run_id, plan)
        probe = remote_probe(shared / "owner.json", baseline["tools"]["mdtest"]["sha256"])
        remote_file = raw / "remote_clients.json"
        if remote_file.exists():
            if load(remote_file)["client_config_sha256"] != probe["client_config_sha256"]:
                raise ValueError("remote BeeGFS client configuration changed")
        else:
            save(remote_file, probe)
        baseline["remote_client_config_sha256"] = probe["client_config_sha256"]
        no_symlinks(raw / "cases")
        (raw / "cases").mkdir(exist_ok=True)
        for unit in plan["units"]:
            case_raw = raw / "cases" / unit["id"]
            case_shared = shared / unit["id"]
            result_file = case_raw / "result.json"
            if result_file.exists():
                no_symlinks(case_raw)
                for name in ("stdout.txt", "stderr.txt", "command.json", "result.json"):
                    if (case_raw / name).is_symlink():
                        raise ValueError(f"{unit['id']}: symlinked raw evidence")
                placement_file = case_raw / "placement.json"
                if placement_file.is_symlink():
                    raise ValueError(f"{unit['id']}: symlinked placement evidence")
                validate_placement(load(placement_file), unit,
                                   case_shared / "work", plan["storage_pools"])
                result = load(result_file)
                after_pools_file = case_raw / "after_storage_pools.json"
                if (after_pools_file.is_symlink() or not after_pools_file.is_file()
                        or result.get("storage_pools_unchanged") is not True):
                    raise ValueError(f"{unit['id']}: missing or failed post-case pool check")
                after_pools = load(after_pools_file)
                after_pools.pop("at", None)
                if (result.get("returncode") != 0 or result.get("interrupted")
                        or result.get("cleanup") != "completed"
                        or after_pools != plan["storage_pools"]
                        or case_shared.exists() or case_shared.is_symlink()
                        or load(case_raw / "command.json") != command(unit, case_shared / "work", items)
                        or sha256(case_raw / "stdout.txt") != result.get("stdout_sha256")
                        or sha256(case_raw / "stderr.txt") != result.get("stderr_sha256")):
                    raise ValueError(f"{unit['id']}: saved raw case is incomplete or changed")
                continue
            measure(unit, raw, shared, owner, items,
                    PILOT_CASE_TIMEOUT if pilot else FULL_CASE_TIMEOUT, baseline)
        print(f"{mode}: {len(plan['units'])}/{len(plan['units'])} mdtest exits and cleanups; "
              f"native phase summaries are preserved, not parsed; raw results: {raw}")


def main(argv=None):
    """Expose only run identity and pilot/full choice; cluster control is fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    def on_term(_signal, _frame):
        raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGTERM, on_term)
    try:
        benchmark(args.run_id, args.pilot)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"Metadata stopped: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    sys.exit(main())
