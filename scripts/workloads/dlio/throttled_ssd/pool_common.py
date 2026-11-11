"""Shared mechanics for the two separate DLIO SSD-speed-mix runners."""

import csv
import importlib.metadata
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NORMAL = Path(__file__).resolve().parents[1] / "normal"
sys.path.insert(0, str(NORMAL))
sys.path.insert(0, str(NORMAL.parent))
from dlio_pool_profiles import DATA_ROOT, RANKS, RESULT_ROOT, WORKLOADS  # noqa: E402
from normal.pool_common import FIELDS, checkpoint_sizes, metrics  # noqa: E402
import dlio_darshan  # noqa: E402
from experiment_config import FAST_MIB_S, SLOW_MIB_S, FAST_BYTES_S, SLOW_BYTES_S, RESULTS_SUBDIR  # noqa: E402

TARGETS = (104, 205, 305, 306)
FAST = {14: TARGETS, 15: (104, 205, 305), 16: (104, 205), 17: (104,), 18: ()}
EXPERIMENT_POOLS = (14, 16, 18)
DEVICES = {104: ("colva1", "/dev/nvme1n1"),
           205: ("colva2", "/dev/nvme1n1"),
           305: ("colva3", "/dev/nvme1n1"),
           306: ("colva3", "/dev/nvme2n1")}
CLASSES = {"resnet": "normal.run_resnet50_pools.ResNetCheckpointing",
           "unet": "normal.run_unet3d_pools.SeededPyTorchCheckpointing",
           "cosmoflow": "normal.run_cosmoflow_pools.CosmoFlowCheckpointing"}

# Enabled only by the reservation-aware ResNet runner. Other runners retain
# their existing subprocess behavior.
DEADLINE = None
MANAGED = False


class DeadlineReached(RuntimeError):
    pass


def remaining():
    if DEADLINE is None:
        return None
    seconds = DEADLINE - time.monotonic()
    if seconds <= 0:
        raise DeadlineReached("Reservation run budget exhausted")
    return seconds


def stop_group(process):
    """Also stop orphaned local ranks if their MPI launcher has exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command, **kwargs):
    if DEADLINE is None and not MANAGED:
        return subprocess.run(command, **kwargs)
    seconds = remaining()
    check = kwargs.pop("check", False)
    data = kwargs.pop("input", None)
    if data is not None:
        kwargs["stdin"] = subprocess.PIPE
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input=data, timeout=seconds)
        except BaseException as error:
            # mpirun and its local rank processes must stop before pool moves.
            stop_group(process)
            if isinstance(error, subprocess.TimeoutExpired):
                raise DeadlineReached("Reservation run budget exhausted") from error
            raise
        if check and process.returncode:
            stop_group(process)
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def beegfs(*args):
    return execute(["sudo", "beegfs-ctl", *map(str, args)], check=True,
                          text=True, capture_output=True).stdout


def verify_membership(pool):
    listing = beegfs("--liststoragepools")
    fast = len(FAST[pool])
    match = re.search(rf"^[ \t]*{pool}[ \t]+dlio_ssd_f{fast}_s{4-fast}[ \t]+([\d,]+)", listing, re.M)
    members = set(map(int, match[1].split(","))) if match else set()
    if members != set(TARGETS):
        raise RuntimeError(f"Pool {pool} must contain {TARGETS}; found {members}")


def verify(pool):
    verify_membership(pool)
    for host in ("colva1", "colva2", "colva3"):
        text = execute(["ssh", host, shlex.join(("systemctl", "show",
            "beegfs-storage.service", "-p", "IOReadBandwidthMax", "-p", "IOWriteBandwidthMax"))],
            check=True, capture_output=True, text=True).stdout
        selected = [(target, device) for target, (node, device) in DEVICES.items() if node == host]
        for direction in ("IOReadBandwidthMax", "IOWriteBandwidthMax"):
            seen = {}
            for line in text.splitlines():
                if line.startswith(direction + "="):
                    device, rate = line.split("=", 1)[1].split()
                    seen[device] = int(rate)
            expected = {device: FAST_BYTES_S if target in FAST[pool] else SLOW_BYTES_S
                        for target, device in selected}
            if seen != expected:
                raise RuntimeError(f"{host}: {direction}={seen}, expected {expected}")
        physical = [Path(device).name.rsplit("p", 1)[0] for _, device in selected]
        ids = execute(["ssh", host, shlex.join(("cat", *(
            f"/sys/class/block/{device}/dev" for device in physical)))],
            check=True, capture_output=True, text=True).stdout.splitlines()
        if len(ids) != len(physical):
            raise RuntimeError(f"{host}: missing physical SSD device IDs")
        command = ('cg=$(systemctl show -P ControlGroup beegfs-storage.service); '
                   'sudo -n cat "/sys/fs/cgroup${cg}/io.max"')
        kernel = execute(["ssh", host, command], check=True,
                                capture_output=True, text=True).stdout
        limits = {line.split()[0]: dict(field.split("=", 1) for field in line.split()[1:])
                  for line in kernel.splitlines()}
        for (target, _), device_id in zip(selected, ids):
            expected_rate = str(FAST_BYTES_S if target in FAST[pool] else SLOW_BYTES_S)
            actual = limits.get(device_id, {})
            if actual.get("rbps") != expected_rate or actual.get("wbps") != expected_rate:
                raise RuntimeError(f"{host} target {target}: io.max {device_id} is {actual}, "
                                   f"expected read/write {expected_rate} B/s")


def pattern(path, pool):
    if not path.exists():
        path.mkdir()
        beegfs("--setpattern", "--pattern=raid0", f"--storagepoolid={pool}",
               "--numtargets=4", "--chunksize=512k", path)
    info = beegfs("--getentryinfo", path)
    if (not re.search(rf"Storage Pool:\s*{pool}\b", info, re.I)
            or "Chunksize: 512K" not in info
            or not re.search(r"Number of storage targets: desired: 4\b", info)):
        raise RuntimeError(f"Unexpected BeeGFS pattern: {path}\n{info}")


def placement(files, output):
    with output.open("w") as record:
        for file in files:
            info = beegfs("--getentryinfo", "--verbose", file)
            targets = set(map(int, re.findall(r"\+ (\d+) @ ", info)))
            if targets != set(TARGETS):
                raise RuntimeError(f"Unexpected actual stripe targets: {file}: {targets}")
            record.write(f"{file}\t{','.join(map(str, sorted(targets)))}\n")
    return ",".join(map(str, TARGETS))


def physical_counters():
    """Snapshot each selected SSD's physical read/write sectors on its OSS."""
    result = {}
    for host in ("colva1", "colva2", "colva3"):
        targets = [(target, Path(device).name.rsplit("p", 1)[0])
                   for target, (node, device) in DEVICES.items() if node == host]
        paths = [f"/sys/class/block/{device}/stat" for _, device in targets]
        lines = execute(["ssh", host, shlex.join(("cat", *paths))],
                               check=True, capture_output=True, text=True).stdout.splitlines()
        if len(lines) != len(targets):
            raise RuntimeError(f"{host}: incomplete physical SSD counters")
        observed_at = time.monotonic()
        for (target, device), line in zip(targets, lines):
            fields = list(map(int, line.split()))
            result[target] = (observed_at, fields[2] * 512, fields[6] * 512, device)
    return result


def physical_io(pool, repetition, before, after):
    rows = []
    for target in TARGETS:
        start, read0, write0, device = before[target]
        end, read1, write1, _ = after[target]
        if end <= start or read1 < read0 or write1 < write0:
            raise RuntimeError(f"Pool {pool}, repetition {repetition}: SSD counter reset on target {target}")
        seconds = end - start
        rows.append((pool, repetition, target, DEVICES[target][0], device, seconds,
                     read1 - read0, write1 - write0,
                     (read1 - read0) / 1048576 / seconds,
                     (write1 - write0) / 1048576 / seconds))
    return rows


def prepare(profile, pool):
    source = DATA_ROOT / profile.folder / "pool1/dataset/train"
    inputs = sorted(source.glob(f"*.{profile.extension}"))
    if len(inputs) != profile.files:
        raise RuntimeError(f"Expected {profile.files} normal input files, got {len(inputs)}")
    root = DATA_ROOT / "throttled_ssd" / profile.folder / f"pool{pool}"
    root.mkdir(parents=True, exist_ok=True)
    dataset, checkpoints = root / "dataset", root / "checkpoints"
    pattern(dataset, pool)
    pattern(checkpoints, pool)
    train = dataset / "train"
    pattern(train, pool)
    (dataset / "valid").mkdir(exist_ok=True)
    present = list(train.iterdir())
    if DEADLINE is not None or MANAGED:
        expected_names = {p.name for p in inputs}
        if any(p.name not in expected_names and p.name != ".copy-partial" for p in present):
            raise RuntimeError(f"Unexpected dataset contents: {train}")
        print(f"{profile.profile}: staging/resuming {len(inputs)} inputs into pool {pool}", flush=True)
        for file in inputs:
            remaining()
            target = train / file.name
            if target.exists():
                if target.stat().st_size != file.stat().st_size:
                    raise RuntimeError(f"Changed input: {target}")
                continue
            temporary = train / ".copy-partial"
            execute(["cp", "--", str(file), str(temporary)], check=True)
            temporary.replace(target)
        (train / ".copy-partial").unlink(missing_ok=True)
        present = list(train.iterdir())
        print(f"{profile.profile}: pool {pool} input staging complete", flush=True)
    if present and ({p.name for p in present} != {p.name for p in inputs}
                    or any((train / file.name).stat().st_size != file.stat().st_size for file in inputs)):
        raise RuntimeError(f"Partial or changed dataset: {train}")
    if not present:
        print(f"{profile.profile}: copying {len(inputs)} inputs into pool {pool}", flush=True)
        for file in inputs:
            shutil.copyfile(file, train / file.name)
        print(f"{profile.profile}: pool {pool} input copy complete", flush=True)
    if any((train / file.name).stat().st_size != file.stat().st_size for file in inputs):
        raise RuntimeError(f"Incomplete dataset copy: {train}")
    observed = inputs
    if len(inputs) > 1024:
        stride = max(1, len(inputs) // 64)
        observed = list(dict.fromkeys(inputs[::stride][:64] + inputs[-1:]))
    placement((train / file.name for file in observed), root / "dataset-osts.tsv")
    return root


def run(name, pool, pilot=False, *, output=None, completed=(), on_complete=None, darshan=None, repetitions=3):
    if repetitions<1:
        raise ValueError("repetitions must be positive")
    repetitions=1 if pilot else repetitions
    if pool not in FAST or name not in CLASSES:
        raise ValueError("Expected ResNet, U-Net or CosmoFlow and a condition pool 14–18")
    if importlib.metadata.version("dlio-benchmark") != "2.0.0":
        raise RuntimeError("DLIO 2.0.0 required")
    if darshan is not None:
        darshan = dlio_darshan.validate(darshan)
    verify(pool)
    profile = WORKLOADS[name]
    root = prepare(profile, pool)
    output = output or RESULT_ROOT / RESULTS_SUBDIR / profile.folder / (
        ("pilot-" if pilot else "runs-") + f"pool{pool}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=True)
    condition = {"pool": pool, "fast": FAST[pool],
        "slow": sorted(set(TARGETS) - set(FAST[pool])), "targets": TARGETS,
        "fast_MiB_s": FAST_MIB_S, "slow_MiB_s": SLOW_MIB_S, "profile": profile.profile}
    condition_path = output / "condition.json"
    if condition_path.exists() and json.loads(condition_path.read_text()) != json.loads(json.dumps(condition)):
        raise RuntimeError(f"Existing condition differs from configured rates or workload: {condition_path}")
    condition_path.write_text(json.dumps(condition, indent=2) + "\n")
    env = os.environ.copy()
    if darshan is not None:
        env = dlio_darshan.clean_environment(env)
    env["PYTHONPATH"] = os.pathsep.join((str(NORMAL.parent), str(NORMAL),
                                         env.get("PYTHONPATH", "")))
    retained = {}
    for filename in ("metrics.csv", "device_io.csv"):
        path = output / filename
        if path.exists():
            with path.open(newline="") as file:
                retained[filename] = [row for row in csv.DictReader(file)
                                      if int(row["repetition"]) in completed]
        # Per-repetition evidence makes rebuilding the combined CSV safe even
        # if the previous process was killed while replacing that CSV.
        for repetition in completed:
            journal = output / f"rep{repetition}-pool{pool}" / filename
            if journal.exists():
                retained[filename] = [row for row in retained.get(filename, [])
                                      if int(row["repetition"]) != repetition]
                with journal.open(newline="") as file:
                    retained[filename].extend(csv.DictReader(file))
        for repetition in completed:
            expected_rows = 8 if filename == "metrics.csv" else 4
            if sum(int(row["repetition"]) == repetition for row in retained.get(filename, [])) != expected_rows:
                raise RuntimeError(f"Completed repetition {repetition} has missing/duplicate evidence: {path}")
    with (output / "metrics.csv").open("w", newline="") as csvfile, \
            (output / "device_io.csv").open("w", newline="") as devicefile:
        writer = csv.writer(csvfile)
        writer.writerow(FIELDS)
        devices = csv.writer(devicefile)
        devices.writerow(("pool", "repetition", "target", "host", "device", "duration_s",
                           "read_bytes", "write_bytes", "read_MiB_s", "write_MiB_s"))
        writer.writerows([row[field] for field in FIELDS] for row in retained.get("metrics.csv", []))
        device_fields = ("pool", "repetition", "target", "host", "device", "duration_s",
                         "read_bytes", "write_bytes", "read_MiB_s", "write_MiB_s")
        devices.writerows([row[field] for field in device_fields]
                          for row in retained.get("device_io.csv", []))
        csvfile.flush()
        devicefile.flush()
        for repetition in range(1, repetitions+1):
            if repetition in completed:
                print(f"{profile.profile}: pool {pool}, run {repetition}/{repetitions} already complete; skipping", flush=True)
                continue
            remaining()
            directory = output / f"rep{repetition}-pool{pool}"
            checkpoint_dir = root / "checkpoints" / f"{output.name}-rep{repetition}"
            if directory.exists() or checkpoint_dir.exists():
                archived = output / "interrupted" / f"rep{repetition}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
                archived.mkdir(parents=True)
                if directory.exists():
                    directory.rename(archived / "raw")
                if checkpoint_dir.exists():
                    checkpoint_archive = checkpoint_dir.with_name(checkpoint_dir.name + "-interrupted-" + archived.name)
                    checkpoint_dir.rename(checkpoint_archive)
                    (archived / "checkpoint-path.txt").write_text(str(checkpoint_archive) + "\n")
            directory.mkdir()
            checkpoint_dir.mkdir()
            print(f"{profile.profile}: pool {pool}, run {repetition}/"
                  f"{repetitions} starting", flush=True)
            size = min(profile.model_bytes, 16 * 1048576) if pilot else profile.model_bytes
            command = ["mpirun", "-np", str(RANKS), "dlio_benchmark", f"workload={profile.profile}",
                       f"++workload.dataset.num_files_train={profile.files}",
                       "++workload.workflow.generate_data=False", "++workload.workflow.train=True",
                       "++workload.workflow.checkpoint=True",
                       f"++workload.dataset.data_folder={root / 'dataset'}",
                       f"++workload.checkpoint.checkpoint_folder={checkpoint_dir}",
                       "++workload.checkpoint.type=all_ranks",
                       f"++workload.checkpoint.checkpoint_mechanism_classname={CLASSES[name]}",
                       "++workload.checkpoint.checkpoint_after_epoch=2",
                       "++workload.checkpoint.epochs_between_checkpoints=2",
                       f"++workload.checkpoint.model_size={size}",
                       "++workload.checkpoint.num_layers=0", "++workload.checkpoint.layer_parameters=[]",
                       f"++workload.reader.batch_size={profile.batch_size}",
                       f"++workload.reader.read_threads={profile.reader_threads}",
                       "++workload.reader.file_shuffle=seed", "++workload.reader.sample_shuffle=seed",
                       "++workload.reader.prefetch_size=2",
                       f"++workload.train.epochs={2 if pilot else 5}",
                       "++workload.train.computation_time=0", f"hydra.run.dir={directory}"]
            if pilot:
                command.insert(-1, "++workload.train.total_training_steps=2")
            if darshan is not None:
                command = dlio_darshan.instrument(command, directory, root/"dataset", checkpoint_dir, darshan)
            execute(["sync"], check=True)
            for host in ("colva1", "colva2", "colva3", "colva4"):
                execute(["ssh", host,
                                "sudo -n sh -c 'sync && printf 3 > /proc/sys/vm/drop_caches'"],
                               check=True)
            execute(["sudo", "tee", "/proc/sys/vm/drop_caches"], input="3\n",
                           text=True, stdout=subprocess.DEVNULL, check=True)
            if remaining() is not None and remaining() < 30:
                raise DeadlineReached("Insufficient time for cache preparation")
            time.sleep(30)
            before = physical_counters()
            with (directory / "console.log").open("w") as log:
                execute(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            io_rows = physical_io(pool, repetition, before, physical_counters())
            devices.writerows(io_rows)
            devicefile.flush()
            if darshan is not None:
                dlio_darshan.check_logs(directory)
            files = sorted(file for file in checkpoint_dir.rglob("*") if file.is_file())
            sizes = checkpoint_sizes(files, profile, (2,) if pilot else (2, 4), size)
            checkpoint_osts = placement(files, directory / "checkpoint-osts.tsv")
            if pilot:
                epochs = json.loads((directory / "per_epoch_stats.json").read_text())
                if len(epochs) != 2 or "ckpt1" not in epochs["2"]:
                    raise RuntimeError("Pilot checkpoint missing")
                print(f"Pilot checkpoint retained: {checkpoint_dir}")
            else:
                metric_rows = list(metrics(directory, pool, repetition,
                                           ",".join(map(str, TARGETS)), checkpoint_osts, sizes))
                writer.writerows(metric_rows)
                csvfile.flush()
                if on_complete is not None:
                    for filename, fields, rows in (("metrics.csv", FIELDS, metric_rows),
                                                    ("device_io.csv", device_fields, io_rows)):
                        temporary = directory / (filename + ".tmp")
                        with temporary.open("w", newline="") as journal:
                            recorded = csv.writer(journal)
                            recorded.writerow(fields)
                            recorded.writerows(rows)
                        temporary.replace(directory / filename)
                shutil.rmtree(checkpoint_dir)
            if on_complete is not None:
                on_complete(repetition)
            print(f"{profile.profile}: pool {pool}, run {repetition}/"
                  f"{repetitions} complete", flush=True)
    print(f"Output: {output}")


def run_all(name, darshan=None, repetitions=3, pools=None):
    """Run the workload's conditions, restoring pool 6 after full success."""
    from configure_pool import restore, transition

    if name not in CLASSES:
        raise ValueError(f"Unsupported workload: {name}")
    if repetitions<1:
        raise ValueError("repetitions must be positive")
    allowed=EXPERIMENT_POOLS
    pools=allowed if pools is None else tuple(pools)
    if not pools or len(set(pools))!=len(pools) or not set(pools)<=set(allowed):
        raise ValueError(f"Expected distinct pools from {allowed}")
    if list(pools) != sorted(pools) or any((a,b) not in ((14,16),(16,18)) for a,b in zip(pools,pools[1:])):
        raise ValueError("Selected pools must follow 14→16→18 without skipping transitions")
    if darshan is not None:
        darshan = dlio_darshan.validate(darshan)
    profile = WORKLOADS[name]
    source = DATA_ROOT / profile.folder / "pool1/dataset/train"
    if len(list(source.glob(f"*.{profile.extension}"))) != profile.files:
        raise RuntimeError(f"Expected {profile.files} prepared input files in {source}")
    previous = 6
    for pool in pools:
        transition(previous, pool)
        print(f"Starting {repetitions} full {name} invocation(s) on pool {pool}", flush=True)
        run(name, pool, darshan=darshan, repetitions=repetitions)
        previous = pool
    restore(previous)
