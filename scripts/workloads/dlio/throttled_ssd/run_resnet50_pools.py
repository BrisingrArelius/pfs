#!/usr/bin/env python3
"""ResNet-50: reservation-aware SSD sweep, or one configured pool."""

import argparse
import csv
import fcntl
import json
import re
import signal
import time
from datetime import datetime
from pathlib import Path

import pool_common as common
from configure_pool import apply_limits, clear_limits, normal_members, restore, transition, verify_normal


CONDITIONS = (14, 16, 18)


def duration(value):
    """Accept seconds or a duration such as 6h, 90m or 1h30m."""
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value)
    else:
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", value)
        if not match or not any(match.groups()):
            raise argparse.ArgumentTypeError("Use seconds, 6h, 90m or 1h30m")
        seconds = sum(float(number or 0) * unit for number, unit in zip(match.groups(), (3600, 60, 1)))
    if seconds <= 0:
        raise argparse.ArgumentTypeError("Deadline duration must be positive")
    return seconds


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def current_pool():
    listing = common.beegfs("--liststoragepools")
    candidates = [pool for pool in (6, *common.FAST)
                  if set(common.TARGETS) <= normal_members(listing, pool)]
    if len(candidates) != 1:
        raise RuntimeError("Experiment targets are split or missing; inspect pool membership before continuing")
    return candidates[0]


def adopt_legacy(base):
    """Import completed evidence from the old run_all('resnet') launcher."""
    def candidates(pool):
        found = []
        for directory in sorted(base.glob(f"runs-pool{pool}-*")):
            condition = directory / "condition.json"
            if condition.exists() and json.loads(condition.read_text()).get("profile") == "resnet50_v100":
                found.append(directory)
        return found

    anchors = candidates(14)
    if not anchors:
        raise RuntimeError("No interrupted sweep journal or legacy ResNet pool-14 results found")
    anchor = anchors[-1]
    first_stamp = anchor.name.removeprefix("runs-pool14-")
    state = {"version": 1, "workload": "resnet", "status": "interrupted",
             "adopted_from_legacy": True, "pools": {}}
    for pool in CONDITIONS:
        found = [path for path in candidates(pool)
                 if path.name.removeprefix(f"runs-pool{pool}-") >= first_stamp]
        if not found:
            continue
        directory = found[-1]
        condition = json.loads((directory / "condition.json").read_text())
        if (condition["pool"] != pool or set(condition["targets"]) != set(common.TARGETS)
                or set(condition["fast"]) != set(common.FAST[pool])):
            raise RuntimeError(f"Cannot adopt mismatched condition: {directory}")
        evidence = {}
        for filename in ("metrics.csv", "device_io.csv"):
            path = directory / filename
            with path.open(newline="") as file:
                evidence[filename] = list(csv.DictReader(file))
        completed = []
        for repetition in (1, 2, 3):
            raw = directory / f"rep{repetition}-pool{pool}"
            rows = {name: [row for row in values if row["repetition"] == str(repetition)]
                    for name, values in evidence.items()}
            if len(rows["metrics.csv"]) != 8 or len(rows["device_io.csv"]) != 4:
                continue
            required = [raw / name for name in ("summary.json", "per_epoch_stats.json", "checkpoint-osts.tsv",
                                                *(f"{rank}_output.json" for rank in range(common.RANKS)))]
            if not all(path.exists() for path in required):
                continue
            summary = json.loads((raw / "summary.json").read_text())
            native = json.loads((raw / "per_epoch_stats.json").read_text())
            if (not summary.get("end") or summary.get("num_files_train") != common.WORKLOADS["resnet"].files
                    or summary.get("num_accelerators") != common.RANKS
                    or sorted(map(int, native)) != [1, 2, 3, 4, 5]
                    or any("duration" not in native[str(epoch)] for epoch in range(1, 6))
                    or any("ckpt1" not in native[str(epoch)] for epoch in (2, 4))):
                continue
            if (any(row["pool"] != str(pool) for values in rows.values() for row in values)
                    or {int(row["target"]) for row in rows["device_io.csv"]} != set(common.TARGETS)):
                raise RuntimeError(f"Cannot adopt wrong-pool or duplicate device evidence: {raw}")
            keys = {(row["phase"], row["epoch"]) for row in rows["metrics.csv"]}
            expected = {("training_input_block", str(epoch)) for epoch in range(1, 6)}
            expected.update(("checkpoint_save_rank0", str(epoch)) for epoch in (2, 4))
            expected.add(("training_summary", ""))
            if keys != expected:
                continue
            # Make adopted repetitions as crash-resilient as new ones.
            for filename, values in rows.items():
                temporary = raw / (filename + ".tmp")
                with temporary.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=values[0])
                    writer.writeheader()
                    writer.writerows(values)
                temporary.replace(raw / filename)
            completed.append(repetition)
        state["pools"][str(pool)] = {"output": str(directory), "completed": completed}
        print(f"Adopting pool {pool}: completed repetitions {completed}; {directory}", flush=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    state_path = base / f"sweep-{stamp}.json"
    save_state(state_path, state)
    return state_path, state


def sweep(base, resume=False, seconds=None):
    reservation_end = time.monotonic() + seconds if seconds is not None else None
    reserve = min(60, seconds * .1) if seconds is not None else 0
    common.DEADLINE = reservation_end - reserve if reservation_end is not None else None
    common.MANAGED = True
    unfinished = []
    for path in sorted(base.glob("sweep-*.json"), reverse=True):
        state = json.loads(path.read_text())
        if state["status"] != "complete":
            unfinished.append((path, state))
    if resume:
        if not unfinished:
            state_path, state = adopt_legacy(base)
        else:
            state_path, state = unfinished[0]
    else:
        if unfinished:
            raise RuntimeError(f"Unfinished sweep exists: {unfinished[0][0]}; use --continue")
        # Validate before recording a new sweep or touching target membership.
        verify_normal()
        source = common.DATA_ROOT / common.WORKLOADS["resnet"].folder / "pool1/dataset/train"
        if len(list(source.glob("*.tfrecord"))) != common.WORKLOADS["resnet"].files:
            raise RuntimeError(f"ResNet inputs are not prepared: {source}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        state_path = base / f"sweep-{stamp}.json"
        state = {"version": 1, "workload": "resnet", "status": "running", "pools": {}}
    state["status"] = "running"
    state["conditions"] = list(CONDITIONS)
    save_state(state_path, state)
    print(f"Sweep state: {state_path}", flush=True)
    if seconds is not None:
        print(f"Reservation budget: {seconds / 3600:.2f} hours; reserving {reserve:.0f}s for stopping/restoration", flush=True)
    try:
        active = current_pool()
        if active == 6:
            if resume:
                clear_limits()
            verify_normal()
        else:
            # A previous interruption may have left partially installed caps.
            apply_limits(active)
            common.verify(active)
            restore(active)
            active = 6
        for pool in CONDITIONS:
            progress = state["pools"].get(str(pool))
            if progress is not None and progress["completed"] == [1, 2, 3]:
                print(f"Pool {pool}: all three runs already complete; skipping", flush=True)
                continue
            common.remaining()
            if active != pool:
                transition(active, pool)
                active = pool
            if progress is None:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                progress = {"output": str(base / f"runs-pool{pool}-{stamp}"), "completed": []}
                state["pools"][str(pool)] = progress
                save_state(state_path, state)

            def completed(repetition):
                progress["completed"] = sorted(set(progress["completed"]) | {repetition})
                save_state(state_path, state)

            common.run("resnet", pool, output=Path(progress["output"]),
                       completed=progress["completed"], on_complete=completed)
        # Use the reserved tail even if the last repetition consumed its budget.
        common.DEADLINE = reservation_end
        restore(active) if active != 6 else verify_normal()
        state["status"] = "complete"
        save_state(state_path, state)
        print("ResNet sweep complete; pool 6 restored.", flush=True)
    except (common.DeadlineReached, KeyboardInterrupt) as error:
        state["status"] = "interrupted"
        save_state(state_path, state)
        print(f"Stopping: {error or 'interrupted'}. Completed repetitions are saved.", flush=True)
        common.DEADLINE = reservation_end if reservation_end is not None else time.monotonic() + 60
        try:
            active = current_pool()
            if active != 6:
                apply_limits(active)
                restore(active)
            else:
                clear_limits()
                verify_normal()
        except Exception as cleanup_error:
            print(f"Restoration incomplete: {cleanup_error}. Inspect target membership/caps before releasing the reservation.", flush=True)
        print("Continue with: python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --continue --deadline 6h", flush=True)
    except Exception:
        state["status"] = "interrupted"
        save_state(state_path, state)
        raise
    finally:
        common.DEADLINE = None
        common.MANAGED = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pool", type=int, choices=CONDITIONS, help="Run one already configured condition")
    mode.add_argument("--all-pools", action="store_true", help="Run pools 14, 16, 18 (4/0, 2/2, 0/4), then restore pool 6 (default)")
    mode.add_argument("--continue", dest="resume", action="store_true", help="Continue the newest unfinished sweep; restart its partial repetition")
    parser.add_argument("--deadline", type=duration, help="Total wall-clock budget from invocation, e.g. 6h or 90m; includes staging and cleanup")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pool is not None:
        if args.deadline is not None:
            parser.error("--deadline applies to the managed sweep, not single --pool runs")
        common.run("resnet", args.pool, args.pilot)
        return
    if args.pilot:
        parser.error("--pilot requires a single --pool")
    base = common.RESULT_ROOT / "throttled_ssd" / common.WORKLOADS["resnet"].folder
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another managed ResNet sweep is already running")
        def stop(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        old_handlers = {signum: signal.signal(signum, stop)
                        for signum in (signal.SIGTERM, signal.SIGHUP)}
        try:
            sweep(base, args.resume, args.deadline)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            common.DEADLINE = None
            common.MANAGED = False


if __name__ == "__main__":
    main()
