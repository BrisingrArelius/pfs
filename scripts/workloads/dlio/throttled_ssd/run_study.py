#!/usr/bin/env python3
"""Reservation-aware, resumable CosmoFlow then U-Net Darshan study."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from datetime import datetime
from uuid import uuid4

import pool_common as common
from configure_pool import clear_limits, normal_members, restore, transition, verify_normal


WORKLOADS = ("cosmoflow", "unet")
POOLS = (14, 16, 18)
REPETITIONS = 3


def duration(value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError("Use e.g. 6h, 6hours, 90m or seconds")
    number, unit = match.groups()
    unit = (unit or "s").lower()
    factor = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
    result = float(number)*factor
    if result <= 0:
        raise argparse.ArgumentTypeError("Deadline must be positive")
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(library):
    here = Path(__file__).resolve().parent
    normal = here.parent/"normal"
    files = [here/name for name in ("run_study.py", "pool_common.py", "configure_pool.py", "experiment_config.py")]
    files += [normal/name for name in ("dlio_darshan.py", "dlio_pool_profiles.py", "pool_common.py",
                                        "run_cosmoflow_pools.py", "run_unet3d_pools.py")]
    return {"fast_MiB_s":common.FAST_MIB_S, "slow_MiB_s":common.SLOW_MIB_S,
            "results_subdir":common.RESULTS_SUBDIR, "pools":list(POOLS), "repetitions":REPETITIONS,
            "targets":list(common.TARGETS), "workloads":{name:asdict(common.WORKLOADS[name]) for name in WORKLOADS},
            "dlio_version":importlib.metadata.version("dlio-benchmark"),
            "darshan_library":str(library), "darshan_sha256":sha(library),
            "sources":{str(path.relative_to(here.parent)):sha(path) for path in files}}


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2)+"\n")
    temporary.replace(path)


def current_pool():
    listing = common.beegfs("--liststoragepools")
    candidates = [pool for pool in (6, *common.FAST)
                  if set(common.TARGETS) <= normal_members(listing, pool)]
    if len(candidates) != 1:
        raise RuntimeError("Study targets are split or missing; inspect pool membership before moving targets")
    return candidates[0]


def normalize(identifier):
    common.stop_owned(identifier)
    active = current_pool()
    if active == 6:
        clear_limits()
        verify_normal()
    else:
        restore(active)


def sweep(base, library, seconds, resume=False, started=None):
    reservation_end = (time.monotonic() if started is None else started)+seconds
    reserve = min(120, seconds*.1)
    common.DEADLINE = reservation_end-reserve
    common.MANAGED = True
    ident = identity(library)
    if ident["dlio_version"] != "2.0.0":
        raise RuntimeError("DLIO 2.0.0 is required")
    existing = []
    for path in sorted(base.glob("study-*.json"), reverse=True):
        state = json.loads(path.read_text())
        if state["status"] != "complete":
            existing.append((path,state))
    if len(existing)>1:
        raise RuntimeError("Multiple unfinished studies: inspect the study journals before continuing")
    if existing:
        state_path,state = existing[0]
        if state["identity"] != ident:
            raise RuntimeError(f"Cannot continue {state_path}: runtime, source, workload or rate settings changed")
        print(f"Continuing unfinished study: {state_path}",flush=True)
    elif resume:
        raise RuntimeError(f"No unfinished study found in {base}")
    else:
        common.remaining()
        verify_normal()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        state_path = base/f"study-{stamp}.json"
        state = {"version":1, "id":uuid4().hex, "status":"running", "identity":ident,
                 "workloads":{name:{"pools":{}} for name in WORKLOADS}}
        save(state_path,state)
        print(f"New study: {state_path}",flush=True)
    common.MANAGED_ID = state["id"]
    state["status"] = "running"
    save(state_path,state)
    print(f"Budget: {seconds:g}s; {reserve:g}s reserved for stopping and restoring pool 6.",flush=True)
    try:
        normalize(state["id"])
        active = 6
        for name in WORKLOADS:
            profile = common.WORKLOADS[name]
            for pool in POOLS:
                progress = state["workloads"][name]["pools"].get(str(pool))
                if progress and progress["completed"] == [1,2,3]:
                    print(f"{name} pool {pool}: all three repetitions complete; skipping",flush=True)
                    continue
                common.remaining()
                if active != pool:
                    common.stop_owned(state["id"])
                    if active != 6 and (active,pool) not in ((14,16),(16,18)):
                        restore(active)
                        active = 6
                    transition(active,pool)
                    active = pool
                if progress is None:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    output = base/profile.folder/f"runs-pool{pool}-{stamp}"
                    progress = {"output":str(output), "completed":[]}
                    state["workloads"][name]["pools"][str(pool)] = progress
                    save(state_path,state)

                def completed(repetition):
                    progress["completed"] = sorted(set(progress["completed"]) | {repetition})
                    save(state_path,state)

                common.run(name,pool,output=Path(progress["output"]),completed=progress["completed"],
                           on_complete=completed,darshan=library,repetitions=REPETITIONS)
        common.DEADLINE = reservation_end
        normalize(state["id"])
        state["status"] = "complete"
        save(state_path,state)
        print(f"Both workloads complete; pool 6 restored. Study: {state_path}",flush=True)
        return 0
    except BaseException as error:
        state["status"] = "interrupted"
        state["last_error"] = str(error)
        save(state_path,state)
        print(f"Study interrupted: {error or type(error).__name__}. Completed repetitions are saved.",file=sys.stderr,flush=True)
        common.DEADLINE = reservation_end
        try:
            normalize(state["id"])
            print("Pool 6 restored; continue with run.sh --continue --deadline 6hours",flush=True)
        except BaseException as cleanup_error:
            print(f"Restoration incomplete: {cleanup_error}. Inspect running processes and pool membership.",file=sys.stderr,flush=True)
        if isinstance(error,(common.DeadlineReached,KeyboardInterrupt)):
            return 75
        raise
    finally:
        common.DEADLINE = None
        common.MANAGED = False
        common.MANAGED_ID = None


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline",type=duration,required=True,help="Shared wall-clock budget, e.g. 6hours or 6h")
    parser.add_argument("--continue",dest="resume",action="store_true",help="Require and resume an unfinished study")
    parser.add_argument("--darshan-library",type=Path,default=Path(os.environ.get("DARSHAN_RUNTIME_LIB",str(Path.home()/"darshan-nompi/lib/libdarshan.so.0.0.0"))))
    args = parser.parse_args()
    library = common.dlio_darshan.validate(args.darshan_library)
    base = common.RESULT_ROOT/common.RESULTS_SUBDIR
    base.mkdir(parents=True,exist_ok=True)
    with (base/".cosmoflow-unet.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another combined study is running")
        def stop(signum,frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        old = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGHUP)}
        try:
            return sweep(base,library,args.deadline,args.resume,started=started)
        finally:
            for sig,handler in old.items():
                signal.signal(sig,handler)
            common.DEADLINE = None
            common.MANAGED = False
            common.MANAGED_ID = None


if __name__ == "__main__":
    sys.exit(main())
