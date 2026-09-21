#!/usr/bin/env python3
"""Run the local-storage FIO matrix across every target on the current host."""

import argparse
from datetime import datetime
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

from local_storage_fio import (
    cleanup_data_files,
    prepare_read_files,
    run_fio,
    write_json_atomic,
)
from local_storage_run_control import (
    BudgetExpired,
    capacity_snapshot,
    check_cache_drop_permission,
    drop_caches,
    error_record,
    extend_deadline,
    fio_version,
    initialize_deadline,
    load_json,
    parse_deadline,
    parse_duration,
    parse_fio_size,
    read_active_deadline,
    run_with_deadline,
    verify_inventory_mount,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def main():
    """Validate the host, restore run state, and process targets sequentially."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", help="Directory for the result JSON (default: a new run directory)")
    parser.add_argument("--resume", action="store_true", help="Resume completed targets from --results-dir")
    deadline_group = parser.add_mutually_exclusive_group()
    deadline_group.add_argument("--time-limit", type=parse_duration, metavar="DURATION")
    deadline_group.add_argument("--deadline", type=parse_deadline, metavar="ISO_TIME")
    parser.add_argument("--extend-deadline", type=parse_duration, metavar="DURATION")
    parser.add_argument("--cleanup-buffer", type=parse_duration, default=300, metavar="DURATION")
    args = parser.parse_args()

    if args.extend_deadline is not None:
        if not args.results_dir:
            parser.error("--extend-deadline requires --results-dir")
        deadline_file = Path(args.results_dir) / "deadline_state.json"
        if not deadline_file.exists():
            parser.error(f"no active deadline state found in {args.results_dir}")
        try:
            deadline_state = extend_deadline(deadline_file, args.extend_deadline)
        except ValueError as error:
            parser.error(str(error))
        print(f"Deadline extended to {deadline_state['deadline']}")
        return

    config_file = SCRIPT_DIR / "fio_config.json"
    inventory_file = SCRIPT_DIR / "target_inventory.json"
    config = load_json(config_file)
    inventory = load_json(inventory_file)
    host = socket.gethostname().split(".")[0]
    if host not in inventory:
        parser.error(f"host {host!r} is not present in {inventory_file}")
    targets = [
        (f"TARGET_{target['target_id']}", target["mount"], target)
        for target in inventory[host]
    ]
    free_space_buffer = parse_fio_size(config["free_space_buffer"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.resume and not args.results_dir:
        parser.error("--resume requires --results-dir")

    deadline = args.deadline
    if args.time_limit is not None:
        deadline = time.time() + args.time_limit
    if deadline is not None and deadline - args.cleanup_buffer <= time.time():
        parser.error("deadline must leave more time than --cleanup-buffer")

    try:
        check_cache_drop_permission()
    except RuntimeError as error:
        parser.error(str(error))

    results_dir = Path(args.results_dir) if args.results_dir else (
        REPO_ROOT / "results" / "microbenchmarks" / "runs" / f"fio-{ts}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = results_dir / "run_manifest.json"
    deadline_file = results_dir / "deadline_state.json"
    if not args.resume and manifest_file.exists():
        parser.error(f"{results_dir} already contains a run; use --resume or a new directory")

    if args.resume:
        if not manifest_file.exists():
            parser.error(f"no run manifest found in {results_dir}")
        manifest = load_json(manifest_file)
        if manifest.get("host") != host:
            parser.error("run manifest belongs to a different host")
        if manifest.get("config") != config:
            parser.error("current FIO configuration differs from the run manifest")
        if manifest.get("selected_targets") != [target[0] for target in targets]:
            parser.error("current target selection differs from the run manifest")
        out_file = results_dir / manifest["results_file"]
        all_results = load_json(out_file) if out_file.exists() else []
        manifest.setdefault("resumed_at", []).append(datetime.now().astimezone().isoformat())
        manifest.setdefault("resume_commands", []).append(sys.argv)
        manifest["status"] = "running"
        manifest["deadline"] = (
            datetime.fromtimestamp(deadline).astimezone().isoformat() if deadline else None
        )
    else:
        out_file = results_dir / f"matrix_results_{ts}.json"
        all_results = []
        manifest = {
            "run_id": f"fio-{ts}",
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "host": host,
            "command": sys.argv,
            "results_file": out_file.name,
            "config_file": str(config_file.resolve()),
            "config": config,
            "inventory_file": str(inventory_file.resolve()),
            "inventory_targets": inventory[host],
            "fio_version": fio_version(),
            "effective_uid": os.geteuid(),
            "python_version": platform.python_version(),
            "kernel": platform.release(),
            "selected_targets": [target[0] for target in targets],
            "deadline": datetime.fromtimestamp(deadline).astimezone().isoformat() if deadline else None,
            "deadline_state_file": deadline_file.name,
            "cleanup_buffer_seconds": args.cleanup_buffer,
            "free_space_buffer_bytes": free_space_buffer,
            "cache_drop_enabled": True,
            "targets": {},
            "failures": [],
        }

    initialize_deadline(deadline_file, deadline)

    write_json_atomic(out_file, all_results)
    write_json_atomic(manifest_file, manifest)

    def run_fio_command(command):
        """Bind each FIO subprocess to this run's mutable reservation deadline."""
        return run_with_deadline(command, deadline_file, args.cleanup_buffer)

    for target in targets:
        pool, target_dir, target_metadata = target
        target_state = manifest["targets"].setdefault(pool, {
            "status": "pending",
            "path": target_dir,
            "metadata": target_metadata,
            "attempts": 0,
        })
        if target_state["status"] == "completed":
            print(f"Skipping completed target {pool}.")
            continue

        # A target is the resume unit. Partial rows from an earlier attempt are not reused.
        all_results = [row for row in all_results if row.get("pool") != pool]
        write_json_atomic(out_file, all_results)

        print(f"\n=====================================")
        print(f" Starting matrix for {pool} ({target_dir})")
        print(f"=====================================")

        if not os.path.exists(target_dir):
            error = RuntimeError(f"target directory {target_dir} is missing")
            failure = error_record(error, "target_preflight", {"target": pool})
            target_state.update({"status": "failed", "failure": failure})
            manifest["failures"].append(failure)
            write_json_atomic(manifest_file, manifest)
            continue
        if not os.path.ismount(target_dir):
            error = RuntimeError(f"target directory {target_dir} is not mounted")
            failure = error_record(error, "target_preflight", {"target": pool})
            target_state.update({"status": "failed", "failure": failure})
            manifest["failures"].append(failure)
            write_json_atomic(manifest_file, manifest)
            continue

        try:
            target_state["verified_mount"] = verify_inventory_mount(target_dir, target_metadata)
        except Exception as error:
            failure = error_record(error, "target_preflight", {"target": pool})
            target_state.update({"status": "failed", "failure": failure})
            manifest["failures"].append(failure)
            write_json_atomic(manifest_file, manifest)
            continue

        work_dir = os.path.join(target_dir, f"fio_matrix_bench")
        target_artifact_dir = results_dir / "raw" / pool
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(target_artifact_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        target_state.update({
            "status": "running",
            "attempts": target_state.get("attempts", 0) + 1,
            "started_at": datetime.now().astimezone().isoformat(),
            "capacity_before": capacity_snapshot(target_dir),
        })
        write_json_atomic(manifest_file, manifest)

        context = {"target": pool}
        try:
            for num_files in config.get("num_files", [1]):
                for fsize in config["file_sizes"]:
                    required_bytes = parse_fio_size(fsize) * config["num_jobs"] + free_space_buffer
                    current_capacity = capacity_snapshot(target_dir)
                    if current_capacity["free_bytes"] < required_bytes:
                        raise RuntimeError(
                            f"insufficient free space: need {required_bytes} bytes, have {current_capacity['free_bytes']} bytes"
                        )

                    for workload in config["workloads"]:
                        mode = workload["name"]
                        context = {
                            "target": pool,
                            "mode": mode,
                            "num_files": num_files,
                            "fsize": fsize,
                        }
                        fio_rw = workload["rw"]
                        fio_bs = workload["block_size"]
                        job_name = f"{pool}_{mode}_{num_files}f_{fsize}"
                        artifact_dir = target_artifact_dir / f"{num_files}f_{fsize}" / mode
                        read_only = workload["prepare"]

                        if read_only:
                            print(f"  [{pool}] Preparing files for {mode} ({num_files} files, {fsize})...")
                            prepare_read_files(
                                job_name, num_files, fsize,
                                workload.get("prepare_block_size", "1m"),
                                config["io_depth"], config["num_jobs"], work_dir,
                                artifact_dir, run_fio_command
                            )

                        try:
                            for run_idx in range(1, config["runs_per_test"] + 1):
                                context["run"] = run_idx
                                print(f"  [{pool}] {mode} ({num_files} files, {fsize}) - Run {run_idx}/{config['runs_per_test']}...")

                                active_deadline = read_active_deadline(deadline_file)
                                if active_deadline is not None and time.time() >= active_deadline - args.cleanup_buffer:
                                    raise BudgetExpired("reservation deadline reached before cache drop")
                                drop_caches(deadline_file, args.cleanup_buffer)
                                time.sleep(1)

                                res = run_fio(
                                    job_name=job_name,
                                    run_index=run_idx,
                                    num_files=num_files,
                                    size=fsize,
                                    mode=fio_rw,
                                    block_size=fio_bs,
                                    io_depth=config["io_depth"],
                                    num_jobs=config["num_jobs"],
                                    time_based=workload.get("time_based", False),
                                    runtime=workload.get("runtime_seconds", 0),
                                    ramp_time=workload.get("ramp_time_seconds", 0),
                                    work_dir=work_dir,
                                    artifact_dir=artifact_dir,
                                    run_command=run_fio_command,
                                    cleanup_files=not read_only
                                )

                                all_results.append({
                                    "pool": pool,
                                    "mode": mode,
                                    "num_files": num_files,
                                    "fsize": fsize,
                                    "run": run_idx,
                                    "host": host,
                                    "target": target_metadata,
                                    "read_bw_mib": res["read"]["bw_bytes"] / 1048576,
                                    "write_bw_mib": res["write"]["bw_bytes"] / 1048576,
                                    "read_iops": res["read"]["iops"],
                                    "write_iops": res["write"]["iops"],
                                    "read_lat_ms": res["read"].get("clat_ns", {}).get("mean", 0) / 1000000,
                                    "write_lat_ms": res["write"].get("clat_ns", {}).get("mean", 0) / 1000000,
                                    "fio_job_file": str((artifact_dir / f"run_{run_idx}.fio").relative_to(results_dir)),
                                    "raw_fio_json": str((artifact_dir / f"run_{run_idx}.json").relative_to(results_dir))
                                })
                                write_json_atomic(out_file, all_results)
                        finally:
                            if read_only:
                                cleanup_data_files(work_dir, job_name)

            target_state.update({
                "status": "completed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "capacity_after": capacity_snapshot(target_dir),
            })
            write_json_atomic(manifest_file, manifest)
        except (BudgetExpired, KeyboardInterrupt) as error:
            phase = "budget" if isinstance(error, BudgetExpired) else "interrupt"
            failure = error_record(error, phase, context)
            target_state.update({
                "status": "interrupted",
                "capacity_after": capacity_snapshot(target_dir),
                "failure": failure,
            })
            manifest["failures"].append(failure)
            manifest["status"] = "budget_expired" if isinstance(error, BudgetExpired) else "interrupted"
            manifest["final_deadline_state"] = load_json(deadline_file)
            write_json_atomic(out_file, all_results)
            write_json_atomic(manifest_file, manifest)
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"\nRun interrupted. Resume with the same options plus --resume --results-dir {results_dir}")
            return
        except Exception as error:
            failure = error_record(error, "benchmark", context)
            target_state.update({
                "status": "failed",
                "capacity_after": capacity_snapshot(target_dir),
                "failure": failure,
            })
            manifest["failures"].append(failure)
            write_json_atomic(manifest_file, manifest)
            print(f"Target {pool} failed: {error}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    failed_targets = [state for state in manifest["targets"].values() if state["status"] != "completed"]
    manifest["status"] = "completed_with_failures" if failed_targets else "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["final_deadline_state"] = load_json(deadline_file)
    write_json_atomic(manifest_file, manifest)
    print(f"\nAll Done. Full output saved to {out_file}")

if __name__ == "__main__":
    main()
