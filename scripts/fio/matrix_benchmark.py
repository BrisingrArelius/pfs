#!/usr/bin/env python3
import argparse
import json
import subprocess
import os
import time
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

SCRIPT_DIR = Path(__file__).resolve().parent

def load_config(config_file):
    with open(config_file) as f:
        return json.load(f)

def run_fio(job_name, run_idx, fsize, mode_rw, block_size, io_depth, num_jobs, bdir, beegfs_trace=False):
    fio_text = f"""
[global]
ioengine=libaio
direct=1
fallocate=none
group_reporting=1
time_based=1
runtime=60
ramp_time=5
nrfiles=1
filesize={fsize}
iodepth={io_depth}
numjobs={num_jobs}
directory={bdir}

[{job_name}]
rw={mode_rw}
bs={block_size}
stonewall
"""
    fio_tmp = Path(bdir) / f"{job_name}_{run_idx}.fio"
    with open(fio_tmp, "w") as f:
        f.write(fio_text)
    
    # run fio and capture json
    cmd = ["fio", str(fio_tmp), "--output-format=json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
    except Exception as e:
        print(f"Error running fio: {e}")
        if hasattr(e, 'stdout'):
            print("stdout:", e.stdout)
        if hasattr(e, 'stderr'):
            print("stderr:", e.stderr)
        return None

    # Track BeegFS OST hits and clean up FIO test files
    ost_hits = []
    for p in Path(bdir).glob(f"{job_name}*"):
        if p == fio_tmp: continue
        
        if beegfs_trace:
            try:
                res = subprocess.run(["beegfs-ctl", "--getentryinfo", str(p)], capture_output=True, text=True)
                for line in res.stdout.splitlines():
                    if "Chunk ID:" in line or "Target:" in line:
                        ost_hits.append(line.strip())
            except:
                pass
                
        # Delete the large test files immediately after gathering entry info to prevent disk space accumulation
        try:
            p.unlink()
        except:
            pass

    return {
        "job": job_name,
        "fsize": fsize,
        "run": run_idx,
        "fio_data": data["jobs"][0],
        "ost_hits": ost_hits
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fio_config.json")
    parser.add_argument("--beegfs", action="store_true")
    parser.add_argument("--ost", action="store_true")
    parser.add_argument("--pool", default="all", help="Pool name to run on (all, hdd, ssd, or custom)")
    parser.add_argument("--custom-dir", default=None, help="Mount point if using a custom pool name")
    parser.add_argument("--hdd-dir", default="/mnt/beegfs/advay/hdd")
    parser.add_argument("--ssd-dir", default="/mnt/beegfs/advay/ssd")
    parser.add_argument("--hdd-ost-dir", default="/mnt/hdd")
    parser.add_argument("--ssd-ost-dir", default="/mnt/nvme")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-drop-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    modes_map = {
        "seq_read":   ("read", config["block_size_seq"]),
        "seq_write":  ("write", config["block_size_seq"]),
        "rand_read":  ("randread", config["block_size_rand"]),
        "rand_write": ("randwrite", config["block_size_rand"]),
        "seq_rw":     ("rw", config["block_size_seq"]),
        "rand_rw":    ("randrw", config["block_size_rand"]),
    }

    targets = []
    if args.beegfs:
        if args.pool in ["all", "hdd"]:
            targets.append(("HDD", args.hdd_dir, True))
        if args.pool in ["all", "ssd"]:
            targets.append(("SSD", args.ssd_dir, True))
        if args.pool not in ["all", "hdd", "ssd"]:
            target_d = args.custom_dir if args.custom_dir else f"/mnt/beegfs/advay/{args.pool}"
            targets.append((args.pool.upper(), target_d, True))
        os.makedirs(args.results_dir, exist_ok=True)
    if args.ost:
        res_dir = args.results_dir if args.beegfs else "ost_results"
        os.makedirs(res_dir, exist_ok=True)
        if args.pool in ["all", "hdd"]:
            for i in range(1, 5): targets.append((f"HDD_OST{i}", f"{args.hdd_ost_dir}{i}", False))
        if args.pool in ["all", "ssd"]:
            for i in range(1, 4): targets.append((f"SSD_OST{i}", f"{args.ssd_ost_dir}{i}", False))
        if args.pool not in ["all", "hdd", "ssd"]:
            target_d = args.custom_dir if args.custom_dir else f"/mnt/{args.pool}_ost"
            targets.append((f"{args.pool.upper()}_OST", target_d, False))

    if not targets:
        print("Must specify --beegfs or --ost")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.results_dir if args.beegfs else "ost_results") / f"matrix_results_{ts}.json"
    
    all_results = []

    for pool, target_dir, is_beegfs in targets:
        print(f"\n=====================================")
        print(f" Starting matrix for {pool} ({target_dir})")
        print(f"=====================================")

        if not os.path.exists(target_dir):
            print(f"Target directory {target_dir} missing, skipping.")
            continue

        work_dir = os.path.join(target_dir, f"fio_matrix_bench")
        os.makedirs(work_dir, exist_ok=True)

        for fsize in config["file_sizes"]:
            for mode, enabled in config["modes"].items():
                if not enabled: continue

                fio_rw, fio_bs = modes_map[mode]

                for run_idx in range(1, config["runs_per_test"] + 1):
                    print(f"  [{pool}] {mode} ({fsize}) - Run {run_idx}/{config['runs_per_test']}...")
                    
                    if not args.no_drop_cache:
                        subprocess.run("sync", shell=True)
                        subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, stderr=subprocess.DEVNULL)
                        time.sleep(1)

                    res = run_fio(
                        job_name=f"{pool}_{mode}_{fsize}",
                        run_idx=run_idx,
                        fsize=fsize,
                        mode_rw=fio_rw,
                        block_size=fio_bs,
                        io_depth=config["io_depth"],
                        num_jobs=config["num_jobs"],
                        bdir=work_dir,
                        beegfs_trace=is_beegfs
                    )
                    
                    if res:
                        all_results.append({
                            "pool": pool,
                            "mode": mode,
                            "fsize": fsize,
                            "run": run_idx,
                            "read_bw_mib": res["fio_data"]["read"]["bw_bytes"] / 1048576,
                            "write_bw_mib": res["fio_data"]["write"]["bw_bytes"] / 1048576,
                            "read_iops": res["fio_data"]["read"]["iops"],
                            "write_iops": res["fio_data"]["write"]["iops"],
                            "read_lat_ms": res["fio_data"]["read"].get("clat_ns", {}).get("mean", 0) / 1000000,
                            "write_lat_ms": res["fio_data"]["write"].get("clat_ns", {}).get("mean", 0) / 1000000,
                            "ost_hits": res["ost_hits"]
                        })
                        
                        # Save midway to not lose data
                        with open(out_file, "w") as f:
                            json.dump(all_results, f, indent=2)

        # clear work dir for this pool
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nAll Done. Full output saved to {out_file}")

if __name__ == "__main__":
    main()
