#!/usr/bin/env python3
"""
run_workloads.py

Compiles posix_synthetic_workload.c, reads workload profiles from profiles.json,
runs each profile under Darshan instrumentation, then invokes parse_darshan.py
on the resulting log file.

Usage:
    python run_workloads.py [options]

Examples:
    # Run all profiles once
    python run_workloads.py

    # Run all profiles 5 times each
    python run_workloads.py --runs 5

    # Run only specific profiles
    python run_workloads.py --only read_heavy write_heavy

    # Run a specific profile 10 times
    python run_workloads.py --only read_heavy --runs 10

    # Dry run — show what would be executed without running anything
    python run_workloads.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR    = os.path.join(SCRIPT_DIR, "workloads")
PROFILES_JSON    = os.path.join(WORKLOADS_DIR, "profiles.json")
WORKLOAD_SRC     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload.c")
WORKLOAD_BIN     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload")
WORKLOAD_WORK_DIR = "/mnt/beegfs/advay/ssd/workloads/tmp"          # scratch dir for workload files
PARSE_SCRIPT     = os.path.join(SCRIPT_DIR, "parse_darshan.py")
OUTPUT_DIR       = "./output/ssd/darshan"

DARSHAN_PRELOAD  = "/usr/local/lib/libdarshan.so.0.0.0"  # ← update if different on your system
DARSHAN_LOG_DIR  = "/mnt/nfs_shared/darshan-logs"  # ← Darshan's compiled-in path
# MPI configuration
MPICC            = "mpicc"                      # MPI C compiler
MPIRUN           = "mpirun"                     # MPI launcher

# Darshan modules to parse. metadata_heavy only touches POSIX.
DEFAULT_MODULES  = ["posix"]

# Access pattern name → integer for the C binary
ACCESS_PATTERN_MAP = {
    "sequential": 0,
    "random":     1,
    "strided":    2,
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile posix_synthetic_workload.c, run profiles, parse Darshan logs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of times to run each profile (default: 1)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="PROFILE",
        default=None,
        help="Run only these profiles by name. Runs all if omitted."
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["posix", "mpi", "stdio"],
        default=None,
        help=f"Darshan modules to extract (default: {DEFAULT_MODULES})"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for CSVs (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--workload-dir",
        default=WORKLOAD_WORK_DIR,
        help=f"Directory where workload files are created (default: {WORKLOAD_WORK_DIR})"
    )
    parser.add_argument(
        "--profiles",
        default=PROFILES_JSON,
        help=f"Path to profiles JSON file (default: {PROFILES_JSON})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing anything"
    )
    return parser.parse_args()


# =============================================================================
# COMPILATION
# =============================================================================

def compile_workload(dry_run):
    """Compile posix_synthetic_workload.c with mpicc + explicit Darshan linkage."""
    cmd = [
        MPICC, "-O2", 
        "-o", WORKLOAD_BIN, 
        WORKLOAD_SRC,
        "-L/usr/local/lib", "-ldarshan", "-lpthread", "-lrt", "-lz"
    ]
    print(f"Compiling: {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Compilation failed.")
        return False
    print("Compilation successful.\n")
    return True


# =============================================================================
# PROFILE LOADING
# =============================================================================

def load_profiles(profiles_path, only=None):
    """Load and validate profiles from JSON. Returns list of (name, params) tuples."""
    if not os.path.isfile(profiles_path):
        print(f"Error: profiles file not found: {profiles_path}")
        sys.exit(1)

    with open(profiles_path) as f:
        raw = json.load(f)

    profiles = []
    for name, params in raw.items():
        if only and name not in only:
            continue
        # Validate access pattern
        if params["access_pattern"] not in ACCESS_PATTERN_MAP:
            print(f"Warning: unknown access_pattern '{params['access_pattern']}' "
                  f"in profile '{name}' — skipping.")
            continue
        profiles.append((name, params))

    if only:
        found = {n for n, _ in profiles}
        for requested in only:
            if requested not in found:
                print(f"Warning: profile '{requested}' not found in {profiles_path}")

    return profiles


# =============================================================================
# DARSHAN LOG DETECTION
# =============================================================================

def find_latest_darshan_log(before_files):
    """
    Find the newest .darshan log written to DARSHAN_LOG_DIR since before_files snapshot.
    Darshan organizes logs by date: DARSHAN_LOG_DIR/YYYY/M/D/*.darshan
    """
    import glob
    import time
    
    # Wait briefly for filesystem to sync
    time.sleep(0.5)
    
    # Search recursively for .darshan files
    all_logs = glob.glob(f"{DARSHAN_LOG_DIR}/**/*.darshan", recursive=True)
    
    if not all_logs:
        return None
    
    # Filter to only newly created logs (compare by mtime)
    # Since we can't track exact before_files in subdirs, just return the newest
    return max(all_logs, key=os.path.getmtime)


def needs_setup(params):
    """
    Returns True only for pure-read profiles (read_ratio == 1.0).
    These need a setup pass to pre-populate files before the measured run.

    Mixed profiles (0 < read_ratio < 1.0) create their own files in workload
    mode — the write phases handle file creation, so no setup is needed.

    metadata_heavy never needs setup — it manages its own files internally.
    """
    return params["read_ratio"] >= 1.0


def build_workload_cmd(name, params, mode, workload_dir):
    """Build the CLI arg list for the posix_synthetic_workload binary.
    
    For mode=0 (setup): run directly without MPI (no Darshan needed)
    For mode=1 (workload): run with mpirun -np 1 (Darshan requires MPI)
    """
    pattern_int = ACCESS_PATTERN_MAP[params["access_pattern"]]
    base_args = [
        WORKLOAD_BIN,
        name,
        str(params["read_ratio"]),
        str(pattern_int),
        str(params["stride_size"]),
        str(params["op_size"]),
        str(params["num_ops"]),
        str(params["num_files"]),
        str(params["num_phases"]),
        str(params["fsync_interval"]),
        workload_dir,  # Use passed directory instead of global
        str(mode),   # 0 = setup, 1 = workload
    ]
    
    # Workload mode (mode=1): wrap with mpirun for Darshan
    if mode == 1:
        return [MPIRUN, "-np", "1"] + base_args
    else:
        # Setup mode (mode=0): run directly
        return base_args


# =============================================================================
# RUNNING A SINGLE PROFILE
# =============================================================================

def run_profile(name, params, run_index, modules, output_dir, workload_dir, dry_run):
    """
    Run one profile once and parse the resulting Darshan log.

    For profiles with read_ratio >= 1.0 (pure-read):
      1. Run binary in mode 0 (setup) — writes files only,
         no Darshan instrumentation, no log generated.
      2. Run binary in mode 1 (workload) with mpirun -np 1 — measured run,
         reads from pre-existing files, Darshan records only the target I/O.

    For pure-write/mixed profiles (read_ratio < 1.0):
      - Run binary in mode 1 directly with mpirun. No setup needed.

    metadata_heavy:
      - Always run in mode 1 directly. It manages its own files internally.
    """
    label = f"{name}_run{run_index}"
    setup_required = needs_setup(params) and name != "metadata_heavy"

    setup_cmd    = build_workload_cmd(name, params, mode=0, workload_dir=workload_dir)
    workload_cmd = build_workload_cmd(name, params, mode=1, workload_dir=workload_dir)

    module_flags = [f"--{m}" for m in modules]
    parse_cmd = [
        sys.executable, PARSE_SCRIPT,
        "--label", label,
        "--output-dir", output_dir,
    ] + module_flags + ["--log", "<log>"]

    if dry_run:
        if setup_required:
            print(f"  [{label}] Would run setup (no Darshan): {' '.join(setup_cmd)}")
        print(f"  [{label}] Would run workload (MPI+Darshan): {' '.join(workload_cmd)}")
        print(f"  [{label}] Would parse: {' '.join(parse_cmd)}")
        return True

    # --- Setup phase (no Darshan, no MPI) ---
    if setup_required:
        print(f"  [{label}] Setup (no instrumentation): {' '.join(setup_cmd)}")
        result = subprocess.run(setup_cmd)
        if result.returncode != 0:
            print(f"  [{label}] Setup exited with code {result.returncode} — skipping workload.")
            return False

    # --- Workload phase (MPI + Darshan attached) ---
    print(f"  [{label}] Workload (instrumented): {' '.join(workload_cmd)}")
    before_files = set(os.listdir(DARSHAN_LOG_DIR))

    # Note: Darshan library is preloaded by the MPI runtime when configured properly.
    # If your system requires explicit LD_PRELOAD, you can uncomment and set it:
    # env = os.environ.copy()
    # env["LD_PRELOAD"] = DARSHAN_PRELOAD
    # result = subprocess.run(workload_cmd, env=env)
    
    result = subprocess.run(workload_cmd)
    if result.returncode != 0:
        print(f"  [{label}] Workload exited with code {result.returncode} — skipping parse.")
        return False

    # --- Locate and parse the Darshan log ---
    log_path = find_latest_darshan_log(before_files)
    if not log_path:
        print(f"  [{label}] No new .darshan log found in {DARSHAN_LOG_DIR} — skipping parse.")
        return False

    print(f"  [{label}] Log: {log_path}")
    parse_cmd[-1] = log_path
    parse_result = subprocess.run(parse_cmd)
    if parse_result.returncode != 0:
        print(f"  [{label}] Parser exited with code {parse_result.returncode}.")
        return False

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    modules = args.modules if args.modules else DEFAULT_MODULES

    # Compile
    if not compile_workload(args.dry_run):
        sys.exit(1)

    # Load profiles
    profiles = load_profiles(args.profiles, only=args.only)
    if not profiles:
        print("No profiles to run.")
        sys.exit(0)

    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.dry_run:
        os.makedirs(args.workload_dir, exist_ok=True)

    print(f"Profiles:         {[n for n, _ in profiles]}")
    print(f"Runs per profile: {args.runs}")
    print(f"Modules:          {modules}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Workload dir:     {args.workload_dir}")
    print(f"DARSHAN_LOG_DIR = {DARSHAN_LOG_DIR}\n")
    if args.dry_run:
        print("Dry run — nothing will be executed.\n")

    total     = len(profiles) * args.runs
    completed = 0
    failed    = 0

    for name, params in profiles:
        print(f"\n{'='*50}")
        print(f"Profile: {name}  ({args.runs} run(s))")
        print(f"{'='*50}")
        for i in range(1, args.runs + 1):
            success = run_profile(name, params, i, modules, args.output_dir, args.workload_dir, args.dry_run)
            if success:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"Done. {completed}/{total} completed. {failed} failed.")


if __name__ == "__main__":
    main()
