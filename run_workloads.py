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

WORKLOADS_DIR    = "./workloads"
PROFILES_JSON    = "./workloads/profiles.json"
WORKLOAD_SRC      = "./workloads/posix_synthetic_workload.c"
WORKLOAD_BIN      = "./workloads/posix_synthetic_workload"
WORKLOAD_WORK_DIR = "./workloads/tmp"          # scratch dir for workload files
PARSE_SCRIPT     = "./parse_darshan.py"
OUTPUT_DIR       = "./darshan_output"
DARSHAN_PRELOAD  = "/path/to/libdarshan.so"    # ← update this path
DARSHAN_LOG_DIR  = "/tmp"                       # ← where Darshan writes logs

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
        default=1,
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
    """Compile posix_synthetic_workload.c → posix_synthetic_workload binary."""
    cmd = ["gcc", "-O2", "-o", WORKLOAD_BIN, WORKLOAD_SRC]
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
    """
    after_files = set(os.listdir(DARSHAN_LOG_DIR))
    new_files = [
        f for f in (after_files - before_files)
        if f.endswith(".darshan")
    ]
    if not new_files:
        return None
    new_files_full = [os.path.join(DARSHAN_LOG_DIR, f) for f in new_files]
    return max(new_files_full, key=os.path.getmtime)


def needs_setup(params):
    """
    Returns True only for pure-read profiles (read_ratio == 1.0).
    These need a setup pass to pre-populate files before the measured run.

    Mixed profiles (0 < read_ratio < 1.0) create their own files in workload
    mode — the write phases handle file creation, so no setup is needed.

    metadata_heavy never needs setup — it manages its own files internally.
    """
    return params["read_ratio"] >= 1.0


def build_workload_cmd(name, params, mode):
    """Build the CLI arg list for the posix_synthetic_workload binary."""
    pattern_int = ACCESS_PATTERN_MAP[params["access_pattern"]]
    return [
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
        WORKLOAD_WORK_DIR,
        str(mode),   # 0 = setup, 1 = workload
    ]


# =============================================================================
# RUNNING A SINGLE PROFILE
# =============================================================================

def run_profile(name, params, run_index, modules, output_dir, dry_run):
    """
    Run one profile once and parse the resulting Darshan log.

    For profiles with read_ratio > 0:
      1. Run binary in mode 0 (setup) WITHOUT LD_PRELOAD — writes files only,
         no Darshan instrumentation, no log generated.
      2. Run binary in mode 1 (workload) WITH LD_PRELOAD — measured run,
         reads from pre-existing files, Darshan records only the target I/O.

    For pure-write profiles (read_ratio == 0):
      - Run binary in mode 1 directly under Darshan. No setup needed.

    metadata_heavy:
      - Always run in mode 1 directly. It manages its own files internally.
    """
    label = f"{name}_run{run_index}"
    setup_required = needs_setup(params) and name != "metadata_heavy"

    setup_cmd    = build_workload_cmd(name, params, mode=0)
    workload_cmd = build_workload_cmd(name, params, mode=1)

    module_flags = [f"--{m}" for m in modules]
    parse_cmd = [
        sys.executable, PARSE_SCRIPT,
        "--label", label,
        "--output-dir", output_dir,
    ] + module_flags + ["--log", "<log>"]

    if dry_run:
        if setup_required:
            print(f"  [{label}] Would run setup (no Darshan): {' '.join(setup_cmd)}")
        print(f"  [{label}] Would run workload (Darshan): {' '.join(workload_cmd)}")
        print(f"  [{label}] Would parse: {' '.join(parse_cmd)}")
        return True

    # --- Setup phase (no Darshan) ---
    if setup_required:
        print(f"  [{label}] Setup (no instrumentation): writing files...")
        # Run without LD_PRELOAD so Darshan does not record the setup writes
        clean_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        result = subprocess.run(setup_cmd, env=clean_env)
        if result.returncode != 0:
            print(f"  [{label}] Setup exited with code {result.returncode} — skipping workload.")
            return False

    # --- Workload phase (Darshan attached) ---
    print(f"  [{label}] Workload (instrumented): {' '.join(workload_cmd)}")
    before_files = set(os.listdir(DARSHAN_LOG_DIR))

    env = os.environ.copy()
    if DARSHAN_PRELOAD and os.path.isfile(DARSHAN_PRELOAD):
        env["LD_PRELOAD"] = DARSHAN_PRELOAD
    else:
        print(f"  Warning: DARSHAN_PRELOAD not found ({DARSHAN_PRELOAD}). "
              "Running without instrumentation.")

    result = subprocess.run(workload_cmd, env=env)
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
        os.makedirs(WORKLOAD_WORK_DIR, exist_ok=True)

    print(f"Profiles:         {[n for n, _ in profiles]}")
    print(f"Runs per profile: {args.runs}")
    print(f"Modules:          {modules}")
    print(f"Output dir:       {args.output_dir}")
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
            success = run_profile(name, params, i, modules, args.output_dir, args.dry_run)
            if success:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"Done. {completed}/{total} completed. {failed} failed.")


if __name__ == "__main__":
    main()
