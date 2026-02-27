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
import time
import signal
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR    = os.path.join(SCRIPT_DIR, "workloads")
PROFILES_JSON    = os.path.join(WORKLOADS_DIR, "profiles.json")
WORKLOAD_SRC     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload.c")
WORKLOAD_BIN     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload")
PARSE_SCRIPT     = os.path.join(SCRIPT_DIR, "parse_darshan.py")

# Timeout configuration
WORKLOAD_TIMEOUT = 600  # 10 minutes in seconds
MAX_RETRIES = 2  # Total attempts per profile (original + 1 retry)

# Error logging
ERROR_LOG_FILE = os.path.join(SCRIPT_DIR, "errors.log")

# Storage pool configurations
STORAGE_POOLS = {
    "hdd": {
        "workload_dir": "/mnt/beegfs/advay/hdd/workloads/tmp",
        "output_dir": "./output/hdd"
    },
    "ssd": {
        "workload_dir": "/mnt/beegfs/advay/ssd/workloads/tmp",
        "output_dir": "./output/ssd"
    }
}

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
    "contiguous": 0,  # alias for sequential
    "random":     1,
    "strided":    2,
    "nd_strided": 3,  # multi-dimensional strided (alternates row/column major)
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def log_error(message, profile_name=None, storage_type=None, run_index=None):
    """Log error to both console and error log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the log message
    if profile_name:
        prefix = f"[{profile_name}"
        if storage_type:
            prefix += f"/{storage_type.upper()}"
        if run_index:
            prefix += f"/run{run_index}"
        prefix += "]"
        log_msg = f"{timestamp} {prefix} {message}"
    else:
        log_msg = f"{timestamp} {message}"
    
    # Print to console
    print(f"ERROR: {log_msg}")
    
    # Append to error log file
    try:
        with open(ERROR_LOG_FILE, 'a') as f:
            f.write(f"{log_msg}\n")
    except Exception as e:
        print(f"WARNING: Failed to write to error log: {e}")


def init_error_log():
    """Initialize error log file with header."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*80}\nWorkload Execution Log - Started at {timestamp}\n{'='*80}\n"
    try:
        with open(ERROR_LOG_FILE, 'a') as f:
            f.write(header)
        print(f"Error logging to: {ERROR_LOG_FILE}")
    except Exception as e:
        print(f"WARNING: Failed to initialize error log: {e}")


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
        default=None,
        help=f"Output directory for CSVs (default: auto-determined by storage pool)"
    )
    parser.add_argument(
        "--workload-dir",
        default=None,
        help=f"Directory where workload files are created (default: auto-determined by storage pool)"
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
# PROFILE LOADING AND VARIANT GENERATION
# =============================================================================

def generate_size_variant(base_name, base_params, size_gb):
    """
    Generate a single size variant of a profile by scaling num_ops.
    Returns (variant_name, variant_params).
    """
    variant_params = base_params.copy()
    op_size = variant_params["op_size"]
    
    # Calculate num_ops needed for target size
    target_bytes = size_gb * 1024 * 1024 * 1024
    num_ops = target_bytes // op_size
    
    # Update parameters
    variant_params["num_ops"] = num_ops
    
    # Generate variant name
    variant_name = f"{base_name}_{size_gb}gb"
    
    return (variant_name, variant_params)


def load_profiles(profiles_path, only=None):
    """
    Load and validate profiles from JSON. 
    If a profile has 'file_size_gb' field (list), generate variants for each size.
    Returns list of (name, params) tuples.
    """
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
        
        # Check if this profile has size variants
        if "file_size_gb" in params and isinstance(params["file_size_gb"], list):
            # Generate a variant for each size
            size_variants = params["file_size_gb"]
            for size_gb in size_variants:
                variant_name, variant_params = generate_size_variant(name, params, size_gb)
                profiles.append((variant_name, variant_params))
        else:
            # No variants, use profile as-is
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
# TIMEOUT EXECUTION
# =============================================================================

def cleanup_workload_files(workload_dir, profile_name, dry_run=False):
    """
    Remove all workload files for a given profile to free up cluster space.
    Matches pattern: workload_{profile_name}_f*
    """
    if dry_run:
        print(f"  [cleanup] Would remove files matching: {workload_dir}/workload_{profile_name}_f*")
        return
    
    import glob
    pattern = os.path.join(workload_dir, f"workload_{profile_name}_f*")
    files = glob.glob(pattern)
    
    if not files:
        return
    
    removed_count = 0
    for filepath in files:
        try:
            os.remove(filepath)
            removed_count += 1
        except OSError as e:
            print(f"  [cleanup] Warning: Failed to remove {filepath}: {e}")
    
    if removed_count > 0:
        print(f"  [cleanup] Removed {removed_count} workload file(s) for {profile_name}")


def run_with_timeout(cmd, timeout_seconds, label):
    """
    Run a command with timeout. Kill process if it exceeds timeout.
    
    Returns:
        tuple: (success: bool, timed_out: bool, returncode: int)
    """
    print(f"  [{label}] Executing with {timeout_seconds}s timeout: {' '.join(cmd)}")
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for clean kill
        )
        
        try:
            # Wait for completion with timeout
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            if process.returncode != 0:
                print(f"  [{label}] Process exited with code {process.returncode}")
                if stderr:
                    print(f"  [{label}] STDERR: {stderr.decode()[:500]}")
                return False, False, process.returncode
            
            return True, False, 0
            
        except subprocess.TimeoutExpired:
            # Timeout occurred - kill entire process group
            print(f"  [{label}] TIMEOUT after {timeout_seconds}s - terminating process...")
            
            try:
                # Kill process group (handles child processes too)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)  # Give it time to terminate gracefully
                
                # Force kill if still alive
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5)
                    
            except Exception as e:
                print(f"  [{label}] Warning during cleanup: {e}")
            
            return False, True, -1
            
    except Exception as e:
        print(f"  [{label}] Exception during execution: {e}")
        return False, False, -1


# =============================================================================
# CACHE CLEARING
# =============================================================================

def clear_caches(dry_run):
    """
    Clear all system caches to ensure clean measurements:
    - Drop page cache, dentries, and inodes
    - Sync all filesystem buffers to disk
    - Wait 2 minutes for system to stabilize
    """
    if dry_run:
        print("  [cache] Would sync filesystems")
        print("  [cache] Would drop page cache, dentries, and inodes")
        print("  [cache] Would sleep for 120 seconds")
        return
    
    print("  [cache] Syncing all filesystems to disk...")
    subprocess.run(["sudo", "sync"], check=False)
    
    print("  [cache] Dropping page cache, dentries, and inodes...")
    # echo 3 > /proc/sys/vm/drop_caches clears:
    # 1 = page cache
    # 2 = dentries and inodes
    # 3 = both
    result = subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=False
    )
    
    if result.returncode != 0:
        print("  [cache] WARNING: Failed to drop caches (may need sudo privileges)")
    
    print("  [cache] Waiting 120 seconds for system to stabilize...")
    time.sleep(120)
    print("  [cache] Cache clearing complete.\n")


# =============================================================================
# RUNNING A SINGLE PROFILE
# =============================================================================

def run_profile(name, params, run_index, modules, output_dir, workload_dir, dry_run, storage_type=None):
    """
    Run one profile once and parse the resulting Darshan log with timeout support.

    For profiles with read_ratio >= 1.0 (pure-read):
      1. Run binary in mode 0 (setup) — writes files only,
         no Darshan instrumentation, no log generated.
      2. Run binary in mode 1 (workload) with mpirun -np 1 — measured run,
         reads from pre-existing files, Darshan records only the target I/O.

    For pure-write/mixed profiles (read_ratio < 1.0):
      - Run binary in mode 1 directly with mpirun. No setup needed.

    metadata_heavy:
      - Always run in mode 1 directly. It manages its own files internally.
      
    Returns:
        tuple: (success: bool, timed_out: bool)
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
        return True, False

    # --- Setup phase (no Darshan, no MPI) ---
    if setup_required:
        print(f"  [{label}] Setup (no instrumentation): {' '.join(setup_cmd)}")
        success, timed_out, returncode = run_with_timeout(setup_cmd, WORKLOAD_TIMEOUT, label)
        if timed_out:
            log_error(f"Setup timed out after {WORKLOAD_TIMEOUT}s", name, storage_type, run_index)
            return False, True
        if not success:
            log_error(f"Setup failed with code {returncode}", name, storage_type, run_index)
            return False, False

    # --- Workload phase (MPI + Darshan attached) ---
    before_files = set(os.listdir(DARSHAN_LOG_DIR))
    
    success, timed_out, returncode = run_with_timeout(workload_cmd, WORKLOAD_TIMEOUT, label)
    if timed_out:
        log_error(f"Workload timed out after {WORKLOAD_TIMEOUT}s", name, storage_type, run_index)
        return False, True
    if not success:
        log_error(f"Workload failed with code {returncode}", name, storage_type, run_index)
        return False, False

    # --- Locate and parse the Darshan log ---
    log_path = find_latest_darshan_log(before_files)
    if not log_path:
        log_error(f"No Darshan log found in {DARSHAN_LOG_DIR}", name, storage_type, run_index)
        return False, False

    print(f"  [{label}] Log: {log_path}")
    parse_cmd[-1] = log_path
    parse_result = subprocess.run(parse_cmd)
    if parse_result.returncode != 0:
        log_error(f"Parser failed with code {parse_result.returncode}", name, storage_type, run_index)
        return False, False

    # Cleanup: Remove Darshan log after successful parsing to save space
    if not dry_run:
        try:
            os.remove(log_path)
        except OSError as e:
            print(f"  [{label}] Warning: Failed to remove Darshan log {log_path}: {e}")

    return True, False


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    modules = args.modules if args.modules else DEFAULT_MODULES

    # Initialize error logging
    init_error_log()

    # Compile
    if not compile_workload(args.dry_run):
        sys.exit(1)

    # Load profiles (may include auto-generated size variants)
    profiles = load_profiles(args.profiles, only=args.only)
    if not profiles:
        print("No profiles to run.")
        sys.exit(0)

    print(f"Total profiles (including variants): {len(profiles)}")
    print(f"Profiles: {[n for n, _ in profiles]}")
    print(f"Runs per profile: {args.runs}")
    print(f"Modules:          {modules}")
    print(f"Storage pools:    {list(STORAGE_POOLS.keys())}")
    print(f"Workload timeout: {WORKLOAD_TIMEOUT}s ({WORKLOAD_TIMEOUT//60} minutes)")
    print(f"Max retries:      {MAX_RETRIES} attempts per run")
    print(f"DARSHAN_LOG_DIR = {DARSHAN_LOG_DIR}\n")
    if args.dry_run:
        print("Dry run — nothing will be executed.\n")

    total_runs = len(profiles) * args.runs * len(STORAGE_POOLS)
    completed = 0
    failed = 0
    skipped = 0
    
    # Track consecutive timeout failures per profile
    profile_timeout_failures = {}

    # NEW EXECUTION ORDER:
    # For each profile variant:
    #   - Run 1-5 on HDD
    #   - Run 1-5 on SSD
    # Then move to next profile variant
    
    for name, params in profiles:
        print(f"\n{'='*70}")
        print(f"Profile: {name}  ({args.runs} run(s) × {len(STORAGE_POOLS)} storage pools)")
        print(f"{'='*70}")
        
        # Check if this profile should be skipped due to previous timeout failures
        profile_key = f"{name}"
        if profile_key in profile_timeout_failures and profile_timeout_failures[profile_key] >= MAX_RETRIES:
            skip_msg = f"Skipping profile '{name}' - exceeded {MAX_RETRIES} consecutive timeout failures"
            print(f"\n⚠️  {skip_msg}")
            log_error(skip_msg, name)
            skipped_count = args.runs * len(STORAGE_POOLS)
            skipped += skipped_count
            continue
        
        for storage_type in ["hdd", "ssd"]:
            storage_config = STORAGE_POOLS[storage_type]
            output_dir = args.output_dir if args.output_dir else storage_config["output_dir"]
            workload_dir = args.workload_dir if args.workload_dir else storage_config["workload_dir"]
            
            # Ensure directories exist
            os.makedirs(output_dir, exist_ok=True)
            if not args.dry_run:
                os.makedirs(workload_dir, exist_ok=True)
            
            print(f"\n{'*'*70}")
            print(f"Storage: {storage_type.upper()}")
            print(f"Output:  {output_dir}")
            print(f"Workload dir: {workload_dir}")
            print(f"{'*'*70}")
            
            for run_index in range(1, args.runs + 1):
                print(f"\n{'='*50}")
                print(f"{name}  [{storage_type.upper()}]  (run {run_index}/{args.runs})")
                print(f"{'='*50}")
                
                # Clear caches before each run (except the very first run)
                if not (completed == 0):
                    print(f"\n  Clearing caches before {name} [{storage_type}] run {run_index}...")
                    clear_caches(args.dry_run)
                
                # Try up to MAX_RETRIES times
                attempt = 0
                success = False
                timed_out = False
                
                while attempt < MAX_RETRIES and not success:
                    attempt += 1
                    if attempt > 1:
                        print(f"\n  🔄 Retry attempt {attempt}/{MAX_RETRIES} for {name} [{storage_type}] run {run_index}")
                        log_error(f"Retry attempt {attempt}/{MAX_RETRIES}", name, storage_type, run_index)
                    
                    success, timed_out = run_profile(
                        name, params, run_index, modules, 
                        output_dir, workload_dir, args.dry_run, storage_type
                    )
                    
                    if success:
                        completed += 1
                        # Reset timeout failure counter on success
                        if profile_key in profile_timeout_failures:
                            profile_timeout_failures[profile_key] = 0
                        break
                    elif timed_out:
                        # Track consecutive timeout failures
                        profile_timeout_failures[profile_key] = profile_timeout_failures.get(profile_key, 0) + 1
                        
                        if profile_timeout_failures[profile_key] >= MAX_RETRIES:
                            skip_msg = f"Profile '{name}' exceeded {MAX_RETRIES} consecutive timeout failures - will skip remaining runs"
                            print(f"\n⚠️  {skip_msg}")
                            log_error(skip_msg, name)
                            break
                    else:
                        # Non-timeout failure - don't increment timeout counter
                        if attempt >= MAX_RETRIES:
                            failed += 1
                            log_error(f"Failed after {MAX_RETRIES} attempts", name, storage_type, run_index)
                
                # If all retries exhausted
                if not success:
                    if timed_out and profile_timeout_failures.get(profile_key, 0) >= MAX_RETRIES:
                        # Profile will be skipped - count as skipped
                        skipped += 1
                    else:
                        failed += 1
            
            # Cleanup: Remove all workload files for this profile/storage after all runs complete
            print(f"\n  Cleaning up {name} [{storage_type}] workload files...")
            cleanup_workload_files(workload_dir, name, args.dry_run)

    print(f"\n{'='*70}")
    print(f"ALL RUNS COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {completed}/{total_runs}")
    print(f"Failed:    {failed}")
    print(f"Skipped:   {skipped}")
    print(f"Error log: {ERROR_LOG_FILE}")


if __name__ == "__main__":
    main()
