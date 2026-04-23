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
import glob
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.dirname(SCRIPT_DIR)
WORKLOADS_DIR    = os.path.join(SCRIPT_DIR, "workloads")
PROFILES_JSON    = os.path.join(WORKLOADS_DIR, "profiles.json")
WORKLOAD_SRC     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload.c")
WORKLOAD_BIN     = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload_IOR.py")
PARSE_SCRIPT     = os.path.join(SCRIPT_DIR, "parse_darshan.py")

# Timeout configuration
WORKLOAD_TIMEOUT = 600  # 10 minutes in seconds
MAX_RETRIES = 2  # Total attempts per profile (original + 1 retry)

# Error logging and checkpoint tracking
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs_and_checkpoints")
ERROR_LOG_FILE  = os.path.join(LOGS_DIR, "errors.log")
CHECKPOINT_FILE = os.path.join(LOGS_DIR, "checkpoint.json")
OST_LOG_FILE    = os.path.join(LOGS_DIR, "ost_space_and_usage_test.log")

# Storage pool configurations
STORAGE_POOLS = {
    "hdd": {
        "workload_dir": "/mnt/beegfs/advay/hdd/workloads/tmp",
        "output_dir": os.path.join(PROJECT_ROOT, "output", "hdd")
    },
    "ssd": {
        "workload_dir": "/mnt/beegfs/advay/ssd/workloads/tmp",
        "output_dir": os.path.join(PROJECT_ROOT, "output", "ssd")
    }
}

DARSHAN_PRELOAD  = "/usr/local/lib/libdarshan.so.0.0.0"
DARSHAN_LOG_DIR  = "/mnt/nfs_shared/darshan-logs"
# MPI configuration
MPICC            = "mpicc"
MPIRUN           = "mpirun"

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
    
    print(f"ERROR: {log_msg}")
    
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


def load_checkpoint():
    """Load checkpoint data from file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return {}
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Failed to load checkpoint: {e}")
        return {}


def save_checkpoint(checkpoint_data):
    """Save checkpoint data to file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {e}")


def is_run_completed(checkpoint, profile_name, storage_type, run_index):
    """Check if a specific run has been completed."""
    key = f"{profile_name}_{storage_type}_run{run_index}"
    return checkpoint.get(key, False)


def mark_run_completed(checkpoint, profile_name, storage_type, run_index):
    """Mark a specific run as completed in checkpoint."""
    key = f"{profile_name}_{storage_type}_run{run_index}"
    checkpoint[key] = True
    checkpoint["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_checkpoint(checkpoint)


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
        "--storage-type",
        choices=["hdd", "ssd"],
        default=None,
        help="Run only specific storage type (hdd or ssd). If not specified, runs both."
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint - skip already completed runs"
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear checkpoint file and start fresh"
    )
    return parser.parse_args()


# =============================================================================
# COMPILATION
# =============================================================================

def compile_workload(dry_run):
    """No compilation needed — wrapper is pure Python. Verify IOR is available."""
    if dry_run:
        print("Dry run: skipping IOR version check")
        return True

    import shutil
    ior_path = shutil.which("ior")
    if ior_path is None:
        print("ERROR: IOR not found on PATH. Install IOR before running workloads.")
        return False
    print(f"IOR found: {ior_path}\n")

    posix_bin = os.path.join(WORKLOADS_DIR, "posix_synthetic_workload")
    if not os.path.isfile(posix_bin):
        print("WARNING: posix_synthetic_workload binary not found — nd_strided profiles will fail.")
        print(f"  Compile it with: mpicc -O2 -o {posix_bin} {WORKLOAD_SRC} -ldarshan -lpthread -lrt -lz")

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
    
    target_bytes = size_gb * 1024 * 1024 * 1024
    num_ops = int(target_bytes // op_size)
    
    variant_params["num_ops"] = num_ops
    
    mb = size_gb * 1024
    if mb < 1024:
        size_label = f"{int(mb)}mb"
    elif size_gb == int(size_gb):
        size_label = f"{int(size_gb)}gb"
    else:
        size_label = f"{size_gb}gb"
    variant_name = f"{base_name}_{size_label}"
    
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
        
        if params["access_pattern"] not in ACCESS_PATTERN_MAP:
            print(f"Warning: unknown access_pattern '{params['access_pattern']}' "
                  f"in profile '{name}' — skipping.")
            continue
        
        if "file_size_gb" in params and isinstance(params["file_size_gb"], list):
            size_variants = params["file_size_gb"]
            for size_gb in size_variants:
                variant_name, variant_params = generate_size_variant(name, params, size_gb)
                profiles.append((variant_name, variant_params))
        else:
            profiles.append((name, params))

    if only:
        found = {n for n, _ in profiles}
        # Check against base names (before size variant suffix)
        base_names_found = {"_".join(n.split("_")[:-1]) for n in found} | found
        for requested in only:
            if requested not in base_names_found:
                print(f"Warning: profile '{requested}' not found in {profiles_path}")

    return profiles


# =============================================================================
# DARSHAN LOG DETECTION
# =============================================================================

def find_latest_darshan_log(before_time):
    """
    Find the newest .darshan log written to DARSHAN_LOG_DIR after before_time.

    before_time is a float timestamp (from time.time()) recorded just before
    the workload run started. Only logs with mtime > before_time are considered,
    ensuring we never pick up a stale log from a previous run.

    Darshan organizes logs by date: DARSHAN_LOG_DIR/YYYY/M/D/*.darshan
    """
    # Wait briefly for filesystem to sync before scanning
    time.sleep(1.0)
    
    all_logs = glob.glob(f"{DARSHAN_LOG_DIR}/**/*.darshan", recursive=True)
    
    if not all_logs:
        return None
    
    # Filter to only logs created after the snapshot timestamp
    new_logs = [l for l in all_logs if os.path.getmtime(l) > before_time]
    
    if not new_logs:
        print(f"  WARNING: No new Darshan logs found after snapshot "
              f"(checked {len(all_logs)} existing logs)")
        return None
    
    return max(new_logs, key=os.path.getmtime)


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
    """Build the CLI arg list for the IOR wrapper.
    
    For mode=0 (setup): run directly without MPI (no Darshan needed)
    For mode=1 (workload): run with mpirun -np 1 (Darshan requires MPI)
    """
    pattern_int = ACCESS_PATTERN_MAP[params["access_pattern"]]
    base_args = [
        "python3", WORKLOAD_BIN,
        name,
        str(params["read_ratio"]),
        str(pattern_int),
        str(params["stride_size"]),
        str(params["op_size"]),
        str(params["num_ops"]),
        str(params["num_files"]),
        str(params["num_phases"]),
        str(params["fsync_interval"]),
        workload_dir,
        str(mode),   # 0 = setup, 1 = workload
    ]
    
    # Workload mode (mode=1): wrap with mpirun for Darshan
    if mode == 1:
        return [MPIRUN, "-np", "1"] + base_args
    else:
        # Setup mode (mode=0): run directly, no MPI, no Darshan
        return base_args


# =============================================================================
# TIMEOUT EXECUTION
# =============================================================================

def cleanup_workload_files(workload_dir, profile_name, dry_run=False):
    """
    Remove all workload files in the workload_dir to free up cluster space.
    """
    if dry_run:
        print(f"  [cleanup] Would remove all files in: {workload_dir}")
        return

    files = glob.glob(os.path.join(workload_dir, "*"))

    if not files:
        return

    removed_count = 0
    for filepath in files:
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                removed_count += 1
            except OSError as e:
                print(f"  [cleanup] Warning: Failed to remove {filepath}: {e}")

    if removed_count > 0:
        print(f"  [cleanup] Removed {removed_count} workload file(s) from {workload_dir}")


def run_with_timeout(cmd, timeout_seconds, label, env=None):
    """
    Run a command with timeout. Kill process if it exceeds timeout.
    env: optional environment dict — if None, inherits parent environment.
    
    Returns:
        tuple: (success: bool, timed_out: bool, returncode: int)
    """
    print(f"  [{label}] Executing with {timeout_seconds}s timeout: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group for clean kill
            env=env                 # None = inherit parent env
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            if process.returncode != 0:
                print(f"  [{label}] Process exited with code {process.returncode}")
                if stderr:
                    print(f"  [{label}] STDERR: {stderr.decode()[:500]}")
                return False, False, process.returncode
            
            return True, False, 0
            
        except subprocess.TimeoutExpired:
            print(f"  [{label}] TIMEOUT after {timeout_seconds}s - terminating process...")
            
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)
                
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
# OST SPACE AND USAGE LOGGING
# =============================================================================

def _write_ost_log(lines):
    """Append lines to the OST space and usage log file."""
    try:
        with open(OST_LOG_FILE, 'a') as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"  [ost] WARNING: Failed to write OST log: {e}")


def log_ost_space(label, dry_run):
    if dry_run:
        print(f"  [ost] Would log OST space: {label}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Storage targets ---
    storage_header = [
        f"",
        f"### OST SPACE (STORAGE) {label}",
        f"### Timestamp: {timestamp}",
    ]
    result = subprocess.run(
        ["sudo", "beegfs-ctl", "--listtargets", "--nodetype=storage", "--spaceinfo"],
        capture_output=True, text=True
    )
    lines = storage_header + result.stdout.strip().splitlines()
    if result.returncode != 0 and result.stderr.strip():
        lines += [f"# WARNINGS (offline nodes, not in active pools): {result.stderr.strip()[:200]}"]
    _write_ost_log(lines)

    # --- Metadata targets ---
    meta_header = [
        f"",
        f"### MDS SPACE (METADATA) {label}",
        f"### Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    meta_result = subprocess.run(
        ["sudo", "beegfs-ctl", "--listtargets", "--nodetype=meta", "--spaceinfo"],
        capture_output=True, text=True
    )
    meta_lines = meta_header + meta_result.stdout.strip().splitlines()
    if meta_result.returncode != 0 and meta_result.stderr.strip():
        meta_lines += [f"# WARNINGS: {meta_result.stderr.strip()[:200]}"]
    _write_ost_log(meta_lines)

    print(f"  [ost] Logged OST + MDS space: {label}")


def log_file_ost_layout(workload_dir, profile_name, label, dry_run):
    """
    For every workload file currently on disk, query beegfs-ctl --getentryinfo
    to find which OST targets its chunks are striped across.
    Call this AFTER the workload run, BEFORE cleanup deletes the files.
    label: human-readable tag e.g. "AFTER small_random_read_heavy_freq_102mb [SSD] run1"
    """
    if dry_run:
        print(f"  [ost] Would log per-file OST layout: {label}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"",
        f"### FILE OST LAYOUT {label}",
        f"### Timestamp: {timestamp}",
    ]
    _write_ost_log(header)

    pattern = os.path.join(workload_dir, f"workload_{profile_name}_f*")
    files   = glob.glob(pattern)

    if not files:
        _write_ost_log([f"  (no files found matching {pattern})"])
        print(f"  [ost] No files found for layout query: {profile_name}")
        return

    for filepath in sorted(files):
        result = subprocess.run(
            ["sudo", "beegfs-ctl", "--getentryinfo", "--verbose", filepath],
            capture_output=True, text=True
        )
        file_lines = [f"  FILE: {filepath}"]
        if result.returncode != 0:
            file_lines.append(f"    ERROR: {result.stderr.strip()}")
        else:
            # Indent each line for readability
            file_lines += ["    " + l for l in result.stdout.strip().splitlines()]
        _write_ost_log(file_lines)

    print(f"  [ost] Logged per-file OST layout: {label}")

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
    result = subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=False
    )
    
    if result.returncode != 0:
        print("  [cache] WARNING: Failed to drop caches (may need sudo privileges)")
    
    print("  [cache] Waiting 120 seconds for system to stabilize...")
    #time.sleep(120)  # Uncomment for real data collection runs
    print("  [cache] Cache clearing complete.\n")


# =============================================================================
# RUNNING A SINGLE PROFILE
# =============================================================================

def run_profile(name, params, run_index, modules, output_dir, workload_dir, dry_run, storage_type=None):
    """
    Run one profile once and parse the resulting Darshan log with timeout support.

    For profiles with read_ratio >= 1.0 (pure-read):
      1. Run wrapper in mode 0 (setup) — writes files only,
         no Darshan instrumentation, no log generated.
      2. Run wrapper in mode 1 (workload) with mpirun -np 1 — measured run,
         reads from pre-existing files, Darshan records only the target I/O.

    For pure-write/mixed profiles (read_ratio < 1.0):
      - Run wrapper in mode 1 directly with mpirun. No setup needed.

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
        # Explicitly strip LD_PRELOAD so Darshan cannot attach to setup
        setup_env = os.environ.copy()
        setup_env.pop("LD_PRELOAD", None)
        success, timed_out, returncode = run_with_timeout(
            setup_cmd, WORKLOAD_TIMEOUT, label, env=setup_env
        )
        if timed_out:
            log_error(f"Setup timed out after {WORKLOAD_TIMEOUT}s", name, storage_type, run_index)
            return False, True
        if not success:
            log_error(f"Setup failed with code {returncode}", name, storage_type, run_index)
            return False, False

    # --- Workload phase (MPI + Darshan attached) ---
    # Record timestamp BEFORE starting workload so we can filter Darshan logs
    # to only those created by this specific run, not stale logs from previous runs.
    before_time = time.time()

    # Inject LD_PRELOAD so Darshan attaches to the IOR subprocess
    workload_env = os.environ.copy()
    workload_env["LD_PRELOAD"] = DARSHAN_PRELOAD
    print(f"  [{label}] LD_PRELOAD={DARSHAN_PRELOAD}")

    success, timed_out, returncode = run_with_timeout(
        workload_cmd, WORKLOAD_TIMEOUT, label, env=workload_env
    )
    if timed_out:
        log_error(f"Workload timed out after {WORKLOAD_TIMEOUT}s", name, storage_type, run_index)
        return False, True
    if not success:
        log_error(f"Workload failed with code {returncode}", name, storage_type, run_index)
        return False, False

    # --- Locate and parse the Darshan log ---
    # Pass before_time so only logs newer than this run are considered
    log_path = find_latest_darshan_log(before_time)
    if not log_path:
        log_error(f"No Darshan log found in {DARSHAN_LOG_DIR} after run started", name, storage_type, run_index)
        return False, False

    print(f"  [{label}] Log: {log_path}")
    parse_cmd[-1] = log_path
    parse_result = subprocess.run(parse_cmd)
    if parse_result.returncode != 0:
        log_error(f"Parser failed with code {parse_result.returncode}", name, storage_type, run_index)
        return False, False

    # Log per-file OST layout AFTER successful run, BEFORE files are deleted
    _st = (storage_type or "unknown").upper()
    ost_label_after = f"AFTER {name} [{_st}] run{run_index}"
    log_file_ost_layout(workload_dir, name, ost_label_after, dry_run)
    log_ost_space(ost_label_after, dry_run)

    # Cleanup: Remove Darshan log after successful parsing to save space
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

    # Handle checkpoint reset
    if args.reset_checkpoint:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(f"✓ Checkpoint cleared: {CHECKPOINT_FILE}")
        else:
            print("No checkpoint file found.")
        return

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {}
    if args.resume:
        completed_runs = len([k for k in checkpoint.keys() if k != "last_updated"])
        print(f"📍 Resume mode: Found {completed_runs} completed runs in checkpoint")
        if "last_updated" in checkpoint:
            print(f"   Last updated: {checkpoint['last_updated']}")
        print(f"   Checkpoint file: {CHECKPOINT_FILE}")

    # Initialize error logging
    init_error_log()

    # Check IOR available and nd_strided binary exists
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
    
    profile_timeout_failures = {}

    for name, params in profiles:
        print(f"\n{'='*70}")
        print(f"Profile: {name}  ({args.runs} run(s) × {len(STORAGE_POOLS)} storage pools)")
        print(f"{'='*70}")
        
        profile_key = f"{name}"
        if profile_key in profile_timeout_failures and profile_timeout_failures[profile_key] >= MAX_RETRIES:
            skip_msg = f"Skipping profile '{name}' - exceeded {MAX_RETRIES} consecutive timeout failures"
            print(f"\n⚠️  {skip_msg}")
            log_error(skip_msg, name)
            skipped_count = args.runs * len(STORAGE_POOLS)
            skipped += skipped_count
            continue
        
        storage_types = [args.storage_type] if args.storage_type else ["hdd", "ssd"]
        
        for storage_type in storage_types:
            storage_config = STORAGE_POOLS[storage_type]
            output_dir = args.output_dir if args.output_dir else storage_config["output_dir"]
            workload_dir = args.workload_dir if args.workload_dir else storage_config["workload_dir"]
            
            os.makedirs(output_dir, exist_ok=True)
            if not args.dry_run:
                os.makedirs(workload_dir, exist_ok=True)
            
            # Clean up any stale files from previous crashed runs
            print(f"  Cleaning up any stale {name} [{storage_type}] files before starting...")
            cleanup_workload_files(workload_dir, name, args.dry_run)
            
            print(f"\n{'*'*70}")
            print(f"Storage: {storage_type.upper()}")
            print(f"Output:  {output_dir}")
            print(f"Workload dir: {workload_dir}")
            print(f"{'*'*70}")
            
            for run_index in range(1, args.runs + 1):
                if is_run_completed(checkpoint, name, storage_type, run_index):
                    print(f"\n{'='*50}")
                    print(f"{name}  [{storage_type.upper()}]  (run {run_index}/{args.runs})")
                    print(f"{'='*50}")
                    print(f"  ⏭️  Skipping - already completed (resume mode)")
                    completed += 1
                    skipped += 1
                    continue
                
                print(f"\n{'='*50}")
                print(f"{name}  [{storage_type.upper()}]  (run {run_index}/{args.runs})")
                print(f"{'='*50}")
                
                # Clear caches before every run for consistent cold-start measurements
                print(f"\n  Clearing caches before {name} [{storage_type}] run {run_index}...")
                clear_caches(args.dry_run)

                # Log OST space AFTER cache clear, BEFORE the workload runs
                ost_label_before = f"BEFORE {name} [{storage_type.upper()}] run{run_index}"
                log_ost_space(ost_label_before, args.dry_run)
                
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
                        mark_run_completed(checkpoint, name, storage_type, run_index)
                        if profile_key in profile_timeout_failures:
                            profile_timeout_failures[profile_key] = 0
                        break
                    elif timed_out:
                        profile_timeout_failures[profile_key] = profile_timeout_failures.get(profile_key, 0) + 1
                        
                        if profile_timeout_failures[profile_key] >= MAX_RETRIES:
                            skip_msg = f"Profile '{name}' exceeded {MAX_RETRIES} consecutive timeout failures - will skip remaining runs"
                            print(f"\n⚠️  {skip_msg}")
                            log_error(skip_msg, name)
                            break
                    else:
                        if attempt >= MAX_RETRIES:
                            failed += 1
                            log_error(f"Failed after {MAX_RETRIES} attempts", name, storage_type, run_index)
                
                if not success:
                    if timed_out and profile_timeout_failures.get(profile_key, 0) >= MAX_RETRIES:
                        skipped += 1
                    else:
                        failed += 1

                print(f"\n  Cleaning up {name} [{storage_type}] workload files...")
                cleanup_workload_files(workload_dir, name, args.dry_run)
        
            

    print(f"\n{'='*70}")
    print(f"ALL RUNS COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {completed}/{total_runs}")
    print(f"Failed:    {failed}")
    print(f"Skipped:   {skipped}")
    print(f"Error log: {ERROR_LOG_FILE}")
    if args.resume or checkpoint:
        print(f"Checkpoint: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()