#!/usr/bin/env python3
"""
ior_workload_wrapper.py

Drop-in replacement for posix_synthetic_workload.c.
Translates the same 11 CLI arguments into IOR commands for POSIX workloads.
nd_strided and strided profiles are delegated to the original compiled C binary
by run_workloads.py — this wrapper never receives them.

Usage (identical interface to posix_synthetic_workload):
    python3 ior_workload_wrapper.py <profile_name> <read_ratio> <access_pattern>
                                    <stride_size> <op_size> <num_ops> <num_files>
                                    <num_phases> <fsync_interval> <work_dir> <mode>

access_pattern integers:
    0 = sequential
    1 = random
    2 = strided    → handled by run_workloads.py, delegated to C binary directly
    3 = nd_strided → handled by run_workloads.py, delegated to C binary directly

mode:
    0 = SETUP    — write files only, no Darshan. Pure-read profiles only.
    1 = WORKLOAD — measured run, Darshan attached via LD_PRELOAD by caller.

IOR flags used:
    -a POSIX          POSIX API (matches original O_DIRECT syscall behavior)
    -b <bytes>        block size  = op_size * ops_per_file  (total per-file data)
    -t <bytes>        transfer size = op_size               (per-op size)
    -s 1              one segment (all data in one contiguous block per file)
    -w / -r           write / read
    -k                keep files after write (setup mode, or when read follows)
    -z                random offsets (access_pattern == random)
    -i 1              one iteration (phases handled by multiple IOR invocations)
    -v                verbose output to stderr (useful for debugging)
    --posix.odirect   bypass page cache on writes (matches O_DIRECT in the C binary)
    -e                fsync on close (used when fsync_interval > 0)

Random geometry:
    IOR randomizes offsets *within* a block, so blocksize must be > transfersize.
    We use a single block sized to next_power_of_2(total_bytes + op_size) with
    -s 1. The power-of-2 rounding means the actual file is slightly larger than
    requested, but is the only way to satisfy IOR's -z constraint cleanly.

    O_DIRECT is intentionally skipped on random reads — random offsets are not
    guaranteed to be 4096-aligned internally in IOR, which causes MPI_Abort.
    Cache is cleared before each run so cold-storage measurements remain valid.

nd_strided / strided delegation:
    run_workloads.py detects strided and nd_strided profiles and calls the C
    binary directly via mpirun, bypassing this wrapper entirely. This avoids
    the PMI_Init failure that occurs when the C binary is launched as a
    subprocess of a Python process that is itself an MPI rank.
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# IOR binary — must be on PATH or set absolute path here
IOR_BIN = "ior"

# Access pattern constants (mirrors the C binary)
PATTERN_SEQUENTIAL = 0
PATTERN_RANDOM     = 1
PATTERN_STRIDED    = 2
PATTERN_ND_STRIDED = 3

# Mode constants
MODE_SETUP    = 0
MODE_WORKLOAD = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv):
    if len(argv) < 12:
        print(
            f"Usage: {argv[0]} <profile_name> <read_ratio> <access_pattern (0|1|2|3)>\n"
            f"           <stride_size> <op_size> <num_ops> <num_files>\n"
            f"           <num_phases> <fsync_interval> <work_dir> <mode (0|1)>",
            file=sys.stderr
        )
        sys.exit(1)

    return {
        "profile_name":   argv[1],
        "read_ratio":     float(argv[2]),
        "access_pattern": int(argv[3]),
        "stride_size":    int(argv[4]),
        "op_size":        int(argv[5]),
        "num_ops":        int(argv[6]),
        "num_files":      int(argv[7]),
        "num_phases":     int(argv[8]),
        "fsync_interval": int(argv[9]),
        "work_dir":       argv[10],
        "mode":           int(argv[11]),
    }


# ---------------------------------------------------------------------------
# IOR command construction
# ---------------------------------------------------------------------------

def ops_per_file(p):
    """
    Distribute num_ops evenly across files.
    Returns (base_ops, last_file_ops) — last file absorbs integer remainder,
    matching the C binary's distribution logic exactly.
    """
    base = p["num_ops"] // p["num_files"]
    last = p["num_ops"] - base * (p["num_files"] - 1)
    return base, last


def file_path(p, file_index):
    """
    Deterministic file path — identical scheme to the C binary so setup and
    workload mode find the same files:
        {work_dir}/workload_{profile_name}_f{file_index}
    """
    return os.path.join(
        p["work_dir"],
        f"workload_{p['profile_name']}_f{file_index}"
    )


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def build_ior_base_flags(p, n_ops, is_write, is_read, keep_files):
    """
    Build the flags common to all IOR invocations for this profile.

    IOR geometry depends on access pattern:

    Sequential:
      -t = op_size
      -b = op_size   (one transfer per block)
      -s = n_ops     (number of blocks = number of ops)

    Random (-z):
      IOR randomizes offsets *within* a block, so blocksize must be > transfersize.
      We use:
        -t = op_size
        -b = next_power_of_2(n_ops * op_size + op_size)  (whole file as one block)
        -s = 1
    """
    transfer = p["op_size"]

    if p["access_pattern"] == PATTERN_RANDOM:
        total = n_ops * p["op_size"]
        block = next_power_of_2(total + p["op_size"])  # +op_size ensures block > transfer
        segs  = 1
    else:
        # Sequential
        block = p["op_size"]
        segs  = n_ops

    flags = [
        IOR_BIN,
        "-a", "POSIX",
        "-b", str(block),
        "-t", str(transfer),
        "-s", str(segs),
        "-i", "1",   # one iteration per IOR call — phases handled by caller loop
        "-v",        # verbose output to stderr
    ]

    # O_DIRECT only on writes — random read offsets may not be 4096-aligned
    # inside IOR, causing MPI_Abort. Caches are cleared before each run so
    # omitting O_DIRECT on reads still gives valid cold-storage measurements.
    if is_write:
        flags.append("--posix.odirect")

    if is_write:
        flags.append("-w")
    if is_read:
        flags.append("-r")
    if keep_files:
        flags.append("-k")

    # fsync on close when fsync_interval is set
    if p["fsync_interval"] > 0:
        flags.append("-e")

    # Random access — randomize offsets within the single large block
    if p["access_pattern"] == PATTERN_RANDOM:
        flags.append("-z")

    return flags


def run_ior(flags, filepath, label="", use_mpi=True, env=None):
    """
    Execute one IOR invocation. Filepath is passed via -o.

    use_mpi=True  -- wrap with mpirun -np 1 (workload mode, Darshan attaches via
                     MPI_Init/Finalize triggered by mpirun)
    use_mpi=False -- run IOR directly (setup mode, Darshan must NOT attach;
                     LD_PRELOAD is stripped by the caller before this call)
    """
    if use_mpi:
        cmd = ["mpirun", "-np", "1"] + flags + ["-o", filepath]
    else:
        cmd = flags + ["-o", filepath]
    tag = f"[{label}] " if label else ""
    print(f"{tag}IOR: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"{tag}IOR exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Phase planning
# ---------------------------------------------------------------------------

def plan_phases(p):
    """
    Reproduce the C binary's phase ordering logic:
      - Pure read  (workload mode): all phases are reads
      - Pure write:                 all phases are writes
      - Mixed:                      even phases = write, odd phases = read

    Returns a list of "W" / "R" strings of length num_phases.
    """
    if p["mode"] == MODE_WORKLOAD and p["read_ratio"] >= 1.0:
        return ["R"] * p["num_phases"]

    phases = []
    for ph in range(p["num_phases"]):
        phases.append("W" if ph % 2 == 0 else "R")
    return phases


def ops_for_phase(phase_type, total_write_ops, total_read_ops,
                  write_phases, read_phases, write_count, read_count):
    """
    Distribute total ops evenly across phases of the same type,
    with the last phase of each type absorbing the remainder —
    identical to the C binary's distribution.
    """
    if phase_type == "W":
        per = total_write_ops // write_phases if write_phases else 0
        if write_count == write_phases - 1:
            return total_write_ops - per * (write_phases - 1)
        return per
    else:
        per = total_read_ops // read_phases if read_phases else 0
        if read_count == read_phases - 1:
            return total_read_ops - per * (read_phases - 1)
        return per


# ---------------------------------------------------------------------------
# Setup mode
# ---------------------------------------------------------------------------

def run_setup(p):
    """
    Write files to disk without Darshan (caller must not set LD_PRELOAD).
    Pure-read profiles need files pre-populated before the measured run.

    Setup always writes sequentially regardless of access pattern, matching
    the C binary's setup behavior. For random profiles, the file must be
    written with the same power-of-2 block geometry used during the workload
    phase, so IOR's -z random reads land within bounds.
    """
    os.makedirs(p["work_dir"], exist_ok=True)

    # Strip LD_PRELOAD so Darshan cannot attach to the setup IOR process
    clean_env = os.environ.copy()
    clean_env.pop("LD_PRELOAD", None)

    setup_write_ops = p["num_ops"]
    ops_per_f       = setup_write_ops // p["num_files"]
    last_file_ops   = setup_write_ops - ops_per_f * (p["num_files"] - 1)

    for f in range(p["num_files"]):
        n_ops = last_file_ops if f == p["num_files"] - 1 else ops_per_f
        fp    = file_path(p, f)

        # Match the workload geometry so the file is the right size:
        # random profiles use a single power-of-2 block; others use n_ops blocks.
        if p["access_pattern"] == PATTERN_RANDOM:
            total       = n_ops * p["op_size"]
            setup_block = next_power_of_2(total + p["op_size"])
            setup_segs  = 1
        else:
            setup_block = p["op_size"]
            setup_segs  = n_ops

        flags = [
            IOR_BIN,
            "-a", "POSIX",
            "-b", str(setup_block),
            "-t", str(p["op_size"]),
            "-s", str(setup_segs),
            "--posix.odirect",
            "-i", "1",
            "-w",
            "-k",   # keep file for workload mode
        ]
        run_ior(flags, fp, label=f"setup f{f}", use_mpi=False, env=clean_env)

        if not os.path.exists(fp):
            print(f"[setup f{f}] WARNING: expected file not found at {fp}",
                  file=sys.stderr)


# ---------------------------------------------------------------------------
# Workload mode
# ---------------------------------------------------------------------------

def run_workload(p):
    """
    Measured run — Darshan is attached by the caller via LD_PRELOAD.
    Iterates over phases, issuing one IOR call per phase per file.
    Cleans up files on exit (matching C binary workload mode behavior).
    """
    os.makedirs(p["work_dir"], exist_ok=True)

    total_read_ops  = int(p["num_ops"] * p["read_ratio"])
    total_write_ops = p["num_ops"] - total_read_ops

    base_ops, last_ops = ops_per_file(p)

    phases       = plan_phases(p)
    write_phases = phases.count("W")
    read_phases  = phases.count("R")

    for f in range(p["num_files"]):
        n_ops = last_ops if f == p["num_files"] - 1 else base_ops
        fp    = file_path(p, f)

        f_read_ops  = int(n_ops * p["read_ratio"])
        f_write_ops = n_ops - f_read_ops

        write_count            = 0
        read_count             = 0
        cumulative_written_ops = 0

        # Pure-read: verify setup file exists, treat all ops as already written
        if p["read_ratio"] >= 1.0:
            if not os.path.exists(fp):
                print(f"ERROR: setup file not found: {fp} — was setup mode run first?",
                      file=sys.stderr)
                sys.exit(1)
            cumulative_written_ops = n_ops

        for ph_idx, phase_type in enumerate(phases):

            if phase_type == "W" and write_phases > 0:
                ph_ops = ops_for_phase(
                    "W", f_write_ops, f_read_ops,
                    write_phases, read_phases,
                    write_count, read_count
                )
                if ph_ops <= 0:
                    write_count += 1
                    continue

                keep  = True #changed to true from read_phases > 0
                flags = build_ior_base_flags(p, ph_ops, is_write=True,
                                             is_read=False, keep_files=keep)
                run_ior(flags, fp, label=f"{p['profile_name']} f{f} ph{ph_idx}(W)")
                write_count            += 1
                cumulative_written_ops += ph_ops

            elif phase_type == "R" and read_phases > 0:
                ph_ops = ops_for_phase(
                    "R", f_write_ops, f_read_ops,
                    write_phases, read_phases,
                    write_count, read_count
                )
                if ph_ops <= 0:
                    read_count += 1
                    continue

                # Cap to what's on disk — IOR aborts if it reads beyond file size
                safe_read_ops = min(ph_ops, cumulative_written_ops)
                if safe_read_ops <= 0:
                    print(f"  WARNING: skipping read phase {ph_idx} — no data written yet",
                          file=sys.stderr)
                    read_count += 1
                    continue

                remaining_reads = phases[ph_idx + 1:].count("R")
                keep  = True #remaining_reads > 0
                flags = build_ior_base_flags(p, safe_read_ops, is_write=False,
                                             is_read=True, keep_files=keep)
                run_ior(flags, fp, label=f"{p['profile_name']} f{f} ph{ph_idx}(R)")
                read_count += 1

    # Cleanup — remove all workload files after the measured run
    '''for f in range(p["num_files"]):
        fp = file_path(p, f)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError as e:
                print(f"Warning: cleanup failed for {fp}: {e}", file=sys.stderr)'''
    # Cleanup handled by run_workloads.py via cleanup_workload_files()
    # after OST layout has been logged.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = parse_args(sys.argv)

    # Validate
    if p["op_size"] <= 0 or p["num_ops"] <= 0 or p["num_phases"] < 1:
        print("ERROR: op_size and num_ops must be > 0; num_phases >= 1", file=sys.stderr)
        sys.exit(1)
    if p["mode"] not in (MODE_SETUP, MODE_WORKLOAD):
        print("ERROR: mode must be 0 (setup) or 1 (workload)", file=sys.stderr)
        sys.exit(1)

    # Strided and nd_strided should never reach this wrapper —
    # run_workloads.py calls the C binary directly for these patterns.
    # Guard here in case the wrapper is called standalone for debugging.
    if p["access_pattern"] in (PATTERN_STRIDED, PATTERN_ND_STRIDED):
        print(
            f"ERROR: strided and nd_strided profiles must be run via the C binary.\n"
            f"  run_workloads.py handles this automatically. If running manually:\n"
            f"  mpirun -np 1 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'posix_synthetic_workload')} "
            f"{' '.join(sys.argv[1:])}",
            file=sys.stderr
        )
        sys.exit(1)

    os.makedirs(p["work_dir"], exist_ok=True)

    if p["mode"] == MODE_SETUP:
        run_setup(p)
    else:
        run_workload(p)


if __name__ == "__main__":
    main()