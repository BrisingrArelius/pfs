#!/usr/bin/env python3
"""
ior_workload_wrapper.py

Drop-in replacement for posix_synthetic_workload.c.
Translates the same 11 CLI arguments into IOR commands for POSIX workloads.
nd_strided profiles are delegated to the original compiled C binary.

Usage (identical interface to posix_synthetic_workload):
    python3 ior_workload_wrapper.py <profile_name> <read_ratio> <access_pattern>
                                    <stride_size> <op_size> <num_ops> <num_files>
                                    <num_phases> <fsync_interval> <work_dir> <mode>

access_pattern integers:
    0 = sequential
    1 = random
    2 = strided
    3 = nd_strided  → delegated to posix_synthetic_workload binary

mode:
    0 = SETUP    — write files only, no Darshan. Pure-read profiles only.
    1 = WORKLOAD — measured run, Darshan attached via LD_PRELOAD by caller.

IOR flags used:
    -a POSIX          POSIX API (matches original O_DIRECT syscall behavior)
    -b <bytes>        block size  = op_size * ops_per_file  (total per-file data)
    -t <bytes>        transfer size = op_size               (per-op size)
    -s 1              one segment (all data in one contiguous block per file)
    -F                file-per-process (each file independent, matches num_files)
    -w / -r           write / read
    -k                keep files after write (setup mode, or when read follows)
    -C                reorder tasks for read (avoids cache hits — not needed for
                      single-rank runs but kept for correctness)
    -z                random offsets (access_pattern == random)
    -i 1              one iteration (phases handled by multiple IOR invocations)
    -v                verbose output to stderr (useful for debugging)
    --posix.odirect   bypass page cache (matches O_DIRECT in the C binary)
    -e                fsync on close (used when fsync_interval > 0)
    -d <stride>       stride between tasks — used for strided pattern via
                      offset calculation (see notes below on strided mapping)

Strided pattern mapping:
    IOR does not have a native strided-offset mode equivalent to the C binary's
    (cursor * stride_size) % file_size formula. We approximate it by setting
    the IOR offset increment to stride_size using --posix.stride, which causes
    each transfer to start stride_size bytes after the previous one. This is
    equivalent to the C binary's strided pattern when stride_size >= op_size.
    When stride_size < op_size the pattern collapses to sequential — same
    behavior as the C binary after op_size alignment rounding.

nd_strided delegation:
    IOR has no equivalent for the alternating row/column-major 2D traversal
    implemented in posix_synthetic_workload.c. These profiles are passed
    directly to the compiled C binary unchanged. The binary path is resolved
    relative to this script's directory.
"""

import os
import sys
import subprocess
import math

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# IOR binary — must be on PATH or set absolute path here
IOR_BIN = "ior"

# Fallback C binary for nd_strided profiles — relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSIX_BIN  = os.path.join(SCRIPT_DIR, "posix_synthetic_workload")

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
# nd_strided delegation — pass all args straight to the C binary
# ---------------------------------------------------------------------------

def delegate_to_c_binary(argv):
    """
    Pass the original argv unchanged to posix_synthetic_workload.
    Used for nd_strided profiles which IOR cannot replicate.
    """
    if not os.path.isfile(POSIX_BIN):
        print(
            f"ERROR: nd_strided profile requires the compiled C binary at:\n"
            f"  {POSIX_BIN}\n"
            f"Compile it first with: mpicc -O2 -o {POSIX_BIN} "
            f"{POSIX_BIN}.c -ldarshan -lpthread -lrt -lz",
            file=sys.stderr
        )
        sys.exit(1)

    cmd = [POSIX_BIN] + argv[1:]
    print(f"[nd_strided] Delegating to C binary: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


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


def block_size(ops, p):
    """Total bytes per file = ops * op_size."""
    return ops * p["op_size"]


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

    Sequential / Strided:
      -t = op_size
      -b = op_size   (one transfer per block)
      -s = n_ops     (number of blocks)

    Random (-z):
      IOR randomizes offsets *within* a block, so blocksize must be > transfersize.
      We use:
        -t = op_size
        -b = next_power_of_2(n_ops * op_size)   (whole file as one block, power-of-2)
        -s = 1
      The power-of-2 rounding means the actual data written is slightly more
      than requested but is the only way to satisfy IOR's -z constraint cleanly.
    """
    transfer = p["op_size"]

    if p["access_pattern"] == PATTERN_RANDOM:
        # Block must be > transfer and power-of-2 for -z to work
        total = n_ops * p["op_size"]
        block = next_power_of_2(total + p["op_size"])  # +op_size ensures block > transfer
        segs  = 1
    else:
        block = p["op_size"]
        segs  = n_ops

    flags = [
        IOR_BIN,
        "-a", "POSIX",
        "-b", str(block),
        "-t", str(transfer),
        "-s", str(segs),
        "-i", "1",          # one iteration per IOR call
        "-v",               # verbose
        # Note: -F (file-per-process) intentionally omitted. With mpirun -np 1
        # there is only one rank so -F has no effect on parallelism, but it
        # causes IOR to append a .00000000 rank suffix to the filepath which
        # breaks the setup->workload file handoff. Without -F, IOR uses the
        # exact path provided, matching what setup wrote.
    ]

    # O_DIRECT only on writes — random read offsets from -z are not guaranteed
    # to be 4096-aligned internally in IOR, causing MPI_Abort on read phases.
    # Cache is cleared before each run so reads without O_DIRECT are still
    # valid cold-storage measurements.
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

    # Random access
    if p["access_pattern"] == PATTERN_RANDOM:
        flags.append("-z")

    # Strided access
    if p["access_pattern"] == PATTERN_STRIDED:
        flags += ["--posix.stride", str(p["stride_size"])]

    return flags
    if p["fsync_interval"] > 0:
        flags.append("-e")

    # Random access
    if p["access_pattern"] == PATTERN_RANDOM:
        flags.append("-z")

    # Strided access — set offset stride via posix.stride
    # IOR advances the file offset by posix.stride bytes between transfers.
    # This approximates (cursor * stride_size) % file_size from the C binary.
    if p["access_pattern"] == PATTERN_STRIDED:
        flags += ["--posix.stride", str(p["stride_size"])]

    return flags


def run_ior(flags, filepath, label="", use_mpi=True, env=None):
    """
    Execute one IOR invocation. Filepath is passed via -o.
    IOR appends '.00000000' etc. to the path for file-per-process mode —
    we strip this by using -o with the exact path and relying on -F with
    a single rank (mpirun -np 1) so the suffix is always .00000000.
    The C binary used a deterministic _f{N} suffix; we mirror that by
    constructing the path ourselves and passing it directly.

    use_mpi=True  -- wrap with mpirun -np 1 (workload mode, Darshan must attach)
    use_mpi=False -- run IOR directly (setup mode, Darshan must NOT attach)
    """
    if use_mpi:
        # Workload mode: wrap with mpirun so Darshan initializes properly via
        # MPI_Init/Finalize. LD_PRELOAD is inherited from run_workloads.py.
        cmd = ["mpirun", "-np", "1"] + flags + ["-o", filepath]
    else:
        # Setup mode: run IOR directly without mpirun so Darshan cannot
        # initialize -- setup I/O must not appear in any Darshan log.
        cmd = flags + ["-o", filepath]
    tag = f"[{label}] " if label else ""
    print(f"{tag}IOR: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, env=env)  # env=None means inherit parent env
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
      - Pure write:                 all phases are writes (odd phases are no-ops)
      - Mixed:                      even phases = write, odd phases = read

    Returns a list of "W" / "R" strings of length num_phases.
    """
    phases = []
    if p["mode"] == MODE_WORKLOAD and p["read_ratio"] >= 1.0:
        phases = ["R"] * p["num_phases"]
    else:
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
    Mixed/write profiles don't call setup — this is consistent with the
    C binary's behavior (run_workloads.py only calls setup for read_ratio >= 1.0).

    Setup always writes sequentially regardless of access pattern, matching
    the C binary's setup behavior.
    """
    os.makedirs(p["work_dir"], exist_ok=True)

    # Strip LD_PRELOAD so Darshan cannot attach to the setup IOR process
    clean_env = os.environ.copy()
    clean_env.pop("LD_PRELOAD", None)

    base_ops, last_ops = ops_per_file(p)

    # For pure-read, setup writes num_ops worth of data so the file is fully
    # populated. This matches the C binary's run_setup() logic.
    setup_write_ops = p["num_ops"]  # always write everything in setup

    ops_per_f   = setup_write_ops // p["num_files"]
    last_file_ops = setup_write_ops - ops_per_f * (p["num_files"] - 1)

    for f in range(p["num_files"]):
        n_ops = last_file_ops if f == p["num_files"] - 1 else ops_per_f
        fp    = file_path(p, f)

        # Setup writes only, always sequential (no -z, no --posix.stride),
        # keeps files (-k) so workload mode can read them.
        # Setup geometry mirrors build_ior_base_flags:
        # random needs -b power_of_2 -s 1, others use -b op_size -s n_ops
        if p["access_pattern"] == PATTERN_RANDOM:
            total = n_ops * p["op_size"]
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
            "-k",
        ]
        run_ior(flags, fp, label=f"setup f{f}", use_mpi=False, env=clean_env)

        # Verify file was created at expected path
        if not os.path.exists(fp):
            print(f"[setup f{f}] WARNING: expected file not found at {fp}",
                  file=sys.stderr)


def _clean_env():
    """Return a copy of the environment with LD_PRELOAD removed."""
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    return env


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

    phases      = plan_phases(p)
    write_phases = phases.count("W")
    read_phases  = phases.count("R")

    for f in range(p["num_files"]):
        n_ops = last_ops if f == p["num_files"] - 1 else base_ops
        fp    = file_path(p, f)

        # Per-file read/write op counts (mirrors C binary integer distribution)
        f_read_ops  = int(n_ops * p["read_ratio"])
        f_write_ops = n_ops - f_read_ops

        write_count = 0
        read_count  = 0

        # For pure-read profiles, verify the setup file exists before proceeding
        if p["read_ratio"] >= 1.0:
            if not os.path.exists(fp):
                print(f"ERROR: setup file not found: {fp} — was setup mode run first?",
                      file=sys.stderr)
                sys.exit(1)

        for ph_idx, phase_type in enumerate(phases):
            is_last_phase = (ph_idx == len(phases) - 1)

            if phase_type == "W" and write_phases > 0:
                ph_ops = ops_for_phase(
                    "W", f_write_ops, f_read_ops,
                    write_phases, read_phases,
                    write_count, read_count
                )
                if ph_ops <= 0:
                    write_count += 1
                    continue

                # Keep file if a read phase follows, or if it's a pure-write
                # profile (no reads will come, but workload cleans up at end)
                keep = read_phases > 0

                flags = build_ior_base_flags(p, ph_ops, is_write=True,
                                             is_read=False, keep_files=keep)
                run_ior(flags, fp, label=f"{p['profile_name']} f{f} ph{ph_idx}(W)")
                write_count += 1

            elif phase_type == "R" and read_phases > 0:
                ph_ops = ops_for_phase(
                    "R", f_write_ops, f_read_ops,
                    write_phases, read_phases,
                    write_count, read_count
                )
                if ph_ops <= 0:
                    read_count += 1
                    continue

                # Keep file only if more read phases follow for this file
                remaining_reads = phases[ph_idx + 1:].count("R")
                keep = remaining_reads > 0

                flags = build_ior_base_flags(p, ph_ops, is_write=False,
                                             is_read=True, keep_files=keep)
                run_ior(flags, fp, label=f"{p['profile_name']} f{f} ph{ph_idx}(R)")
                read_count += 1

    # Cleanup — remove all workload files after the measured run
    for f in range(p["num_files"]):
        fp = file_path(p, f)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError as e:
                print(f"Warning: cleanup failed for {fp}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = parse_args(sys.argv)

    # Validate
    if p["op_size"] <= 0 or p["num_ops"] <= 0 or p["num_phases"] < 1:
        print("ERROR: op_size and num_ops must be > 0; num_phases >= 1", file=sys.stderr)
        sys.exit(1)
    if p["access_pattern"] in (PATTERN_STRIDED, PATTERN_ND_STRIDED) and p["stride_size"] <= 0:
        print("ERROR: stride_size must be > 0 for strided/nd_strided patterns", file=sys.stderr)
        sys.exit(1)
    if p["mode"] not in (MODE_SETUP, MODE_WORKLOAD):
        print("ERROR: mode must be 0 (setup) or 1 (workload)", file=sys.stderr)
        sys.exit(1)

    # nd_strided: delegate entirely to the C binary
    if p["access_pattern"] == PATTERN_ND_STRIDED:
        delegate_to_c_binary(sys.argv)
        # delegate_to_c_binary calls sys.exit() — never reaches here

    os.makedirs(p["work_dir"], exist_ok=True)

    if p["mode"] == MODE_SETUP:
        run_setup(p)
    else:
        run_workload(p)


if __name__ == "__main__":
    main()