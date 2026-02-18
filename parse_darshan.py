#!/usr/bin/env python3
"""
parse_darshan.py

Parses a Darshan log file and extracts I/O counters into:
  - A per-run CSV:  {label}_{modules}.csv  (one row per file accessed)
  - A global CSV:   global.csv             (one row per run, all counters, NaN if not collected)

Usage:
    python parse_darshan.py --log <path.darshan> --label <name> [--posix] [--mpi] [--stdio] [--output-dir <path>]
"""

import argparse
import os
import sys

import darshan
import pandas as pd

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_DIR = "./darshan_output"
GLOBAL_CSV = "global.csv"

# =============================================================================
# COUNTER CONFIGURATION
# Each counter maps to an aggregation strategy:
#   "sum"   - total across all files (counts, bytes, histogram buckets)
#   "max"   - highest value across files (peak times, sizes)
#   "min"   - lowest value across files (earliest timestamps)
#   "first" - take first non-NaN value (alignment properties, same for all files)
# =============================================================================

POSIX_COUNTERS = {
    # Operation counts
    "POSIX_OPENS":                  "sum",
    "POSIX_READS":                  "sum",
    "POSIX_WRITES":                 "sum",
    "POSIX_SEEKS":                  "sum",
    "POSIX_STATS":                  "sum",
    "POSIX_MMAPS":                  "sum",
    "POSIX_FSYNCS":                 "sum",
    "POSIX_FDSYNCS":                "sum",
    # Bytes
    "POSIX_BYTES_READ":             "sum",
    "POSIX_BYTES_WRITTEN":          "sum",
    # Access patterns
    "POSIX_CONSEC_READS":           "sum",
    "POSIX_CONSEC_WRITES":          "sum",
    "POSIX_SEQ_READS":              "sum",
    "POSIX_SEQ_WRITES":             "sum",
    "POSIX_RW_SWITCHES":            "sum",
    # Alignment
    "POSIX_MEM_NOT_ALIGNED":        "sum",
    "POSIX_MEM_ALIGNMENT":          "first",
    "POSIX_FILE_NOT_ALIGNED":       "sum",
    "POSIX_FILE_ALIGNMENT":         "first",
    # Peak operation sizes
    "POSIX_MAX_READ_TIME_SIZE":     "max",
    "POSIX_MAX_WRITE_TIME_SIZE":    "max",
    # Read size histogram buckets
    "POSIX_SIZE_READ_0_100":        "sum",
    "POSIX_SIZE_READ_100_1K":       "sum",
    "POSIX_SIZE_READ_1K_10K":       "sum",
    "POSIX_SIZE_READ_10K_100K":     "sum",
    "POSIX_SIZE_READ_100K_1M":      "sum",
    "POSIX_SIZE_READ_1M_4M":        "sum",
    "POSIX_SIZE_READ_4M_10M":       "sum",
    "POSIX_SIZE_READ_10M_100M":     "sum",
    "POSIX_SIZE_READ_100M_1G":      "sum",
    "POSIX_SIZE_READ_1G_PLUS":      "sum",
    # Write size histogram buckets
    "POSIX_SIZE_WRITE_0_100":       "sum",
    "POSIX_SIZE_WRITE_100_1K":      "sum",
    "POSIX_SIZE_WRITE_1K_10K":      "sum",
    "POSIX_SIZE_WRITE_10K_100K":    "sum",
    "POSIX_SIZE_WRITE_100K_1M":     "sum",
    "POSIX_SIZE_WRITE_1M_4M":       "sum",
    "POSIX_SIZE_WRITE_4M_10M":      "sum",
    "POSIX_SIZE_WRITE_10M_100M":    "sum",
    "POSIX_SIZE_WRITE_100M_1G":     "sum",
    "POSIX_SIZE_WRITE_1G_PLUS":     "sum",
    # Stride patterns
    "POSIX_STRIDE1_STRIDE":         "sum",
    "POSIX_STRIDE2_STRIDE":         "sum",
    "POSIX_STRIDE3_STRIDE":         "sum",
    "POSIX_STRIDE4_STRIDE":         "sum",
    "POSIX_STRIDE1_COUNT":          "sum",
    "POSIX_STRIDE2_COUNT":          "sum",
    "POSIX_STRIDE3_COUNT":          "sum",
    "POSIX_STRIDE4_COUNT":          "sum",
    # Timestamps (float counters)
    "POSIX_F_OPEN_START_TIMESTAMP":  "min",
    "POSIX_F_OPEN_END_TIMESTAMP":    "max",
    "POSIX_F_READ_START_TIMESTAMP":  "min",
    "POSIX_F_READ_END_TIMESTAMP":    "max",
    "POSIX_F_WRITE_START_TIMESTAMP": "min",
    "POSIX_F_WRITE_END_TIMESTAMP":   "max",
    "POSIX_F_CLOSE_START_TIMESTAMP": "min",
    "POSIX_F_CLOSE_END_TIMESTAMP":   "max",
    # Timing (float counters)
    "POSIX_F_READ_TIME":            "sum",
    "POSIX_F_WRITE_TIME":           "sum",
    "POSIX_F_META_TIME":            "sum",
    "POSIX_F_MAX_READ_TIME":        "max",
    "POSIX_F_MAX_WRITE_TIME":       "max",
}

MPI_COUNTERS = {
    # Operation counts
    "MPIIO_INDEP_OPENS":            "sum",
    "MPIIO_COLL_OPENS":             "sum",
    "MPIIO_INDEP_READS":            "sum",
    "MPIIO_INDEP_WRITES":           "sum",
    "MPIIO_COLL_READS":             "sum",
    "MPIIO_COLL_WRITES":            "sum",
    "MPIIO_SPLIT_READS":            "sum",
    "MPIIO_SPLIT_WRITES":           "sum",
    "MPIIO_NB_READS":               "sum",
    "MPIIO_NB_WRITES":              "sum",
    "MPIIO_SYNCS":                  "sum",
    # Bytes
    "MPIIO_BYTES_READ":             "sum",
    "MPIIO_BYTES_WRITTEN":          "sum",
    # Access patterns
    "MPIIO_RW_SWITCHES":            "sum",
    # Peak operation sizes
    "MPIIO_MAX_READ_TIME_SIZE":     "max",
    "MPIIO_MAX_WRITE_TIME_SIZE":    "max",
    # Read size histogram buckets
    "MPIIO_SIZE_READ_AGG_0_100":    "sum",
    "MPIIO_SIZE_READ_AGG_100_1K":   "sum",
    "MPIIO_SIZE_READ_AGG_1K_10K":   "sum",
    "MPIIO_SIZE_READ_AGG_10K_100K": "sum",
    "MPIIO_SIZE_READ_AGG_100K_1M":  "sum",
    "MPIIO_SIZE_READ_AGG_1M_4M":    "sum",
    "MPIIO_SIZE_READ_AGG_4M_10M":   "sum",
    "MPIIO_SIZE_READ_AGG_10M_100M": "sum",
    "MPIIO_SIZE_READ_AGG_100M_1G":  "sum",
    "MPIIO_SIZE_READ_AGG_1G_PLUS":  "sum",
    # Write size histogram buckets
    "MPIIO_SIZE_WRITE_AGG_0_100":    "sum",
    "MPIIO_SIZE_WRITE_AGG_100_1K":   "sum",
    "MPIIO_SIZE_WRITE_AGG_1K_10K":   "sum",
    "MPIIO_SIZE_WRITE_AGG_10K_100K": "sum",
    "MPIIO_SIZE_WRITE_AGG_100K_1M":  "sum",
    "MPIIO_SIZE_WRITE_AGG_1M_4M":    "sum",
    "MPIIO_SIZE_WRITE_AGG_4M_10M":   "sum",
    "MPIIO_SIZE_WRITE_AGG_10M_100M": "sum",
    "MPIIO_SIZE_WRITE_AGG_100M_1G":  "sum",
    "MPIIO_SIZE_WRITE_AGG_1G_PLUS":  "sum",
    # Timestamps (float counters)
    "MPIIO_F_OPEN_START_TIMESTAMP":  "min",
    "MPIIO_F_OPEN_END_TIMESTAMP":    "max",
    "MPIIO_F_READ_START_TIMESTAMP":  "min",
    "MPIIO_F_READ_END_TIMESTAMP":    "max",
    "MPIIO_F_WRITE_START_TIMESTAMP": "min",
    "MPIIO_F_WRITE_END_TIMESTAMP":   "max",
    "MPIIO_F_CLOSE_START_TIMESTAMP": "min",
    "MPIIO_F_CLOSE_END_TIMESTAMP":   "max",
    # Timing (float counters)
    "MPIIO_F_READ_TIME":            "sum",
    "MPIIO_F_WRITE_TIME":           "sum",
    "MPIIO_F_META_TIME":            "sum",
    "MPIIO_F_MAX_READ_TIME":        "max",
    "MPIIO_F_MAX_WRITE_TIME":       "max",
}

STDIO_COUNTERS = {
    # Operation counts
    "STDIO_OPENS":                  "sum",
    "STDIO_READS":                  "sum",
    "STDIO_WRITES":                 "sum",
    "STDIO_FLUSHES":                "sum",
    # Bytes
    "STDIO_BYTES_WRITTEN":          "sum",
    "STDIO_BYTES_READ":             "sum",
    # Timing (float counters)
    "STDIO_F_META_TIME":            "sum",
    "STDIO_F_WRITE_TIME":           "sum",
    "STDIO_F_READ_TIME":            "sum",
    # Timestamps (float counters)
    "STDIO_F_OPEN_START_TIMESTAMP":  "min",
    "STDIO_F_OPEN_END_TIMESTAMP":    "max",
    "STDIO_F_READ_START_TIMESTAMP":  "min",
    "STDIO_F_READ_END_TIMESTAMP":    "max",
    "STDIO_F_WRITE_START_TIMESTAMP": "min",
    "STDIO_F_WRITE_END_TIMESTAMP":   "max",
    "STDIO_F_CLOSE_START_TIMESTAMP": "min",
    "STDIO_F_CLOSE_END_TIMESTAMP":   "max",
}

# Map module name to its counter config dict and Darshan module key
MODULE_CONFIG = {
    "posix": {"counters": POSIX_COUNTERS, "darshan_key": "POSIX"},
    "mpi":   {"counters": MPI_COUNTERS,   "darshan_key": "MPI-IO"},
    "stdio": {"counters": STDIO_COUNTERS, "darshan_key": "STDIO"},
}

# All counters across all modules in order (used for global CSV column layout)
ALL_COUNTERS = list(POSIX_COUNTERS.keys()) + list(MPI_COUNTERS.keys()) + list(STDIO_COUNTERS.keys())


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a Darshan log and extract I/O counters to CSV."
    )
    parser.add_argument("--log",        required=True,  help="Path to the .darshan log file")
    parser.add_argument("--label",      required=True,  help="Workload label (used in output filename)")
    parser.add_argument("--posix",      action="store_true", help="Extract POSIX counters")
    parser.add_argument("--mpi",        action="store_true", help="Extract MPI-IO counters")
    parser.add_argument("--stdio",      action="store_true", help="Extract STDIO counters")
    parser.add_argument("--output-dir", default=None,   help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    if not any([args.posix, args.mpi, args.stdio]):
        parser.error("At least one of --posix, --mpi, --stdio must be specified.")

    if not os.path.isfile(args.log):
        parser.error(f"Log file not found: {args.log}")

    return args


# =============================================================================
# LOG LOADING
# =============================================================================

def load_report(log_path):
    """Load a Darshan log file and return a DarshanReport."""
    print(f"Loading log: {log_path}")
    report = darshan.DarshanReport(log_path, read_all=False)
    return report


# =============================================================================
# COUNTER EXTRACTION
# =============================================================================

def extract_module(report, module_name):
    """
    Extract per-file counter data for a given module.

    Returns a DataFrame with one row per file and all counters for that module
    as columns. Counters missing from the log are filled with NaN.
    Returns None if the module has no records in this log.
    """
    cfg = MODULE_CONFIG[module_name]
    darshan_key = cfg["darshan_key"]
    counter_dict = cfg["counters"]

    if darshan_key not in report.modules:
        print(f"  [{module_name.upper()}] No records found in log — filling with NaN.")
        return None

    report.mod_read_all_records(darshan_key)
    raw = report.records[darshan_key].to_df()

    # raw is a dict with "counters" (int) and "fcounters" (float) DataFrames
    # each has a column "id" identifying the file record
    counters_df  = raw.get("counters")
    fcounters_df = raw.get("fcounters")

    if counters_df is not None and fcounters_df is not None:
        per_file_df = pd.merge(counters_df, fcounters_df, on="id", suffixes=("", "_f"))
    elif counters_df is not None:
        per_file_df = counters_df
    elif fcounters_df is not None:
        per_file_df = fcounters_df
    else:
        print(f"  [{module_name.upper()}] Empty records — filling with NaN.")
        return None

    # Keep only the counters we care about, filling missing ones with NaN
    result_cols = {}
    for counter in counter_dict:
        if counter in per_file_df.columns:
            result_cols[counter] = per_file_df[counter]
        else:
            print(f"  [{module_name.upper()}] Counter not found in log: {counter} — filling with NaN.")
            result_cols[counter] = float("nan")

    # Preserve file id for the per-run CSV
    result_df = pd.DataFrame(result_cols)
    if "id" in per_file_df.columns:
        result_df.insert(0, "file_id", per_file_df["id"])

    return result_df


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_module(per_file_df, module_name):
    """
    Aggregate a per-file DataFrame into a single dict (one value per counter)
    using the aggregation strategy defined in the counter config.
    """
    counter_dict = MODULE_CONFIG[module_name]["counters"]
    aggregated = {}

    for counter, strategy in counter_dict.items():
        if counter not in per_file_df.columns:
            aggregated[counter] = float("nan")
            continue

        col = per_file_df[counter].dropna()
        if col.empty:
            aggregated[counter] = float("nan")
            continue

        if strategy == "sum":
            aggregated[counter] = col.sum()
        elif strategy == "max":
            aggregated[counter] = col.max()
        elif strategy == "min":
            aggregated[counter] = col.min()
        elif strategy == "first":
            aggregated[counter] = col.iloc[0]
        else:
            aggregated[counter] = float("nan")

    return aggregated


# =============================================================================
# FILE NAMING
# =============================================================================

def build_run_csv_name(label, modules, output_dir):
    """
    Build a unique per-run CSV filename.
    Format: {label}_{mod1_mod2}.csv
    If the file already exists, append _1, _2, etc.
    """
    modules_str = "_".join(sorted(modules))
    base_name = f"{label}_{modules_str}"
    candidate = os.path.join(output_dir, f"{base_name}.csv")

    if not os.path.exists(candidate):
        return candidate

    i = 1
    while True:
        candidate = os.path.join(output_dir, f"{base_name}_{i}.csv")
        if not os.path.exists(candidate):
            return candidate
        i += 1


# =============================================================================
# CSV WRITING
# =============================================================================

def write_per_run_csv(per_file_dfs, label, timestamp, output_path):
    """
    Write per-file CSV for this run.
    Columns: timestamp, label, file_id, <module counters...>
    One row per file, only columns for requested modules.
    """
    combined = pd.concat(per_file_dfs, axis=1)

    # Remove duplicate file_id columns from concat
    combined = combined.loc[:, ~combined.columns.duplicated()]

    combined.insert(0, "label", label)
    combined.insert(0, "timestamp", timestamp)

    combined.to_csv(output_path, index=False)
    print(f"  Per-run CSV written: {output_path}")


def write_global_csv(aggregated_row, output_dir):
    """
    Append one row to global.csv.
    All counters always present as columns; NaN for modules not in this run.
    Writes header if global.csv does not exist yet.
    """
    global_path = os.path.join(output_dir, GLOBAL_CSV)

    # Build a row with ALL counters in fixed order, NaN for missing
    full_row = {counter: aggregated_row.get(counter, float("nan")) for counter in ALL_COUNTERS}

    # Prepend metadata columns
    meta = {
        "timestamp":    aggregated_row["timestamp"],
        "label":        aggregated_row["label"],
        "modules_used": aggregated_row["modules_used"],
    }
    full_row = {**meta, **full_row}

    row_df = pd.DataFrame([full_row])
    write_header = not os.path.exists(global_path)

    row_df.to_csv(global_path, mode="a", header=write_header, index=False)
    print(f"  Global CSV updated:  {global_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Determine which modules were requested
    requested_modules = []
    if args.posix: requested_modules.append("posix")
    if args.mpi:   requested_modules.append("mpi")
    if args.stdio: requested_modules.append("stdio")

    timestamp = pd.Timestamp.now().isoformat()

    print(f"\nLabel:   {args.label}")
    print(f"Modules: {', '.join(requested_modules)}")
    print(f"Output:  {output_dir}\n")

    # Load the Darshan log
    report = load_report(args.log)

    # Extract and aggregate each requested module
    per_file_dfs  = []   # for per-run CSV
    aggregated_row = {   # for global CSV
        "timestamp":    timestamp,
        "label":        args.label,
        "modules_used": "_".join(sorted(requested_modules)),
    }

    for module in requested_modules:
        print(f"Extracting {module.upper()} counters...")
        per_file_df = extract_module(report, module)

        if per_file_df is not None:
            per_file_dfs.append(per_file_df)
            agg = aggregate_module(per_file_df, module)
        else:
            # Module not in log — all counters NaN
            agg = {counter: float("nan") for counter in MODULE_CONFIG[module]["counters"]}

        aggregated_row.update(agg)

    # Write per-run CSV (only if we got any data)
    if per_file_dfs:
        run_csv_path = build_run_csv_name(args.label, requested_modules, output_dir)
        write_per_run_csv(per_file_dfs, args.label, timestamp, run_csv_path)
    else:
        print("  No per-file data extracted — skipping per-run CSV.")

    # Write to global CSV
    write_global_csv(aggregated_row, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
