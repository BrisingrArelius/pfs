# parse_darshan.py

Extracts Darshan counters from one or more `.darshan` logs and writes structured CSV output.

## Usage

```bash
python3 scripts/parse_darshan.py --log <path.darshan> --label <name> --posix [--mpi] [--stdio] [--output-dir <path>]
```

At least one of `--posix`, `--mpi`, or `--stdio` is required.

## Arguments

- `--log <path>`: path to a single `.darshan` log file
- `--logs <path1,path2,...>`: comma-separated list of `.darshan` logs
- `--label <name>`: workload label used in output
- `--posix`: extract POSIX counters
- `--mpi`: extract MPI-IO counters
- `--stdio`: extract STDIO counters
- `--output-dir <path>`: output directory (default: `./darshan_output_ssd`)

## Output files

- `{label}_{modules}.csv`
  - One row per accessed file
  - Contains only the requested module counters
- `global.csv`
  - Appends one row per parser invocation
  - Contains all counters from POSIX, MPI, and STDIO modules
  - Missing counters for modules not requested are filled with `NaN`

## Notes

- This parser is used by `scripts/run_workloads.py` to generate run-level Darshan counter summaries.
- The synthetic workload suite primarily generates POSIX counters.
- If you pass `--logs`, all listed logs are aggregated into the same global CSV output.
- By default, `global.csv` is appended, so repeated runs accumulate additional rows.

## Counter aggregation

Counters are aggregated using the strategies defined in the script:

- `sum` for totals and histogram buckets
- `max` for peak values and end timestamps
- `min` for start timestamps
- `first` for alignment and metadata fields

This makes `global.csv` suitable for analysis across runs without preserving per-file detail.

## Example calls

```bash
python3 scripts/parse_darshan.py --log logs/run1.darshan --label read_heavy --posix
python3 scripts/parse_darshan.py --log logs/run2.darshan --label mixed_rw --posix --mpi --stdio --output-dir /tmp/darshan_out
python3 scripts/parse_darshan.py --logs logs/run1.darshan,logs/run2.darshan --label batch_parse --posix
```
