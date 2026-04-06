# parse_du.py

A small utility to parse text output from `du -h` and summarize large directories.

## Usage

```bash
python3 scripts/parse_du.py <du_output.txt>
```

## What it does

The script reads a `du -h` text file and groups entries by size range:
- directories larger than 1 GiB
- directories larger than 1 MiB
- smaller entries that are not shown in detail

## Output

The script prints a simple disk usage summary with the largest directories at the top.
