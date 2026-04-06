#!/usr/bin/env python3
import sys

def parse_du_output(filepath):
    """Parses a du -h output text file and categorizes the usage."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    # Simple categorization based on size suffixes
    gigabytes = []
    megabytes = []
    kilobytes = []
    bytes_others = []

    print("\n--- Disk Usage Summary ---")
    print(f"Reading from: {filepath}\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) >= 2:
            size_str = parts[0]
            path = parts[1]
            
            # Remove any trailing 'M', 'G', 'K', etc to cast to float if needed later
            if 'G' in size_str:
                gigabytes.append((size_str, path))
            elif 'M' in size_str:
                megabytes.append((size_str, path))
            elif 'K' in size_str:
                kilobytes.append((size_str, path))
            else:
                bytes_others.append((size_str, path))

    # Sort largest items within their categories (simple text sort relies on uniform digits, but good enough for a quick glance)
    gigabytes.sort(key=lambda x: float(x[0].replace('G', '')) if x[0].replace('G', '').replace('.','',1).isdigit() else 0, reverse=True)
    megabytes.sort(key=lambda x: float(x[0].replace('M', '')) if x[0].replace('M', '').replace('.','',1).isdigit() else 0, reverse=True)

    # Print the significant offenders
    if gigabytes:
        print(f"🔴 MASSIVE DIRECTORIES (> 1 GiB)  [{len(gigabytes)} found]")
        print("--------------------------------------------------")
        for size, path in gigabytes:
            print(f"{size:>8}  {path}")
        print()

    if megabytes:
        print(f"🟡 LARGE DIRECTORIES (> 1 MiB)    [{len(megabytes)} found]")
        print("--------------------------------------------------")
        for size, path in megabytes[:15]: # Show top 15
            print(f"{size:>8}  {path}")
        if len(megabytes) > 15:
            print(f"  ... and {len(megabytes) - 15} more directories.")
        print()

    print(f"🟢 SMALL DIRECTORIES (< 1 MiB)    [{len(kilobytes) + len(bytes_others)} found]")
    print("--------------------------------------------------")
    print(" (Hiding smaller objects. See raw file for full list)\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 parse_du.py <du_output.txt>")
        sys.exit(1)
        
    parse_du_output(sys.argv[1])