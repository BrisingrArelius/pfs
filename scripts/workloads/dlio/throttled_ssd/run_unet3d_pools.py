#!/usr/bin/env python3
"""U-Net 3D: one SSD condition or the complete five-pool speed-mix sweep."""

import argparse

from pool_common import FAST, run, run_all
from dlio_darshan import add_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--pool", type=int, choices=FAST)
    choice.add_argument("--all-pools", action="store_true",
                        help="Start from original pool 6, run pools 14–18, then restore pool 6")
    parser.add_argument("--pilot", action="store_true")
    add_argument(parser)
    args = parser.parse_args()
    if args.all_pools:
        if args.pilot:
            parser.error("--pilot requires a single --pool")
        run_all("unet", darshan=args.darshan)
    else:
        run("unet", args.pool, args.pilot, darshan=args.darshan)


if __name__ == "__main__":
    main()
