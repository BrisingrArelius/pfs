#!/usr/bin/env python3
"""U-Net 3D: one SSD condition or the complete five-pool speed-mix sweep."""

import argparse

from pool_common import FAST, run, run_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--pool", type=int, choices=FAST)
    choice.add_argument("--all-pools", action="store_true",
                        help="Start from original pool 6, run pools 14–18, then restore pool 6")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.all_pools:
        if args.pilot:
            parser.error("--pilot requires a single --pool")
        run_all("unet")
    else:
        run("unet", args.pool, args.pilot)


if __name__ == "__main__":
    main()
