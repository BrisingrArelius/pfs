#!/usr/bin/env python3
"""U-Net 3D: one SSD condition or the complete five-pool speed-mix sweep."""

import argparse

from pool_common import FAST, run
from configure_pool import restore, transition


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
        for pool in FAST:
            transition(6 if pool == 14 else pool - 1, pool)
            print(f"Starting three full U-Net runs on pool {pool}", flush=True)
            run("unet", pool)
        restore(18)
    else:
        run("unet", args.pool, args.pilot)


if __name__ == "__main__":
    main()
