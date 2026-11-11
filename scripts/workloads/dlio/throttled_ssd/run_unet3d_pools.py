#!/usr/bin/env python3
"""U-Net 3D: one SSD condition or the 4/0, 2/2, 0/4 speed-mix sweep."""

import argparse

from pool_common import EXPERIMENT_POOLS, run, run_all
from dlio_darshan import add_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--pool", type=int, choices=EXPERIMENT_POOLS)
    choice.add_argument("--all-pools", action="store_true",
                        help="Start from original pool 6, run pools 14, 16, 18, then restore pool 6")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repetitions",type=int,default=3,help="Full invocations per pool (default: 3; use 1 for capture plots)")
    parser.add_argument("--pools",type=int,nargs="+",choices=EXPERIMENT_POOLS,help="Pool selection/order for --all-pools")
    add_argument(parser)
    args = parser.parse_args()
    if args.repetitions<1:
        parser.error("--repetitions must be positive")
    if args.pools is not None and (not args.all_pools or len(set(args.pools))!=len(args.pools)):
        parser.error("--pools requires --all-pools and distinct pool IDs")
    if args.all_pools:
        if args.pilot:
            parser.error("--pilot requires a single --pool")
        run_all("unet", darshan=args.darshan, repetitions=args.repetitions, pools=args.pools)
    else:
        run("unet", args.pool, args.pilot, darshan=args.darshan, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
