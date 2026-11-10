#!/usr/bin/env python3
"""CosmoFlow: one SSD condition or the 4/0, 2/2, 0/4 speed-mix sweep."""

import argparse

from pool_common import run, run_all
from dlio_darshan import add_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--pool", type=int, choices=(14, 16, 18))
    choice.add_argument("--all-pools", action="store_true",
                        help="Start from pool 6, run pools 14, 16, 18 (4/0, 2/2, 0/4), then restore pool 6")
    parser.add_argument("--pilot", action="store_true")
    add_argument(parser)
    args = parser.parse_args()
    if args.all_pools:
        if args.pilot:
            parser.error("--pilot requires a single --pool")
        run_all("cosmoflow", darshan=args.darshan)
    else:
        run("cosmoflow", args.pool, args.pilot, darshan=args.darshan)


if __name__ == "__main__":
    main()
