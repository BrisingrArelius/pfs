#!/usr/bin/env python3
"""U-Net 3D: one four-target throttled-SSD condition (pilot or three runs)."""

import argparse

from pool_common import FAST, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, required=True, choices=FAST)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run("unet", args.pool, args.pilot)


if __name__ == "__main__":
    main()
