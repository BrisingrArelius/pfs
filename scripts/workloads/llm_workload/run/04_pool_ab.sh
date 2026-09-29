#!/bin/bash
# 04_pool_ab.sh — the lab's actual question: does putting a checkpoint on NVMe
# targets beat putting it on HDD targets, for a real eval read?
#
# Writes the same checkpoint twice, once pinned to each pool, then runs the
# identical eval read against both with cold caches in between.
#
# WIDTHS sweeps the reader width, which is the point rather than a detail.
# A straight eval read should be network-bound (a 4-target HDD stripe at
# ~800 MB/s already exceeds the ~312 MB/s client NIC), so at low width the two
# pools should agree. What separates them, if anything, is concurrency: at TP32
# the load is ~32 ranks x 4 stripe targets = ~128 concurrent sequential streams
# over 11 HDD targets, ~12 per spindle, which is a seek-thrashing regime NVMe
# does not care about. Prediction: converge low, diverge high. METHODS 4.3.
#
# Why pools 3 and 6 make a fair comparison, and pool 1 does not:
#
#   pool 3 "hdd"  101-103, 201-204, 301-304   11 HDD targets, colva1/2/3
#   pool 6 "ssd"  104-107, 205-206, 305-307    9 NVMe targets, colva1/2/3
#   pool 1 "Default" 401-407                   colva4 ONLY, HDD+NVMe mixed
#
# 3 and 6 span the same three OSSs, so the network path is identical and the
# only difference is media. Pool 1 would confound media with node count.
#
# Two things to fix before trusting the result (see the notes in the final
# summary): target 207 is parked in offline_pool, so ssd has 9 targets against
# hdd's 11; and target 307 is 39% used while its siblings are at 1-5%, which
# biases BeeGFS's capacity-based target selection.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${POOLS:=3 6}"
: "${POOL_NAMES:=hdd ssd}"
: "${AB_MODEL:=llama31_70b}"
: "${WIDTHS:=4 8 16 32}"

names=($POOL_NAMES); i=0
for pool in $POOLS; do
    name="${names[$i]:-pool$pool}"; i=$((i+1))
    echo
    echo "############################################################"
    echo "###  pool $pool ($name)"
    echo "############################################################"
    export POOL_ID="$pool" MODEL="$AB_MODEL"
    export TAG_BASE="${AB_MODEL}_ab_${name}"
    unset CKPT_DIR
    "$HERE/01_populate.sh"
    for tp in $WIDTHS; do
        echo
        echo "---- pool $name, reader TP$tp ----"
        LOAD_CONFIG="TP$tp" NPROC="$tp" MASTER_PORT=$((29700+tp)) \
            MODE=load "$HERE/02_eval_read.sh"
    done
done

echo
echo "############################################################"
echo "###  comparison"
echo "############################################################"
source "$HERE/env.sh"
WIDTHS="$WIDTHS" "$PYTHON" - "$RESULTS_DIR" $POOL_NAMES <<'PY'
import json, sys, glob, os
results_dir, names = sys.argv[1], sys.argv[2:]
widths = os.environ.get("WIDTHS", "").split()
print(f"{'reader':<8}" + "".join(f"{n+' GB/s':>14}" for n in names)
      + f"{'ratio':>10}")
for tp in widths:
    row, rates = f"TP{tp:<6}", []
    for n in names:
        fs = sorted(glob.glob(os.path.join(
            results_dir, f"eval_*_ab_{n}_TP{tp}_load_*.summary.json")))
        if not fs:
            row += f"{'-':>14}"; rates.append(None); continue
        d = json.load(open(fs[-1]))
        io = max(r["io_s"] for r in d["per_rank"])
        r = d["total_bytes"] / io / 1e9
        rates.append(r); row += f"{r:>14.3f}"
    if len(rates) == 2 and all(rates):
        row += f"{rates[1]/rates[0]:>9.2f}x"
    print(row)
print()
print("Converging at low TP and diverging at high TP is the predicted shape.")
print("Flat across all widths means the run is network-bound throughout and")
print("says nothing about media -- report that as the result, not as a failure.")
print()
print("Caveats to state alongside any number here:")
print("  * the client NIC is 2.5 GbE (~312 MB/s). If both pools land near that")
print("    ceiling the test is network-bound and says nothing about media.")
print("  * ssd pool has 9 targets vs hdd's 11 unless target 207 is returned")
print("    from offline_pool; fewer targets is fewer spindles AND less NIC.")
print("  * target 307 sits at 39% used, so BeeGFS's capacity-weighted target")
print("    selection will avoid it, further shrinking the effective ssd width.")
PY
