#!/bin/bash
# 05_sweep.sh — the read-amplification sweep, at full 80-layer scale.
#
# Runs entirely in MODE=plan, so it needs no RAM and never opens a data file:
# it reads .metadata and runs DCP's real load planner. That plan was verified
# byte-identical to what MODE=load actually fetches, so these are real numbers
# for the full 70B model on a 62 GB node.
#
# What it shows: reads stay ~24 MB regardless of load width (DCP never
# fragments), but once load_TP exceeds the saved TP every stored shard is read
# in full by each rank that wants any of it.
set -euo pipefail
source "$(dirname "$0")/env.sh"

: "${WIDTHS:=1 2 4 8 16 32}"
: "${READ_SET:=eval}"

banner "amplification sweep: $MODEL, saved $SAVE_CONFIG, read-set $READ_SET"
printf "%-7s %11s %12s %11s %10s %9s\n" \
       "loadTP" "requests" "bytes(GB)" "amplif." "avg req" "files"

for tp in $WIDTHS; do
    out="$RESULTS_DIR/sweep_${TAG_BASE}_${READ_SET}_tp${tp}.json"
    "$TORCHRUN" --nproc_per_node="$tp" --master_port=$((29600+tp)) \
        "$WORKLOAD_DIR/eval_load.py" --ckpt "$CKPT_DIR" --model "$MODEL" \
        --load-config "TP$tp" --mode plan --read-set "$READ_SET" \
        --optimizer "$OPTIMIZER" --summary "$out" >/dev/null 2>&1
    "$PYTHON" - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
r, b = d["total_requests"], d["total_bytes"]
base = 141.107412992e9 if d["read_set"] == "eval" else 705.56e9
print("%-7s %11s %12.2f %10.2fx %9s %9d" % (
    d["load_config"].split("/")[0], f"{r:,}", b/1e9, b/base,
    f"{b/max(r,1)/1e6:.1f} MB", d["shard_files_opened"]))
PY
done
echo
echo "results: $RESULTS_DIR/sweep_${TAG_BASE}_${READ_SET}_tp*.json"
