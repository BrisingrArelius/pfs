#!/bin/bash
# 05_run_matrix.sh — run the eval checkpoint read for every config, REPS times.
#
#   ./05_run_matrix.sh                          # 6 configs x 3 reps (~2.5-3 h)
#   RUN_CONFIGS="allF" REPS=1 ./05_run_matrix.sh   # one smoke run
#
# Each run, in order:
#   1. throttles checked on all 12 devices (abort if any are missing)
#   2. page cache dropped on anjuna2 AND colva1-4 (abort if any drop fails;
#      a warm OSS serves from RAM and bypasses the throttle)
#   3. torchrun eval_load.py --read-set eval on the config dir, under Darshan
#   4. parse_darshan.py with that config's target map
#
# Configs are shuffled within each rep so drift (other users, thermal, ...)
# does not line up with one config. Every run is recorded in
# $RESULTS_DIR/runs.tsv; re-running the script skips (config, rep) pairs that
# already finished OK, so an interrupted session resumes where it stopped.
# Run it inside tmux/screen: it takes hours.
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"

REPS="${REPS:-3}"
read -ra RUN_CONFIGS <<<"${RUN_CONFIGS:-${CONFIGS[*]}}"
INDEX="$RESULTS_DIR/runs.tsv"
[[ -f "$INDEX" ]] || printf 'config\trep\ttag\tstart_epoch\tend_epoch\tstatus\n' > "$INDEX"
[[ -f "$RESULTS_DIR/configs.json" ]] || { echo "run 04_make_configs.sh first" >&2; exit 1; }

done_ok() { awk -F'\t' -v c="$1" -v r="$2" '$1==c && $2==r && $6=="ok" {f=1} END {exit !f}' "$INDEX"; }

run_one() {
    local cfg=$1 rep=$2 ckpt tag start rc=0
    ckpt="$(cfg_dir "$cfg")"
    tag="v2_${cfg}_r${rep}_$(date +%Y%m%d_%H%M%S)"
    [[ -e "$ckpt/.metadata" ]] || { echo "  $ckpt missing" >&2; return 1; }

    banner "[$cfg rep $rep] $tag"
    echo "  checkpoint $ckpt"
    echo "  files 0-7  ${PLACEMENT[$cfg]}"
    # rc 2 = the setup is invalid for every later run too; the caller stops.
    # (set -e does not apply inside a function called from `||`, hence the
    # explicit returns.)
    require_throttles || return 2
    banner "cold cache"
    drop_caches_strict || return 2
    banner "tracing"
    setup_darshan || { echo "  Darshan is required for per-op latency" >&2; return 2; }

    start=$(date +%s)
    banner "running"
    time "${DARSHAN_ENV[@]}" "$TORCHRUN" --nproc_per_node="$NPROC" \
        --master_port="${MASTER_PORT:-29570}" \
        "$WORKLOAD_DIR/eval_load.py" \
        --ckpt "$ckpt" --model "$MODEL" --load-config "$LOAD_CONFIG" \
        --read-set eval --optimizer "$OPTIMIZER" --mode load --device "$DEVICE" \
        --only-layers "$ONLY_LAYERS" \
        --summary    "$RESULTS_DIR/$tag.summary.json" \
        --emit-reads "$RESULTS_DIR/$tag.reads.csv" || rc=$?

    if [[ $rc -eq 0 ]]; then
        banner "darshan"
        local logs="$RESULTS_DIR/$tag.darshanlogs"; mkdir -p "$logs"
        find "$DARSHAN_LOGS" -maxdepth 4 -name '*.darshan' -newermt "@$start" \
             -exec mv {} "$logs/" \; 2>/dev/null
        # --match on V2_CKPT, not the config dir: Darshan may record the path
        # with the symlink resolved (copy_F/...), which is still under V2_CKPT.
        "$PYTHON" "$WORKLOAD_DIR/parse_darshan.py" "$logs" \
            --ckpt "$ckpt" --match "$V2_CKPT/" \
            --emit-reads "$RESULTS_DIR/$tag.reads.csv" \
            --summary    "$RESULTS_DIR/$tag.summary.json" \
            --chunksize "$(numfmt --from=iec "${CHUNKSIZE^^}")" \
            --numtargets "$STRIPE_COUNT" \
            --target-map "$RESULTS_DIR/targets_cfg_$cfg.json" \
            --out "$RESULTS_DIR/$tag" || rc=$?
        [[ -s "$RESULTS_DIR/$tag.ops.csv" ]] || { echo "  no ops.csv: DXT trace missing" >&2; rc=1; }
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$rep" "$tag" "$start" "$(date +%s)" \
        "$([[ $rc -eq 0 ]] && echo ok || echo "fail($rc)")" >> "$INDEX"
    return $rc
}

banner "preflight"
require_fs_pools
require_throttles
LOG="$RESULTS_DIR/matrix_$(date +%Y%m%d_%H%M%S).log"
echo "  log: $LOG"
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!

for rep in $(seq 1 "$REPS"); do
    order=$(printf '%s\n' "${RUN_CONFIGS[@]}" | shuf)
    banner "rep $rep order: $(echo $order)"
    for cfg in $order; do
        if done_ok "$cfg" "$rep"; then echo "  [$cfg rep $rep] already done, skipping"; continue; fi
        # A failed read is recorded and the matrix continues; a missing
        # throttle or failed cache drop (rc 2) stops it, because every later run
        # would be invalid too.
        rc=0; run_one "$cfg" "$rep" || rc=$?
        if [[ $rc -eq 2 ]]; then echo "setup check failed; stopping" >&2; exit 1; fi
        if [[ $rc -ne 0 ]]; then echo "  [$cfg rep $rep] FAILED (rc=$rc); continuing" >&2; fi
    done
done

banner "matrix done"
column -t -s $'\t' "$INDEX"
echo
echo "next: ./06_analyze.sh"
# let tee drain before the shell prompt comes back, so the last lines do not
# print after it and look like a hang
exec >&- 2>&-
wait "$TEE_PID" 2>/dev/null || true
