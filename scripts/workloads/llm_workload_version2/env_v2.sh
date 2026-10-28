#!/bin/bash
# env_v2.sh — settings for the throttled F/S rank-balance experiment. Source it;
# do not execute it. Everything not set here comes from ../llm_workload/run/env.sh
# (drop_caches, setup_darshan, set_stripe, ...), which this sources at the end.
#
# The experiment matrix is defined ONCE, here (CONFIGS / PLACEMENT). The config
# builder writes it to configs.json and the analysis reads it from there.

V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_RUN="$(cd "$V2_DIR/../llm_workload/run" && pwd)"

# --- workload (same checkpoint shape as the earlier eval-read runs) ------
export MODEL="${MODEL:-llama31_8b}"
export SAVE_CONFIG="${SAVE_CONFIG:-TP4/PP2/DP1}"
export LOAD_CONFIG="${LOAD_CONFIG:-TP2}"
export OSS_HOSTS="${OSS_HOSTS:-colva1 colva2 colva3 colva4}"
# Pinned rather than autosized, so every run reads exactly the same bytes. 32 is
# the whole llama31_8b, and is what the earlier runs on anjuna2 got.
export ONLY_LAYERS="${ONLY_LAYERS:-32}"
export STRIPE_COUNT="${STRIPE_COUNT:-4}"
export CHUNKSIZE="${CHUNKSIZE:-512k}"

# --- pools and throttle classes -----------------------------------------
# Must agree with throttle.sh. 206 and 207 are deliberately in neither.
export F_POOL="${F_POOL:-11}"           # llm_F
export S_POOL="${S_POOL:-12}"           # llm_S
export D_POOL_DESC="${D_POOL_DESC:-llm_D}"
F_TARGETS="104 105 205 305 404 405"
S_TARGETS="106 107 306 307 406 407"
F_RBPS=20971520                          # 20 MiB/s
S_RBPS=5242880                           #  5 MiB/s

# --- layout on BeeGFS ---------------------------------------------------
# Checkpoint copies and config dirs live under V2_CKPT; results go elsewhere.
# V2_CKPT is also the Darshan --match filter, so it must NOT contain the
# results dir (the reader writes its summary/reads files there).
export BEEGFS_ROOT="${BEEGFS_ROOT:-/mnt/beegfs/$USER/ckpt_experiment}"
export V2_CKPT="${V2_CKPT:-$BEEGFS_ROOT/llmv2_ckpt}"
export RESULTS_DIR="${RESULTS_DIR:-$BEEGFS_ROOT/llmv2_results}"
copy_dir() { echo "$V2_CKPT/copy_$1"; }      # copy_dir F|S|D
cfg_dir()  { echo "$V2_CKPT/cfg_$1"; }       # cfg_dir allF|...|D

# --- the experiment matrix ----------------------------------------------
# Which copy each file __N_0.distcp comes from, for N = 0..7.
# Read order (from the earlier traces): rank 0 reads 4,5,0,1; rank 1 reads 6,7,2,3.
#
#   config       rank 0 (4 5 0 1)   rank 1 (6 7 2 3)
#   allF         F F F F            F F F F
#   allS         S S S S            S S S S
#   unbalanced   F F F F            S S S S
#   balanced     F F S S            F F S S
#   onefast      F S S S            S S S S
#   D            mixed pool         mixed pool
CONFIGS=(allF allS unbalanced balanced onefast D)
declare -A PLACEMENT=(
    #             files: 0 1 2 3 4 5 6 7
    [allF]="F F F F F F F F"
    [allS]="S S S S S S S S"
    [unbalanced]="F F S S F F S S"
    [balanced]="S S S S F F F F"
    [onefast]="S S S S F S S S"
    [D]="D D D D D D D D"
)

source "$V1_RUN/env.sh"

# --- helpers ------------------------------------------------------------
csv() { echo "$*" | tr ' ' ','; }

list_pools() { sudo beegfs-ctl --liststoragepools 2>/dev/null; }
pool_targets() { list_pools | "$PYTHON" "$V2_DIR/pools.py" targets "$1"; }
pool_id() { list_pools | "$PYTHON" "$V2_DIR/pools.py" id "$1"; }

# Ordered target IDs of one file. The target lines are indented
# ("  + 105 @ colva1"), which is what ../llm_workload/run/make_target_map.sh
# gets wrong. readlink -f resolves a config dir's symlink to the real file.
file_targets() {
    sudo beegfs-ctl --getentryinfo "$(readlink -f "$1")" 2>/dev/null \
        | grep -oP '^\s*\+\s*\K[0-9]+(?=\s*@)' | paste -sd' ' -
}

# Abort unless F_POOL / S_POOL hold exactly the F / S targets.
require_fs_pools() {
    local f s
    f=$(pool_targets "$F_POOL"); s=$(pool_targets "$S_POOL")
    if [[ "$f" != "$(tr ' ' '\n' <<<"$F_TARGETS" | sort -n | paste -sd' ')" ||
          "$s" != "$(tr ' ' '\n' <<<"$S_TARGETS" | sort -n | paste -sd' ')" ]]; then
        echo "  FAIL: pool $F_POOL has [$f], want [$F_TARGETS]" >&2
        echo "        pool $S_POOL has [$s], want [$S_TARGETS]" >&2
        return 1
    fi
    echo "  pool $F_POOL (F): $f"
    echo "  pool $S_POOL (S): $s"
}

# Abort unless all 12 throttles are in place. They are --runtime, so a reboot
# or a beegfs-storage restart on any colva silently removes them.
require_throttles() {
    local out nf ns
    out=$("$V2_DIR/throttle.sh" show 2>/dev/null)
    nf=$(grep -o "rbps=$F_RBPS\b" <<<"$out" | wc -l)
    ns=$(grep -o "rbps=$S_RBPS\b" <<<"$out" | wc -l)
    if [[ $nf -ne 6 || $ns -ne 6 ]]; then
        echo "  FAIL: throttles: $nf/6 F (rbps=$F_RBPS), $ns/6 S (rbps=$S_RBPS)." >&2
        echo "        Re-apply with ./throttle.sh apply" >&2
        return 1
    fi
    echo "  throttles: 6 F + 6 S devices limited"
}

# drop_caches from env.sh only warns. Here a failed drop on any host aborts,
# because a warm OSS serves the read from RAM and skips the throttle entirely.
drop_caches_strict() {
    local out
    out=$(drop_caches 2>&1); echo "$out"
    if grep -q WARNING <<<"$out"; then
        [[ -n "${ALLOW_WARM:-}" ]] && { echo "  ALLOW_WARM set; continuing warm" >&2; return 0; }
        echo "  FAIL: cache drop did not succeed everywhere (ALLOW_WARM=1 to ignore)" >&2
        return 1
    fi
}
