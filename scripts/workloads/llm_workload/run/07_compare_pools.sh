#!/usr/bin/env bash
# 07_compare_pools.sh — the three-pool comparison figures (C1/C2/C3).
#
#   ./07_compare_pools.sh [OUT_DIR]
#
# Picks the newest eval-read run out of each pool's results directory and
# hands the three prefixes to compare_pools.py. Override any of the three by
# exporting PREFIX_DEFAULT / PREFIX_HDD / PREFIX_SSD to a full prefix first --
# useful when a pool has several runs and "newest" is not the one you want.
#
# Like 06_plot.sh this needs matplotlib and pandas, so it usually runs where
# the plotting happens rather than on anjuna2. C3 additionally needs
# `darshan-parser` on PATH and the harvested PREFIX.darshanlogs directories,
# which only exist on the cluster -- run it there if you want that panel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

OUT="${1:-$RESULTS_DIR/pool_compare}"

# Newest run matching a pool's tag, by mtime. Keyed off .ops.csv rather than
# .darshan.json because .ops.csv is what the figures actually need; a run whose
# DXT came back empty should not be silently selected.
newest() {
    local dir="$1" pat="$2" hit
    hit=$(ls -t "$dir"/$pat 2>/dev/null | head -1) || true
    [ -n "$hit" ] && printf '%s' "${hit%.ops.csv}"
}

ROOT="${BEEGFS_ROOT}"
: "${PREFIX_DEFAULT:=$(newest "$ROOT/results"       '*_pool1_TP2_load_*.ops.csv')}"
: "${PREFIX_HDD:=$(newest     "$ROOT/results_pool3" '*_pool3_TP2_load_*.ops.csv')}"
: "${PREFIX_SSD:=$(newest     "$ROOT/results_pool6" '*_pool6_TP2_load_*.ops.csv')}"

args=()
[ -n "${PREFIX_DEFAULT:-}" ] && args+=(--run "Default (pool 1)=$PREFIX_DEFAULT")
[ -n "${PREFIX_HDD:-}" ]     && args+=(--run "HDD (pool 3)=$PREFIX_HDD")
[ -n "${PREFIX_SSD:-}" ]     && args+=(--run "SSD (pool 6)=$PREFIX_SSD")

if [ ${#args[@]} -eq 0 ]; then
    echo "found no *_TP2_load_*.ops.csv under $ROOT/results{,_pool3,_pool6}" >&2
    echo "  set PREFIX_DEFAULT / PREFIX_HDD / PREFIX_SSD by hand" >&2
    exit 1
fi

banner "comparing $(( ${#args[@]} / 2 )) runs -> $OUT"
for ((i = 0; i < ${#args[@]}; i += 2)); do echo "  ${args[i+1]}"; done

"$PYTHON" "$WORKLOAD_DIR/compare_pools.py" "${args[@]}" --out-dir "$OUT"
