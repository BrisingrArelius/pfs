#!/bin/bash
# 00_preflight.sh — check everything the matrix depends on, without changing
# anything. Run it before populating and again before each run session.
set -uo pipefail
source "$(dirname "$0")/env_v2.sh"
fail=0

banner "pools"
list_pools | "$PYTHON" "$V2_DIR/pools.py" table
require_fs_pools || fail=1
d=$(pool_id "$D_POOL_DESC")
if [[ -n "$d" ]]; then
    t=$(pool_targets "$d")
    echo "  pool $d ($D_POOL_DESC): ${t:-(empty)}"
    [[ -n "$t" ]] && { echo "  FAIL: $D_POOL_DESC still holds targets; 02_populate_D.sh did not restore" >&2; fail=1; }
fi

banner "throttles"
require_throttles || fail=1
"$V2_DIR/throttle.sh" show | sed 's/^/  /'

banner "cache drop on anjuna2 + OSS_HOSTS ($OSS_HOSTS)"
drop_caches_strict || fail=1

banner "darshan"
setup_darshan || fail=1
for t in darshan-parser darshan-dxt-parser; do
    command -v "$t" >/dev/null || { echo "  FAIL: $t not on PATH" >&2; fail=1; }
done

banner "checkpoint copies and configs"
for c in F S D; do
    p=$(copy_dir "$c")
    if [[ -e "$p/.metadata" ]]; then
        echo "  copy_$c: $(ls "$p"/*.distcp 2>/dev/null | wc -l) shard files, $(du -sh "$p" | cut -f1)"
    else
        echo "  copy_$c: not populated"
    fi
done
for c in "${CONFIGS[@]}"; do
    [[ -d "$(cfg_dir "$c")" ]] && echo "  cfg_$c: built" || echo "  cfg_$c: not built"
done

banner "RAM for the eval read"
echo "  ONLY_LAYERS=$ONLY_LAYERS; autosize says $(autosize_layers eval) fit"

banner "other users on the throttled targets"
# Anything else reading these SSDs shares the throttle budget and moves the
# numbers. Not a hard check: samples beegfs-storage's read rate over 5 s, which
# should be ~0 when nobody (including us) is reading.
for h in $OSS_HOSTS; do
    ssh -o BatchMode=yes "$h" bash -s <<'EOF' 2>/dev/null \
        | awk -v h="$h" '{print "  " h ": beegfs-storage reading " $1 " MiB/s"}'
f=/sys/fs/cgroup/system.slice/beegfs-storage.service/io.stat
r() { grep -o 'rbytes=[0-9]*' $f | cut -d= -f2 | awk '{s+=$1} END {print s+0}'; }
a=$(r); sleep 5; b=$(r)
echo $(( (b - a) / 5 / 1048576 ))
EOF
done

banner "preflight $([[ $fail -eq 0 ]] && echo OK || echo FAILED)"
exit $fail
