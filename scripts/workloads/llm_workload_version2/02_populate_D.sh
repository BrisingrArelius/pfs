#!/bin/bash
# 02_populate_D.sh — write the "default pool" copy: one checkpoint striped
# across a pool holding all 12 F + S targets, so each file gets a random mix.
#
# A target can be in only one pool, so this temporarily empties F_POOL and
# S_POOL into a third pool (llm_D), writes copy_D there, and moves the targets
# back. An EXIT trap does the move-back even if the populate fails. Existing
# files never move when pool membership changes, so copy_F and copy_S keep
# their placement throughout, and copy_D keeps its mix afterwards.
#
# While this runs (~7-10 min), new files anyone creates in llm_F / llm_S have
# no targets to go to. Agree the window with the other users.
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"

CKPT_D="$(copy_dir D)"
if [[ -e "$CKPT_D/.metadata" ]]; then
    echo "  $CKPT_D already exists. Delete it to rebuild."; exit 0
fi

banner "pool check"
require_fs_pools
ALL="$F_TARGETS $S_TARGETS"

if [[ -z "${YES:-}" ]]; then
    echo
    echo "  This will move targets $(csv $ALL) out of pools $F_POOL and $S_POOL"
    echo "  into '$D_POOL_DESC', write copy_D, then move them back."
    read -r -p "  Proceed? [y/N] " a; [[ $a == [yY] ]] || exit 1
fi

D_POOL=""
restore() {
    local rc=$?
    banner "restore F and S pools"
    if [[ -n "$D_POOL" ]]; then
        sudo beegfs-ctl --modifystoragepool --id="$D_POOL" --rmtargets="$(csv $ALL)" || true
    fi
    sudo beegfs-ctl --modifystoragepool --id="$F_POOL" --addtargets="$(csv $F_TARGETS)" || true
    sudo beegfs-ctl --modifystoragepool --id="$S_POOL" --addtargets="$(csv $S_TARGETS)" || true
    list_pools | "$PYTHON" "$V2_DIR/pools.py" table
    require_fs_pools || echo "  !! F/S pools NOT restored -- fix by hand before running" >&2
    exit $rc
}
trap restore EXIT

banner "build $D_POOL_DESC"
# Removed targets go to Default (pool 1); adding them to another pool takes them
# from there.
sudo beegfs-ctl --modifystoragepool --id="$F_POOL" --rmtargets="$(csv $F_TARGETS)"
sudo beegfs-ctl --modifystoragepool --id="$S_POOL" --rmtargets="$(csv $S_TARGETS)"
D_POOL=$(pool_id "$D_POOL_DESC")
if [[ -z "$D_POOL" ]]; then
    sudo beegfs-ctl --addstoragepool --desc="$D_POOL_DESC" --targets="$(csv $ALL)"
    D_POOL=$(pool_id "$D_POOL_DESC")
else
    # left over (empty) from an earlier run of this script
    sudo beegfs-ctl --modifystoragepool --id="$D_POOL" --addtargets="$(csv $ALL)"
fi
[[ -n "$D_POOL" ]] || { echo "  FAIL: could not create $D_POOL_DESC" >&2; exit 1; }
echo "  $D_POOL_DESC is pool $D_POOL: $(pool_targets "$D_POOL")"

# The metadata server learns pool membership from mgmtd asynchronously. A file
# created right after the pool change fails with EREMOTEIO ("Remote I/O
# error"), because the meta server still sees the pool as having no targets.
# Probe with a real create in a dir pinned to the pool until one succeeds.
banner "wait for the metadata server to see pool $D_POOL"
probe="$V2_CKPT/.probe_D"
rm -rf "$probe"
POOL_ID=$D_POOL set_stripe "$probe" >/dev/null
ok=""
for i in $(seq 1 60); do
    if dd if=/dev/zero of="$probe/f" bs=1M count=1 status=none 2>/dev/null; then
        ok=1; echo "  file create in pool $D_POOL works (after $(( (i - 1) * 5 )) s)"
        echo "  probe file targets: $(file_targets "$probe/f")"
        break
    fi
    rm -f "$probe/f"; sleep 5
done
rm -rf "$probe"
[[ -n "$ok" ]] || { echo "  FAIL: still cannot create files in pool $D_POOL after 5 min" >&2; exit 1; }

banner "populate copy_D"
# leftovers from a failed attempt (no .metadata, so not a finished copy)
rm -rf "$CKPT_D"
mkdir -p "$V2_CKPT"
POOL_ID=$D_POOL CKPT_DIR="$CKPT_D" TAG_BASE=llmv2_copy_D \
    bash "$V1_RUN/01_populate.sh"
# restore() runs on exit. The empty llm_D pool is left in place; 99_restore.sh
# can remove it.
