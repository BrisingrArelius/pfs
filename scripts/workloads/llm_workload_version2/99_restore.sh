#!/bin/bash
# 99_restore.sh — put the cluster back the way it was before the experiment
# (pool state recorded in experiment_context.md):
#   - throttles cleared on colva1-4
#   - 104 105 106 107 205 305 306 307 back in pool 6 (ssd)
#   - 404 405 406 407 back in Default (pool 1), which is where removed targets go
#   - the now-empty llm_F / llm_S / llm_D pools removed (REMOVE_POOLS=0 keeps them)
# The checkpoint copies are NOT deleted; see the runbook for that.
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"
SSD_POOL="${SSD_POOL:-6}"

if [[ -z "${YES:-}" ]]; then
    read -r -p "  Clear throttles and dissolve pools $F_POOL/$S_POOL? [y/N] " a
    [[ $a == [yY] ]] || exit 1
fi

banner "throttles"
"$V2_DIR/throttle.sh" clear

banner "pools"
D_POOL=$(pool_id "$D_POOL_DESC")
for p in $F_POOL $S_POOL $D_POOL; do
    t=$(pool_targets "$p")
    [[ -n "$t" ]] && sudo beegfs-ctl --modifystoragepool --id="$p" --rmtargets="$(csv $t)"
done
sudo beegfs-ctl --modifystoragepool --id="$SSD_POOL" --addtargets=104,105,106,107,205,305,306,307
if [[ "${REMOVE_POOLS:-1}" == 1 ]]; then
    for p in $F_POOL $S_POOL $D_POOL; do
        sudo beegfs-ctl --removestoragepool --id="$p" || true
    done
fi
list_pools | "$PYTHON" "$V2_DIR/pools.py" table
echo
echo "  expected: 1 Default 401-407 | 3 hdd | 4 offline_pool 207 | 6 ssd 104-107 205-206 305-307"
