#!/bin/bash
# reset_pools.sh
#
# Returns all targets to the Default pool (ID 1) and removes the hdd and ssd
# pools, restoring the original single-pool configuration.
#
# The Default pool is never deleted — targets are moved back into it with
# --addstoragepooltargets, then the now-empty hdd/ssd pools are removed.
#
# Run as root or with sudo.

set -euo pipefail

ALL_TARGETS="101,102,103,104,105,106,107,201,202,203,204,205,206,207,301,302,303,304,305,306,307,401,402,403,404,405,406,407"
DEFAULT_POOL_ID=1

# ---------------------------------------------------------------------------
# get_pool_id <description>
#
# Parses --liststoragepools to find the numeric ID for a given description.
# Handles variations in whitespace/column alignment. Skips header lines.
# Prints the ID, or an empty string if not found.
# ---------------------------------------------------------------------------
get_pool_id() {
    local desc="$1"
    sudo beegfs-ctl --liststoragepools 2>/dev/null \
        | awk -v d="$desc" '
            /^[0-9]+/ {
                pool_id = $1
                sub(/^[0-9]+[[:space:]]+/, "")
                if ($0 == d) {
                    print pool_id
                    exit
                }
            }
        '
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=== Returning all targets to Default pool (ID ${DEFAULT_POOL_ID}) ==="
if ! sudo beegfs-ctl --addstoragepooltargets --poolid="${DEFAULT_POOL_ID}" --targets="${ALL_TARGETS}" 2>&1; then
    echo "ERROR: failed to move targets back to Default pool." >&2
    echo "Some targets may still be assigned to hdd/ssd pools." >&2
    exit 1
fi

echo ""
echo "=== Removing hdd and ssd pools ==="
for desc in hdd ssd; do
    POOL_ID=$(get_pool_id "$desc")
    if [ -n "${POOL_ID}" ]; then
        echo "  Removing pool '${desc}' (ID ${POOL_ID})..."
        if ! sudo beegfs-ctl --removestoragepool --poolid="${POOL_ID}" 2>&1; then
            echo "  WARNING: failed to remove pool '${desc}' (ID ${POOL_ID})." >&2
            echo "  Pool may still have assigned targets or other constraints." >&2
        fi
    else
        echo "  Pool '${desc}' not found — skipping."
    fi
done

echo ""
echo "=== Final pool configuration ==="
sudo beegfs-ctl --liststoragepools
