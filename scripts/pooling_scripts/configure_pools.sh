#!/bin/bash
# configure_pools.sh
#
# Splits BeeGFS storage targets into two pools:
#   hdd  — targets x01-x04 across nodes 1-4
#   ssd  — targets x05-x07 across nodes 1-4
#
# Idempotent: skips pool creation if a pool with that description already exists.
# Targets are added to their pools with --addstoragepooltargets, which also
# removes them from whichever pool currently holds them (including Default).
# The Default pool (ID 1) is left intact — it will simply have no targets
# assigned once both moves complete.
#
# Run as root or with sudo.

set -euo pipefail

HDD_TARGETS="101,102,103,104,201,202,203,204,301,302,303,304,401,402,403,404"
SSD_TARGETS="105,106,107,205,206,207,305,306,307,405,406,407"

# ---------------------------------------------------------------------------
# get_pool_id <description>
#
# Parses the output of --liststoragepools to find the numeric ID of the pool
# whose description matches the given string.
#
# --liststoragepools output format (BeeGFS 7.x):
#   Pool ID  Description
#   -------  -----------
#   1        Default
#   2        hdd
#   3        ssd
#
# Handles variations in whitespace/column alignment. Skips header lines
# (anything before the first line containing only a Pool ID number).
#
# Prints the ID, or an empty string if not found.
# ---------------------------------------------------------------------------
get_pool_id() {
    local desc="$1"
    sudo beegfs-ctl --liststoragepools 2>/dev/null \
        | awk -v d="$desc" '
            /^[[:space:]]*[0-9]+/ {
                # Pool ID is first field
                pool_id = $1
                # Description is second field (skip leading spaces)
                pool_desc = $2
                # Trim whitespace from description
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", pool_desc)
                if (pool_desc == d) {
                    print pool_id
                    exit
                }
            }
        '
}

# ---------------------------------------------------------------------------
# ensure_pool <description>
#
# Creates a storage pool with the given description if one does not already
# exist.  Prints the pool ID either way.
# ---------------------------------------------------------------------------
ensure_pool() {
    local desc="$1"
    local id
    id=$(get_pool_id "$desc")
    if [ -z "$id" ]; then
        echo "  Creating pool '${desc}'..." >&2
        if ! sudo beegfs-ctl --addstoragepool --desc="$desc" 2>&1 >&2; then
            echo "ERROR: failed to create pool '${desc}'." >&2
            exit 1
        fi
        # Brief pause to allow BeeGFS to register the new pool
        sleep 1
        id=$(get_pool_id "$desc")
        if [ -z "$id" ]; then
            echo "ERROR: pool '${desc}' was created but could not read its ID." >&2
            exit 1
        fi
    else
        echo "  Pool '${desc}' already exists (ID ${id}) — skipping creation." >&2
    fi
    echo "$id"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=== Ensuring HDD pool exists ==="
HDD_POOL_ID=$(ensure_pool "hdd")
echo "HDD pool ID: ${HDD_POOL_ID}"

echo ""
echo "=== Ensuring SSD pool exists ==="
SSD_POOL_ID=$(ensure_pool "ssd")
echo "SSD pool ID: ${SSD_POOL_ID}"

echo ""
echo "=== Assigning HDD targets (${HDD_TARGETS}) to pool ${HDD_POOL_ID} ==="
if ! sudo beegfs-ctl --modifystoragepool --id="${HDD_POOL_ID}" --addtargets="${HDD_TARGETS}" 2>&1; then
    echo "ERROR: failed to assign HDD targets to pool ${HDD_POOL_ID}." >&2
    echo "Verify that all target IDs in HDD_TARGETS exist and are accessible." >&2
    exit 1
fi

echo ""
echo "=== Assigning SSD targets (${SSD_TARGETS}) to pool ${SSD_POOL_ID} ==="
if ! sudo beegfs-ctl --modifystoragepool --id="${SSD_POOL_ID}" --addtargets="${SSD_TARGETS}" 2>&1; then
    echo "ERROR: failed to assign SSD targets to pool ${SSD_POOL_ID}." >&2
    echo "Verify that all target IDs in SSD_TARGETS exist and are accessible." >&2
    exit 1
fi

echo ""
echo "=== Final pool configuration ==="
sudo beegfs-ctl --liststoragepools
