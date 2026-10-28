#!/bin/bash
# 01_populate_FS.sh — write the F and S copies of the checkpoint (~80 GB each)
# into pools F_POOL and S_POOL. Skips a copy that already exists.
#
# The writer is ../llm_workload/run/01_populate.sh, unchanged: it sets the
# stripe pattern + pool on the empty directory, writes, verifies, and prints the
# placement of one file. The fill is seeded (seed 0), so all copies are
# byte-identical, .metadata included -- that is what lets a config dir mix
# files from different copies under one .metadata.
#
# Writes are not throttled. The verify step reads back a sample of tensors, which
# is throttled, so the S copy's verify takes a few minutes.
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"

banner "pool check"
require_fs_pools

mkdir -p "$V2_CKPT"
for c in ${COPIES:-F S}; do
    case $c in F) pool=$F_POOL ;; S) pool=$S_POOL ;; *) echo "COPIES: F and/or S" >&2; exit 2 ;; esac
    POOL_ID=$pool CKPT_DIR="$(copy_dir "$c")" TAG_BASE="llmv2_copy_$c" \
        bash "$V1_RUN/01_populate.sh"
done
