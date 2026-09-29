#!/bin/bash
# 01_populate.sh — write the checkpoint the reading jobs will read. Run once
# per (layout, pool) combination you want to compare.
#
# Sizes with --optimizer adam, Llama-3.1-70B:
#     141 GB  bf16 weights
#     564 GB  fp32 Adam moments
#     706 GB  total
# Your free space is ~37 TB, so this is ~2%. --optimizer none drops it to
# 141 GB but then the eval job has nothing to skip and the headline
# eval-vs-restart comparison disappears.
#
# Deliberately NOT traced with Darshan. The writer here is our streaming
# writer, not a real training job, so its write pattern is an artefact and a
# trace of it would be misleading. Only the reads in 02/03 are real DCP.
set -euo pipefail
source "$(dirname "$0")/env.sh"

banner "populate $CKPT_DIR"
echo "  model $MODEL  save $SAVE_CONFIG  optimizer $OPTIMIZER"
echo "  layout $LAYOUT${THREAD_COUNT:+ x$THREAD_COUNT}  fill $FILL"

if [[ -e "$CKPT_DIR/.metadata" ]]; then
    echo "  checkpoint already exists. Delete it or change TAG_BASE to rebuild."
    exit 0
fi

# Pattern must be set on the directory BEFORE any file is created in it.
banner "stripe pattern"
set_stripe "$CKPT_DIR"

banner "writing"
time "$PYTHON" "$WORKLOAD_DIR/populate_checkpoint.py" \
    --model "$MODEL" \
    --save-config "$SAVE_CONFIG" \
    --optimizer "$OPTIMIZER" \
    --layout "$LAYOUT" --thread-count "$THREAD_COUNT" \
    --fill "$FILL" \
    --workers "$(nproc)" \
    --out "$CKPT_DIR" \
    --manifest "$RESULTS_DIR/manifest_${TAG_BASE}.json"

banner "verify"
# --skip-crc reads every byte back; on 706 GB over 2.5 GbE that is ~40 minutes.
# Worth it once per layout, skippable on repeats.
"$PYTHON" "$WORKLOAD_DIR/verify_checkpoint.py" --ckpt "$CKPT_DIR" \
    --max-tensors 24 ${SKIP_CRC:+--skip-crc}

banner "on-disk placement"
if command -v beegfs-ctl >/dev/null 2>&1; then
    f=$(find "$CKPT_DIR" -name '*.distcp' | head -1)
    sudo beegfs-ctl --getentryinfo "$f" | sed 's/^/  /'
fi
du -sh "$CKPT_DIR"
