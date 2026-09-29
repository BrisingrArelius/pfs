#!/bin/bash
# 02_eval_read.sh — the measurement. An evaluation job reads the checkpoint.
#
# Four things make this an eval read rather than a restart; each is argued in
# ../METHODS.md with its citation or its label as an assumption:
#
#   --read-set eval      model weights only, no optimizer state
#   --load-config TP4    inference-shaped, and != the TP8/PP4 it was saved at
#   (in eval_load.py)    forward-pass stand-in, then exit, never writes back
#   frequency            handled by schedule_jobs.py, not here
#
# MODE=plan   full 80-layer access pattern, 0 GB RAM, no data files touched.
#             Use for file classification and the amplification sweep.
# MODE=load   real reads. Bounded by RAM, so --only-layers is autosized.
#             Use for timing and the HDD/NVMe pool comparison.
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODE="${MODE:-load}"
TAG="eval_${TAG_BASE}_$(echo "$LOAD_CONFIG" | tr '/' '-')_${MODE}_$(date +%H%M%S)"

banner "eval read: $MODE"
echo "  checkpoint $CKPT_DIR"
echo "  reader     $LOAD_CONFIG ($NPROC ranks, device=$DEVICE)"

EXTRA=()
if [[ "$MODE" == "load" ]]; then
    LAYERS="${ONLY_LAYERS:-$(autosize_layers eval)}"
    if [[ "$LAYERS" -eq 0 ]]; then
        echo "  Nothing fits in RAM. Use MODE=plan, or MODEL=llama31_8b." >&2
        exit 1
    fi
    EXTRA+=(--only-layers "$LAYERS")
    FULL=$("$PYTHON" -c "import sys; sys.path.insert(0,'$WORKLOAD_DIR'); \
import llama_spec; print(llama_spec.preset('$MODEL').num_hidden_layers)")
    if [[ "$LAYERS" -lt "${FULL:-0}" ]]; then
        echo "  layers     $LAYERS of $FULL  (RAM-limited -- a partial model;"
        echo "             say so alongside any number from this run)"
    else
        echo "  layers     $LAYERS of $FULL  (complete model)"
    fi
    banner "cold cache"
    drop_caches
fi

TRACER=()
DARSHAN_ENV=()
if [[ "$MODE" == "load" ]]; then
    banner "tracing"
    if setup_darshan; then :; else
        command -v strace >/dev/null && \
            TRACER=(strace -ff -ttt -e trace=openat,close,read,pread64,lseek
                    -o "$RESULTS_DIR/$TAG.strace")
    fi
fi

START_EPOCH=$(date +%s)
banner "running"
time "${TRACER[@]}" "${DARSHAN_ENV[@]}" "$TORCHRUN" --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29570}" \
    "$WORKLOAD_DIR/eval_load.py" \
    --ckpt "$CKPT_DIR" \
    --model "$MODEL" \
    --load-config "$LOAD_CONFIG" \
    --read-set eval \
    --optimizer "$OPTIMIZER" \
    --mode "$MODE" \
    --device "$DEVICE" \
    "${EXTRA[@]}" \
    --summary    "$RESULTS_DIR/$TAG.summary.json" \
    --emit-reads "$RESULTS_DIR/$TAG.reads.csv"

# Darshan writes one log per process (jobid-env=NONE -> pid in the name).
# Logs from concurrent runs share a directory, so select this run's by mtime.
if [[ "$MODE" == "load" && -n "${DARSHAN_RUNTIME_LIB:-}" ]]; then
    banner "darshan"
    RUNLOGS="$RESULTS_DIR/$TAG.darshanlogs"; mkdir -p "$RUNLOGS"
    find "$DARSHAN_LOGS" -maxdepth 4 -name '*.darshan' -newermt "@$START_EPOCH" \
         -exec mv {} "$RUNLOGS/" \; 2>/dev/null
    n=$(find "$RUNLOGS" -name '*.darshan' | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "  no logs produced. Darshan was preloaded but wrote nothing --" >&2
        echo "  usually an MPI build, or DARSHAN_LOG_DIR_PATH not honoured." >&2
    else
        echo "  $n log(s) in $RUNLOGS"
        "$PYTHON" "$WORKLOAD_DIR/parse_darshan.py" "$RUNLOGS" \
            --ckpt "$CKPT_DIR" \
            --emit-reads "$RESULTS_DIR/$TAG.reads.csv" \
            --summary    "$RESULTS_DIR/$TAG.summary.json" \
            --chunksize "$(numfmt --from=iec "${CHUNKSIZE^^}" 2>/dev/null || echo 524288)" \
            --numtargets "$STRIPE_COUNT" \
            ${TARGET_MAP:+--target-map "$TARGET_MAP"} \
            --out "$RESULTS_DIR/$TAG"
    fi
fi

# strace -ff writes one file per process as $TAG.strace.<pid>
if compgen -G "$RESULTS_DIR/$TAG.strace*" >/dev/null; then
    banner "syscall view"
    # Cross-check against DCP's own byte count. The syscall total must EXCEED
    # the logical total (zip framing); anything less means the trace dropped
    # syscalls and the numbers are not usable.
    EXPECT=$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['total_bytes'])" \
             "$RESULTS_DIR/$TAG.summary.json" 2>/dev/null || echo "")
    "$PYTHON" "$WORKLOAD_DIR/parse_eval_strace.py" \
        "$RESULTS_DIR/$TAG.strace" \
        ${EXPECT:+--expect-bytes "$EXPECT"} \
        --json "$RESULTS_DIR/$TAG.syscalls.json"
fi
echo; echo "results: $RESULTS_DIR/$TAG.*"
