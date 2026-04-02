#!/bin/bash
# run_fio.sh
#
# Runs raw hardware benchmarks against BeeGFS HDD and SSD storage pools
# using fio.  This is a "disk-free" test — it targets the storage hardware
# directly by writing to the PFS mount point with O_DIRECT, bypassing both
# the Linux page cache and any application-level instrumentation (Darshan).
#
# Prerequisites:
#   - fio installed  (apt install fio / yum install fio)
#   - BeeGFS pools configured  (see ../pooling_scripts/configure_pools.sh)
#   - HDD and SSD pools mounted and accessible at HDD_DIR / SSD_DIR below
#   - Sufficient free space: default file layout is 8 GiB × 4 jobs = 32 GiB
#     per pool (adjust filesize= in the .fio files if needed)
#
# Usage:
#   ./run_fio.sh [OPTIONS]
#
# Options:
#   --hdd-only          Run HDD benchmark only
#   --ssd-only          Run SSD benchmark only
#   --hdd-dir PATH      Override HDD mount point  (default: /mnt/beegfs/advay/hdd)
#   --ssd-dir PATH      Override SSD mount point  (default: /mnt/beegfs/advay/ssd)
#   --results-dir PATH  Override results directory (default: ./results)
#   --no-drop-cache     Skip dropping the page cache before each run
#   -h, --help          Show this help
#
# Results are written to:
#   <results-dir>/hdd_<timestamp>.json
#   <results-dir>/ssd_<timestamp>.json
#   <results-dir>/hdd_<timestamp>.txt    (human-readable summary)
#   <results-dir>/ssd_<timestamp>.txt
#
# Example — run both pools, custom mount points:
#   sudo ./run_fio.sh \
#       --hdd-dir /mnt/beegfs/advay/hdd \
#       --ssd-dir /mnt/beegfs/advay/ssd

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — edit here if your mount points differ
# ---------------------------------------------------------------------------
HDD_DIR="/mnt/beegfs/advay/hdd"
SSD_DIR="/mnt/beegfs/advay/ssd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

RUN_HDD=true
RUN_SSD=true
DROP_CACHE=true

# Tracks every PFS scratch dir created so the trap can clean them all up
FIO_SCRATCH_DIRS=()

# ---------------------------------------------------------------------------
# Trap — guaranteed cleanup of PFS scratch dirs on exit, error, or Ctrl-C
# ---------------------------------------------------------------------------
cleanup_scratch() {
    for d in "${FIO_SCRATCH_DIRS[@]-}"; do
        if [[ -n "${d}" && -d "${d}" ]]; then
            echo "  [trap] Removing scratch dir: ${d}" >&2
            rm -rf "${d}" || true
        fi
    done
}
trap cleanup_scratch EXIT INT TERM

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hdd-only)      RUN_SSD=false; shift ;;
        --ssd-only)      RUN_HDD=false; shift ;;
        --hdd-dir)       HDD_DIR="$2"; shift 2 ;;
        --ssd-dir)       SSD_DIR="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --no-drop-cache) DROP_CACHE=false; shift ;;
        -h|--help)
            sed -n '14,36p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '$1'" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TS="$(date '+%Y%m%d_%H%M%S')"

check_dep() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found. Install it and retry." >&2
        exit 1
    fi
}

drop_caches() {
    if [[ "${DROP_CACHE}" == "true" ]]; then
        echo "  Dropping page/dentry/inode caches..."
        sync
        if sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
            echo "  Caches dropped."
        else
            echo "  WARNING: Could not drop caches (no sudo?). Results may include cached data." >&2
        fi
        sleep 3   # brief stabilisation pause
    fi
}

run_benchmark() {
    local label="$1"     # hdd | ssd
    local dir="$2"       # mount point
    local jobfile="${SCRIPT_DIR}/${label}.fio"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Starting ${label^^} benchmark"
    echo "  Target directory : ${dir}"
    echo "  Job file         : ${jobfile}"
    echo "  Results          : ${RESULTS_DIR}/${label}_${TS}.json"
    echo "════════════════════════════════════════════════════════"

    if [[ ! -f "${jobfile}" ]]; then
        echo "ERROR: Job file not found: ${jobfile}" >&2
        exit 1
    fi

    if [[ ! -d "${dir}" ]]; then
        echo "ERROR: Target directory does not exist: ${dir}" >&2
        echo "  Make sure the BeeGFS pool is mounted at that path." >&2
        exit 1
    fi

    # Create a user-namespaced scratch dir so parallel users don't collide.
    # Registered in FIO_SCRATCH_DIRS so the EXIT trap cleans it up even on crash.
    local fio_work_dir="${dir}/fio_scratch_${USER}"
    mkdir -p "${fio_work_dir}"
    FIO_SCRATCH_DIRS+=("${fio_work_dir}")

    drop_caches

    local out_json="${RESULTS_DIR}/${label}_${TS}.json"
    local out_txt="${RESULTS_DIR}/${label}_${TS}.txt"

    echo "  Running fio... (runtime=60s per job + 5s ramp — ~${#}4 jobs × 65s ≈ 4-5 min total)"
    echo "  Live status every 30s will appear below."
    echo "  Full output → ${out_json}"
    echo ""

    fio "${jobfile}" \
        --directory="${fio_work_dir}" \
        --output-format=json+,normal \
        --output="${out_json}" \
        --status-interval=30 \
        |& tee "${out_txt}"

    local bw_read  bw_write  iops_read  iops_write
    # Parse aggregate bw/iops from the JSON for a quick summary line
    if command -v python3 &>/dev/null; then
        read -r bw_read bw_write iops_read iops_write < <(python3 - <<'EOF'
import sys, json, pathlib, os
f = os.environ.get("_FIO_JSON_PATH", "")
try:
    data = json.loads(pathlib.Path(f).read_text())
    jobs = data.get("jobs", [])
    br = sum(j["read"]["bw"] for j in jobs)
    bw_r = sum(j["write"]["bw"] for j in jobs)
    ir = sum(j["read"]["iops"] for j in jobs)
    iw = sum(j["write"]["iops"] for j in jobs)
    print(f"{br} {bw_r} {ir} {iw}")
except Exception:
    print("0 0 0 0")
EOF
        )
        _FIO_JSON_PATH="${out_json}" python3 - <<'PYEOF' | tee -a "${out_txt}" || true
import sys, json, pathlib, os
f = os.environ.get("_FIO_JSON_PATH","")
try:
    data = json.loads(pathlib.Path(f).read_text())
    print("\n── Aggregate summary ──────────────────────────────")
    for j in data.get("jobs",[]):
        r = j["read"];  w = j["write"]
        print(f"  [{j['jobname']:20s}]  "
              f"read: {r['bw']/1024:7.1f} MiB/s  {r['iops']:8.0f} IOPS  "
              f"write: {w['bw']/1024:7.1f} MiB/s  {w['iops']:8.0f} IOPS")
except Exception as e:
    print(f"  (summary parse error: {e})")
PYEOF
    fi

    echo ""
    echo "  Results saved:"
    echo "    JSON : ${out_json}"
    echo "    Text : ${out_txt}"

    # Normal (non-crash) cleanup of PFS scratch files
    echo "  Cleaning up fio scratch files..."
    rm -rf "${fio_work_dir}"
    # Remove from the trap list so it doesn't double-delete
    FIO_SCRATCH_DIRS=("${FIO_SCRATCH_DIRS[@]/"${fio_work_dir}"/}")
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
check_dep fio

mkdir -p "${RESULTS_DIR}"

echo "=== FIO Storage Hardware Benchmark ==="
echo "  Timestamp : ${TS}"
echo "  Results   : ${RESULTS_DIR}"

[[ "${RUN_HDD}" == "true" ]] && run_benchmark "hdd" "${HDD_DIR}"
[[ "${RUN_SSD}" == "true" ]] && run_benchmark "ssd" "${SSD_DIR}"

echo ""
echo "=== All benchmarks complete ==="
echo "  Results written to: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/"
