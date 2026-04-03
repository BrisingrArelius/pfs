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
#   --beegfs            Run BeeGFS cluster benchmark (Client mode)
#   --ost               Run raw storage target benchmark (Storage mode)
#   --hdd-dir PATH      Override HDD mount point  (default: /mnt/beegfs/advay/hdd)
#   --ssd-dir PATH      Override SSD mount point  (default: /mnt/beegfs/advay/ssd)
#   --hdd-ost-dir PATH  Override HDD OST base mount (default: /mnt/hdd)
#   --ssd-ost-dir PATH  Override SSD OST base mount (default: /mnt/nvme)
#   --results-dir PATH  Override results directory (default: ./results or ./ost_results)
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
HDD_OST_DIR="/mnt/hdd"
SSD_OST_DIR="/mnt/nvme"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

RUN_BEEGFS=false
RUN_OST=false
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
        --beegfs)        RUN_BEEGFS=true; shift ;;
        --ost)           RUN_OST=true; shift ;;
        --hdd-dir)       HDD_DIR="$2"; shift 2 ;;
        --ssd-dir)       SSD_DIR="$2"; shift 2 ;;
        --hdd-ost-dir)   HDD_OST_DIR="$2"; shift 2 ;;
        --ssd-ost-dir)   SSD_OST_DIR="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --no-drop-cache) DROP_CACHE=false; shift ;;
        -h|--help)
            sed -n '14,35p' "$0" | sed 's/^# \?//'
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
    local label="$1"       # hdd | ssd (job file to run)
    local dir="$2"         # mount point
    local out_prefix="$3"  # override output name (e.g. hdd_ost1)

    local jobfile="${SCRIPT_DIR}/${label}.fio"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Starting benchmark: ${out_prefix}"
    echo "  Target directory  : ${dir}"
    echo "  Job file          : ${jobfile}"
    echo "  Results           : ${RESULTS_DIR}/${out_prefix}_${TS}.json"
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

    # Use the real user's name even when invoked with sudo, so the scratch
    # dir is named consistently (fio_scratch_pfs not fio_scratch_root).
    local real_user="${SUDO_USER:-${USER}}"
    local fio_work_dir="${dir}/fio_scratch_${real_user}"
    mkdir -p "${fio_work_dir}"
    FIO_SCRATCH_DIRS+=("${fio_work_dir}")

    drop_caches

    local out_json="${RESULTS_DIR}/${out_prefix}_${TS}.json"
    local out_txt="${RESULTS_DIR}/${out_prefix}_${TS}.txt"

    echo "  Running fio... (60s per job + 5s ramp — 4 jobs × 65s ≈ 4-5 min total)"
    echo "  Live status every 30s will appear below."
    echo ""

    # --output-format=normal without --output=FILE: all fio output goes to
    # stdout, which tee forwards to both the terminal and the txt file.
    # (Using --output=FILE silently swallows everything including status-interval.)
    fio "${jobfile}" \
        --directory="${fio_work_dir}" \
        --output-format=normal \
        --status-interval=30 \
        2>&1 | tee "${out_txt}"

    # Append a clean summary extracted from the txt output
    if command -v python3 &>/dev/null; then
        python3 - "${out_txt}" <<'PYEOF' | tee -a "${out_txt}" || true
import sys, re
txt = open(sys.argv[1]).read()
# Extract per-job bw/iops lines printed by fio normal format
print("\n\u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
for line in txt.splitlines():
    if any(k in line for k in ["READ:", "WRITE:", "read:", "write:", "IOPS", "BW"]):
        print(" ", line.strip())
PYEOF
    fi

    echo ""
    echo "  Results saved:"
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

if [[ "${RUN_BEEGFS}" == "false" && "${RUN_OST}" == "false" ]]; then
    echo "ERROR: You must specify --beegfs or --ost mode"
    exit 1
fi

echo "=== FIO Storage Hardware Benchmark ==="
echo "  Timestamp : ${TS}"

if [[ "${RUN_OST}" == "true" ]]; then
    # In OST mode, we use ost_results and loop through the local disk paths
    if [[ "${RESULTS_DIR}" == "${SCRIPT_DIR}/results" ]]; then
        RESULTS_DIR="${SCRIPT_DIR}/ost_results"
    fi
    mkdir -p "${RESULTS_DIR}"
    chmod 777 "${RESULTS_DIR}" 2>/dev/null || true

    echo "  Mode      : OST (Bare Metal)"
    echo "  Results   : ${RESULTS_DIR}"

    # Loop all 4 HDDs
    for i in {1..4}; do
        run_benchmark "hdd" "${HDD_OST_DIR}${i}" "hdd_ost${i}"
    done

    # Loop all 3 NVMe SSDs
    for i in {1..3}; do
        run_benchmark "ssd" "${SSD_OST_DIR}${i}" "ssd_ost${i}"
    done
fi

if [[ "${RUN_BEEGFS}" == "true" ]]; then
    # In BeeGFS mode, we use the standard results dir
    mkdir -p "${RESULTS_DIR}"
    chmod 777 "${RESULTS_DIR}" 2>/dev/null || true

    echo "  Mode      : BeeGFS (Client)"
    echo "  Results   : ${RESULTS_DIR}"

    run_benchmark "hdd" "${HDD_DIR}" "hdd"
    run_benchmark "ssd" "${SSD_DIR}" "ssd"
fi

echo ""
echo "=== All benchmarks complete ==="
echo "  Results written to: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/"
