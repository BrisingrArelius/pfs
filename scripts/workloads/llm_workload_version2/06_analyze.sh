#!/bin/bash
# 06_analyze.sh — average the matrix over reps and draw the comparison plots.
#
#   ./06_analyze.sh                 # reads $RESULTS_DIR, writes $RESULTS_DIR/analysis
#   ./06_analyze.sh DIR [OUT_DIR]   # e.g. a copy scp'd to a laptop:
#       scp -r anjuna2:/mnt/beegfs/pfs/ckpt_experiment/llmv2_results ./
#       ./06_analyze.sh llmv2_results
#
# Needs pandas + matplotlib only (no Darshan tools: parsing already happened in
# 05_run_matrix.sh). Safe to run mid-matrix; it uses whatever runs are done.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
if [[ $# -ge 1 ]]; then
    RES="$1"
else
    RES="${RESULTS_DIR:-/mnt/beegfs/$USER/ckpt_experiment/llmv2_results}"
fi
"$PYTHON" -c "import pandas, matplotlib" 2>/dev/null \
    || { echo "install: pip install --user pandas matplotlib" >&2; exit 1; }
"$PYTHON" "$HERE/analyze_matrix.py" --results "$RES" ${2:+--out "$2"}
