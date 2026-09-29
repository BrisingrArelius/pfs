#!/usr/bin/env bash
# 06_plot.sh -- draw the figure set for one parsed run.
#
# Separate from 02/03 on purpose: plotting needs matplotlib/pandas, which the
# cluster node may not have, while parsing needs the Darshan tools, which your
# laptop does not. The parsed CSVs are small (a few MB), so the normal workflow
# is to parse on the cluster and plot wherever matplotlib lives:
#
#   # on your laptop
#   scp -r anjuna2:/mnt/beegfs/pfs/ckpt_experiment/results/TAG.* ./results/
#   ./run/06_plot.sh results/TAG
#
# Usage: 06_plot.sh PREFIX [OUT_DIR]
#   PREFIX is the results prefix without extension, i.e. what `parse_darshan.py
#   --out` was given: .../eval_llama31_8b_adam_per-rank_pool1_TP2_load_183817
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD_DIR="$(dirname "$HERE")"
PYTHON="${PYTHON:-python3}"

[ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 2; }
PREFIX="${1%.ops.csv}"; PREFIX="${PREFIX%.}"
OUT="${2:-${PREFIX}.plots}"

if [ ! -f "$PREFIX.ops.csv" ]; then
    echo "no $PREFIX.ops.csv" >&2
    echo "  DXT must have been on when the run was traced, and parse_darshan.py" >&2
    echo "  must have been given --out $PREFIX." >&2
    exit 1
fi
"$PYTHON" - <<'PY' || { echo "install: pip install --user matplotlib pandas numpy" >&2; exit 1; }
import importlib, sys
missing = [m for m in ("matplotlib", "pandas", "numpy")
           if not importlib.util.find_spec(m)]
if missing:
    print("missing: " + ", ".join(missing), file=sys.stderr); sys.exit(1)
PY

"$PYTHON" "$WORKLOAD_DIR/plot_io.py" --prefix "$PREFIX" --out-dir "$OUT" \
    --tag "$(basename "$PREFIX")" "${@:3}"
echo
echo "figures: $OUT"
