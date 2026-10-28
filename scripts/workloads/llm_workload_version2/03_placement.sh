#!/bin/bash
# 03_placement.sh — record where every file of each copy actually landed, and
# check it. Writes $RESULTS_DIR/targets_copy_{F,S,D}.json:
#     {"__0_0.distcp": [105, 404, 205, 104], ..., ".metadata": [...]}
# (ordered stripe targets, as parse_darshan.py --target-map wants them).
#
# Fails if any F file touches a non-F target, any S file a non-S target, or any
# file is not striped 4 wide. For D it reports how many files have no S target
# (expected ~3%), because those are the only D files that can read at F speed.
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"

for c in ${COPIES:-F S D}; do
    dir="$(copy_dir "$c")"
    [[ -e "$dir/.metadata" ]] || { echo "  copy_$c not populated; skipping"; continue; }
    out="$RESULTS_DIR/targets_copy_$c.json"
    banner "copy_$c -> $out"
    {
        echo "{"
        sep=""
        for f in "$dir"/.metadata "$dir"/*.distcp; do
            printf '%s  "%s": [%s]' "$sep" "$(basename "$f")" "$(file_targets "$f" | tr ' ' ',')"
            sep=$',\n'
        done
        echo; echo "}"
    } > "$out"
    "$PYTHON" - "$out" "$c" "$F_TARGETS" "$S_TARGETS" "$STRIPE_COUNT" <<'EOF'
import json, sys
tmap = json.load(open(sys.argv[1]))
copy, F, S, width = sys.argv[2], set(map(int, sys.argv[3].split())), set(map(int, sys.argv[4].split())), int(sys.argv[5])
allowed = {"F": F, "S": S, "D": F | S}[copy]
bad = 0
for f, t in tmap.items():
    nf, ns = len(set(t) & F), len(set(t) & S)
    flags = []
    if not t: flags.append("NO TARGETS (getentryinfo failed?)")
    if f != ".metadata" and len(t) != width: flags.append(f"stripe width {len(t)} != {width}")
    if set(t) - allowed: flags.append(f"targets outside copy_{copy}: {sorted(set(t) - allowed)}")
    bad += bool(flags)
    print(f"  {f:<16} {str(t):<26} F:{nf} S:{ns}  {'; '.join(flags) or 'ok'}")
if copy == "D":
    shards = [t for f, t in tmap.items() if f != ".metadata"]
    allf = sum(1 for t in shards if not set(t) & S)
    print(f"  D: {allf}/{len(shards)} files have no S target (only those can read at F speed)")
sys.exit(1 if bad else 0)
EOF
done
