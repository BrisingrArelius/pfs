#!/bin/bash
# 04_make_configs.sh — build one directory per configuration: .metadata plus
# 8 symlinks, each pointing at the copy PLACEMENT (env_v2.sh) assigns it. DCP
# opens files by name relative to the checkpoint dir, so the reader follows the
# symlinks without knowing about them.
#
# Also writes, to $RESULTS_DIR:
#   configs.json            the matrix + F/S target sets, read by the analysis
#   targets_cfg_<cfg>.json  per-config target map for parse_darshan.py
# Needs 03_placement.sh to have run first (it reads targets_copy_*.json).
set -euo pipefail
source "$(dirname "$0")/env_v2.sh"

banner ".metadata equivalent across copies"
# Not a byte cmp: populate records each copy's own directory in
# storage_meta.checkpoint_id, so the files always differ there. The reader never
# uses that field (FileSystemReader resolves files from its own path). Every
# other field -- tensor shapes, and each shard's file/offset/length -- must match.
ref="$(copy_dir F)/.metadata"
for c in S D; do
    m="$(copy_dir "$c")/.metadata"
    [[ -e $m ]] || { echo "  copy_$c missing"; continue; }
    "$PYTHON" - "$ref" "$m" "$c" <<'EOF' || exit 1
import pickle, sys, warnings
warnings.simplefilter("ignore")
a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
ida = getattr(a.storage_meta, "checkpoint_id", None)
idb = getattr(b.storage_meta, "checkpoint_id", None)
a.storage_meta = b.storage_meta = None
if a != b:
    for f in vars(a):
        if getattr(a, f) != getattr(b, f):
            print(f"  FAIL: copy_{sys.argv[3]}/.metadata differs from copy_F's in '{f}'; "
                  "configs would be invalid", file=sys.stderr)
    sys.exit(1)
print(f"  copy_{sys.argv[3]}: same as copy_F ({len(a.storage_data)} storage entries); "
      f"only checkpoint_id differs ({idb})")
EOF
done

banner "config dirs"
for cfg in "${CONFIGS[@]}"; do
    read -ra p <<<"${PLACEMENT[$cfg]}"
    [[ -e "$(copy_dir "${p[0]}")/.metadata" ]] || { echo "  cfg_$cfg: copy_${p[0]} missing, skipped"; continue; }
    d="$(cfg_dir "$cfg")"
    rm -rf "$d"; mkdir -p "$d"
    # .metadata from the copy file 0 comes from; all copies' are identical
    ln -s "$(copy_dir "${p[0]}")/.metadata" "$d/.metadata"
    for i in 0 1 2 3 4 5 6 7; do
        ln -s "$(copy_dir "${p[i]}")/__${i}_0.distcp" "$d/__${i}_0.distcp"
    done
    echo "  cfg_$cfg  files 0-7: ${PLACEMENT[$cfg]}"
done

banner "configs.json + per-config target maps"
# Pass the matrix to python as "cfg=placement" args.
args=(); for cfg in "${CONFIGS[@]}"; do args+=("$cfg=${PLACEMENT[$cfg]}"); done
"$PYTHON" - "$RESULTS_DIR" "$F_TARGETS" "$S_TARGETS" "$F_RBPS" "$S_RBPS" "${args[@]}" <<'EOF'
import json, sys
from pathlib import Path
res, F, S, frate, srate, *cfgs = sys.argv[1:]
res = Path(res)
copies = {c: json.load(open(p)) for c in "FSD" if (p := res / f"targets_copy_{c}.json").exists()}
out = {"order": [], "placement": {}, "targets": {"F": [int(t) for t in F.split()],
                                                 "S": [int(t) for t in S.split()]},
       "rbps": {"F": int(frate), "S": int(srate)},
       "rank_files": {"0": [4, 5, 0, 1], "1": [6, 7, 2, 3]}}
for spec in cfgs:
    cfg, place = spec.split("=")
    place = place.split()
    out["order"].append(cfg)
    out["placement"][cfg] = {f"__{i}_0.distcp": p for i, p in enumerate(place)}
    if not all(p in copies for p in place):
        print(f"  {cfg}: no target map (run 03_placement.sh); skipped"); continue
    tmap = {".metadata": copies[place[0]][".metadata"]}
    tmap.update({f"__{i}_0.distcp": copies[p][f"__{i}_0.distcp"] for i, p in enumerate(place)})
    (res / f"targets_cfg_{cfg}.json").write_text(json.dumps(tmap, indent=1))
    print(f"  targets_cfg_{cfg}.json")
(res / "configs.json").write_text(json.dumps(out, indent=1))
print(f"  {res / 'configs.json'}")
EOF

banner "sanity: what each config's files resolve to"
for cfg in "${CONFIGS[@]}"; do
    d="$(cfg_dir "$cfg")"; [[ -d $d ]] || continue
    printf '  %-11s' "$cfg"
    for i in 0 1 2 3 4 5 6 7; do
        printf ' %s' "$(basename "$(dirname "$(readlink -f "$d/__${i}_0.distcp")")" | sed 's/copy_//')"
    done
    echo
done
