#!/bin/bash
# 00_check_env.sh — preflight. Run this first, on anjuna2, and read the output.
#
# Every line here is a thing that silently produces wrong numbers if it is
# wrong. The st_blksize probe in particular decides whether the small-operation
# finding in METHODS.md 4.3 applies to this cluster at all.
set -uo pipefail
source "$(dirname "$0")/env.sh"

banner "host"
hostname; uname -r
echo "cores: $(nproc)"
free -h | sed -n '1,2p'

banner "python / torch"
"$PYTHON" -c "import sys,torch,numpy; print('python', sys.version.split()[0]);
print('torch ', torch.__version__); print('numpy ', numpy.__version__)" \
  || { echo "FAIL: need torch >= 2.6 and numpy"; exit 1; }
command -v "$TORCHRUN" >/dev/null && echo "torchrun: $(command -v "$TORCHRUN")" \
  || echo "FAIL: torchrun not on PATH"

banner "workload imports + torch compatibility"
# Developed against torch 2.9; anjuna2 runs 2.5.1. An import check alone is not
# enough -- it passed while Metadata(version=...) was still broken on 2.5, which
# only surfaced after a full populate. So this does a real end-to-end run:
# write a checkpoint, build its .metadata, and read it back with dcp.load.
# --fill sparse makes it instant and near-zero on disk; the point is the API
# surface, not the bytes.
PFDIR="$BEEGFS_ROOT/.preflight.$$"
"$PYTHON" - "$WORKLOAD_DIR" "$PFDIR" <<'EOF'
import contextlib, io, shutil, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
out = sys.argv[2]
import llama_spec, populate_checkpoint, eval_load, verify_checkpoint  # noqa
import parse_darshan, parse_eval_strace, capacity  # noqa
print("  imports OK")
populate_checkpoint.verify_templates()
print("  torch.save blob framing self-check OK")
try:
    from pathlib import Path
    import torch, torch.distributed.checkpoint as dcp
    m = llama_spec.preset("llama_tiny", 1)
    cfg = llama_spec.ParallelConfig.parse("TP2/PP1/DP1")
    with contextlib.redirect_stdout(io.StringIO()):   # progress bar is noise here
        info = populate_checkpoint.populate(m, cfg, Path(out), "sparse",
                                            optimizer="adam", layout="per-shard")
    print(f"  wrote {info['shard_files']} files + .metadata "
          f"({info['metadata_bytes']:,} B) on beegfs")
    sd = {s.fqn: torch.zeros(s.shape, dtype=torch.bfloat16)
          for s in m.tensors()}
    with contextlib.redirect_stderr(io.StringIO()):
        dcp.load(sd, checkpoint_id=out)
    print(f"  dcp.load read {len(sd)} tensors back  -- write+read path OK")
finally:
    shutil.rmtree(out, ignore_errors=True)
EOF
[[ $? -ne 0 ]] && echo "  FAIL: fix this before running anything else"
rm -rf "$PFDIR"

banner "BeeGFS mount"
mount | grep -E 'beegfs|fhgfs' || echo "no beegfs mount found"
stat -f "$BEEGFS_ROOT" 2>/dev/null || stat -f /mnt/beegfs
df -h "$(dirname "$BEEGFS_ROOT")" 2>/dev/null | tail -1

banner "st_blksize -- the buffered-read size CPython will use"
# This is the number that matters, and it is NOT the same call as `stat -f`:
# CPython sizes its BufferedReader from fstat().st_blksize on the FILE, while
# `stat -f` reports statfs().f_bsize on the FILESYSTEM. They usually agree on
# BeeGFS; this checks rather than assumes.
probe="$BEEGFS_ROOT/.blkprobe"
mkdir -p "$BEEGFS_ROOT"
dd if=/dev/urandom of="$probe" bs=1M count=8 status=none 2>/dev/null
if [[ -f "$probe" ]]; then
    "$PYTHON" - "$probe" <<'PY'
import os, sys
f = open(sys.argv[1], "rb")
bs = os.fstat(f.fileno()).st_blksize
print(f"  st_blksize = {bs:,} bytes")
print("  -> CPython buffered reads will be this size.")
if bs >= 262144:
    print("  -> LARGE. The <=64 KB read population in METHODS 4.3 will NOT")
    print("     appear here; that finding is ext4-specific. Expect fewer,")
    print("     bigger reads plus ~5-8% framing overhead instead.")
else:
    print("  -> SMALL. The METHODS 4.3 small-operation finding should")
    print("     reproduce, and the HDD/NVMe IOPS argument stays live.")
PY
    if command -v strace >/dev/null 2>&1; then
        echo "  actual syscall issued for a 100-byte read:"
        strace -e trace=read "$PYTHON" -c "open('$probe','rb').read(100)" 2>&1 \
            | grep -E 'read\(3' | head -3 | sed 's/^/    /'
    else
        echo "  (strace not installed; install it or rely on st_blksize above)"
    fi
    rm -f "$probe"
else
    echo "  FAIL: could not write to $BEEGFS_ROOT"
fi

banner "BeeGFS stripe pattern and pools"
if command -v beegfs-ctl >/dev/null 2>&1; then
    sudo beegfs-ctl --getentryinfo "$BEEGFS_ROOT" 2>/dev/null | sed 's/^/  /'
    echo "  --- pools ---"
    sudo beegfs-ctl --liststoragepools 2>/dev/null | grep -vE '^\||^-' | sed 's/^/  /'
    # Report what the pools actually contain rather than asserting it: the
    # membership changes, and a stale claim here is worse than none. Parsed in
    # python because the target list wraps across indented continuation lines,
    # which trips up a line-at-a-time awk (an earlier version silently
    # undercounted by one per line).
    echo "  --- pool sizes ---"
    sudo beegfs-ctl --liststoragepools 2>/dev/null | "$PYTHON" -c '
import re, sys
pools, cur = {}, None
for line in sys.stdin:
    if line.startswith("|") or "Pool ID" in line:
        continue
    if line.strip() and set(line.strip()) <= set("=- "):
        continue
    m = re.match(r"\s*(\d+)\s+(\S+)\s*(.*)$", line)
    if m:
        cur = m.group(1)
        pools[cur] = [m.group(2), set()]
        rest = m.group(3)
    elif cur and line.startswith(" "):
        rest = line
    else:
        continue
    pools[cur][1].update(re.findall(r"\d+", rest))
for pid in sorted(pools, key=int):
    desc, tg = pools[pid]
    print(f"  pool {pid:<3} {desc:<24} {len(tg):>3} targets"
          + (f"  ({min(tg, key=int)}..{max(tg, key=int)})" if tg else "  (empty)"))
'
    if [[ -z "$POOL_ID" ]]; then
      echo "  NOTE: POOL_ID unset -- files inherit the parent directory's pool."
      echo "        Fine for a baseline against whatever Default currently holds;"
      echo "        set it explicitly so the run is reproducible, and required"
      echo "        for a media comparison (the pool must be single-media)."
    fi
else
    echo "  beegfs-ctl not found"
fi

banner "Darshan"
setup_darshan && echo "  OK" || true

banner "what fits in this machine's RAM"
for rs in eval restart; do
    n=$(autosize_layers "$rs")
    echo "  $MODEL / $rs / $LOAD_CONFIG : $n layers"
done
echo
echo "  (--mode plan needs 0 GB and gives the full 80-layer access pattern;"
echo "   only --mode load, i.e. timing, is bounded by RAM)"

banner "preflight complete"
