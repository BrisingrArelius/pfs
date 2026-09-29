#!/bin/bash
# make_target_map.sh CKPT_DIR > targets.json
#
# Emits {"__0_0.distcp": [401,403,405,406], ...} by asking BeeGFS which targets
# each file actually landed on. parse_darshan.py needs this to turn stripe slots
# into real target IDs; without it, slot 0 of two files is usually two different
# disks and the per-target concurrency numbers are meaningless.
#
# On a per-shard layout this is ~14,000 beegfs-ctl calls and takes a while. It
# only needs redoing when the checkpoint is rewritten.
set -euo pipefail
CKPT="${1:?usage: make_target_map.sh CKPT_DIR}"
first=1
echo "{"
for f in "$CKPT"/*.distcp; do
    ids=$(sudo beegfs-ctl --getentryinfo "$f" 2>/dev/null \
          | awk '/^\+ *[0-9]+ @/ {print $2}' | paste -sd, -)
    [[ -z "$ids" ]] && continue
    [[ $first -eq 0 ]] && echo ","
    first=0
    printf '  "%s": [%s]' "$(basename "$f")" "$ids"
done
echo
echo "}"
