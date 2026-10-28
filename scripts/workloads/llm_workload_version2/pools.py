#!/usr/bin/env python3
"""
pools.py — parse `beegfs-ctl --liststoragepools` from stdin.

    sudo beegfs-ctl --liststoragepools | ./pools.py targets 11   -> "104 105 205 305 404 405"
    sudo beegfs-ctl --liststoragepools | ./pools.py id llm_D     -> "13"   (empty if absent)
    sudo beegfs-ctl --liststoragepools | ./pools.py table

The target list wraps onto indented continuation lines and the output is
prefixed by the Enterprise-feature banner, so a line-at-a-time awk gets this
wrong (the same parser as ../llm_workload/run/00_check_env.sh).
"""
import re
import sys


def parse(lines):
    pools, cur = {}, None
    for line in lines:
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
        pools[cur][1].update(int(t) for t in re.findall(r"\d+", rest))
    return pools


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pools = parse(sys.stdin)
    cmd = sys.argv[1]
    if cmd == "targets":
        print(" ".join(str(t) for t in sorted(pools.get(sys.argv[2], ["", set()])[1])))
    elif cmd == "id":
        print(next((pid for pid, (d, _) in pools.items() if d == sys.argv[2]), ""))
    elif cmd == "table":
        for pid in sorted(pools, key=int):
            desc, tg = pools[pid]
            print(f"  pool {pid:<3} {desc:<24} {' '.join(map(str, sorted(tg))) or '(empty)'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
