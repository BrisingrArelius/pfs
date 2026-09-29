#!/usr/bin/env python3
"""
parse_eval_strace.py — turn an strace of an eval load into a per-file I/O
profile, at the syscall level.

Why this exists alongside `eval_load.py --emit-reads`
-----------------------------------------------------
Those two views disagree, and the disagreement is the point.

`--emit-reads` reports DCP's *logical* requests: one per storage item, at the
`(offset, length)` recorded in `.metadata`. That is the request stream a
checkpoint-system paper describes.

This script reports what the kernel actually saw. They differ because every
storage item is a `torch.save` zip archive, and `torch.load` parses that zip's
local headers and central directory through a buffered reader before it touches
the payload. So one 27 MB logical request becomes one large payload read plus a
tail of small framing reads, and a lot of `lseek`.

That gap matters here for two reasons. Darshan records POSIX-level operations,
so a Darshan log of this workload will show the small reads, not the logical
ones -- and the context notes' tf-Darshan warning (Cluster'20) about filtering
DL traces by request size applies directly. And a placement policy keyed on
"average request size" will read this workload completely differently depending
on which of the two views it is fed.

Use -ff, not -f
---------------
With `-f` strace writes every process into one file and splits concurrent
syscalls across lines as `<unfinished ...>` / `<... resumed>` pairs. PyTorch's
loader is threaded, so a large fraction of reads arrive that way. A naive
line-at-a-time parser silently drops them -- when this was first written it
recovered 7% of the bytes DCP actually read, which is worse than no measurement
because it looks plausible.

`-ff` writes one file per process/thread, with no pid prefix and no splitting.
That is what the run scripts use. This parser accepts either form, and still
reassembles unfinished/resumed pairs as a safety net.

Usage
-----
    strace -ff -ttt -e trace=openat,close,read,pread64,lseek \\
           -o eval.strace  <torchrun ... eval_load.py ...>
    python3 parse_eval_strace.py eval.strace          # picks up eval.strace.*

Always cross-check the byte total against `eval_load.py --summary`, which
reports DCP's own view. The syscall total should EXCEED the logical total (by
the zip-framing overhead); if it is lower, the trace is incomplete.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

RE_OPEN = re.compile(r'openat\([^,]+,\s*"([^"]+)"[^)]*\)\s*=\s*(\d+)')
RE_CLOSE = re.compile(r"close\((\d+)\)\s*=\s*0")
RE_READ = re.compile(r"\b(p?read(?:64)?)\((\d+),.*?\)\s*=\s*(-?\d+)")
RE_LSEEK = re.compile(r"lseek\((\d+),")
RE_PID = re.compile(r"^(\d+)\s+")
# split-syscall forms produced by -f (but not -ff)
RE_UNFIN = re.compile(r"\b(\w+)\((\d+)[,)].*<unfinished \.\.\.>")
RE_RESUMED = re.compile(r"<\.\.\. (\w+) resumed>.*?\)\s*=\s*(-?\d+)")


def _new_entry():
    return {"opens": 0, "reads": 0, "bytes": 0, "lseeks": 0, "zero_reads": 0,
            "sizes": collections.Counter()}


def expand_inputs(paths: list[Path]) -> list[Path]:
    """Accept `eval.strace` and pick up the `-ff` siblings `eval.strace.<pid>`."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.strace*")))
            continue
        sibs = sorted(p.parent.glob(p.name + ".*"))
        if sibs:
            out.extend(sibs)
        if p.exists():
            out.append(p)
    if not out:
        raise SystemExit(f"no trace files found for {[str(p) for p in paths]}")
    return out


def parse(paths, match: str):
    """Track (pid, fd) -> filename so reused descriptors are never conflated."""
    fds: dict[tuple[str, str], str] = {}
    pending: dict[str, tuple[str, str]] = {}   # pid -> (syscall, fd)
    per_file = collections.defaultdict(_new_entry)
    stats = {"lines": 0, "resumed": 0, "files": 0}

    for path in paths:
        stats["files"] += 1
        # with -ff the pid is the filename suffix and lines carry no prefix
        suffix = path.name.rsplit(".", 1)[-1]
        file_pid = suffix if suffix.isdigit() else "0"

        for line in open(path, errors="ignore"):
            stats["lines"] += 1
            pid_m = RE_PID.match(line)
            pid = pid_m.group(1) if pid_m else file_pid

            m = RE_UNFIN.search(line)
            if m:
                pending[pid] = (m.group(1), m.group(2))
                continue

            m = RE_RESUMED.search(line)
            if m:
                stats["resumed"] += 1
                call, ret = m.group(1), int(m.group(2))
                got = pending.pop(pid, None)
                if got and got[0] == call and (pid, got[1]) in fds:
                    name = fds[(pid, got[1])]
                    if call.startswith(("read", "pread")):
                        e = per_file[name]
                        e["reads"] += 1
                        if ret > 0:
                            e["bytes"] += ret
                            e["sizes"][ret] += 1
                        elif ret == 0:
                            e["zero_reads"] += 1
                    elif call == "lseek":
                        per_file[name]["lseeks"] += 1
                continue

            m = RE_OPEN.search(line)
            if m:
                name, fd = m.group(1), m.group(2)
                if match in name:
                    base = Path(name).name
                    fds[(pid, fd)] = base
                    per_file[base]["opens"] += 1
                else:
                    fds.pop((pid, fd), None)
                continue

            m = RE_CLOSE.search(line)
            if m:
                fds.pop((pid, m.group(1)), None)
                continue

            m = RE_READ.search(line)
            if m:
                key = (pid, m.group(2))
                if key in fds:
                    n = int(m.group(3))
                    e = per_file[fds[key]]
                    e["reads"] += 1
                    if n > 0:
                        e["bytes"] += n
                        e["sizes"][n] += 1
                    elif n == 0:
                        e["zero_reads"] += 1
                continue

            m = RE_LSEEK.search(line)
            if m and (pid, m.group(1)) in fds:
                per_file[fds[(pid, m.group(1))]]["lseeks"] += 1
    return per_file, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strace", type=Path, nargs="+",
                    help="trace file, -ff prefix, or directory")
    ap.add_argument("--match", default=".distcp",
                    help="substring identifying checkpoint files (also matches .metadata)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--expect-bytes", type=int, default=None,
                    help="DCP's logical byte total; warns if the trace is short")
    args = ap.parse_args()

    inputs = expand_inputs(args.strace)
    shard, stats = parse(inputs, args.match)
    meta, _ = parse(inputs, ".metadata")

    def roll(d):
        return {
            "files": len(d),
            "opens": sum(v["opens"] for v in d.values()),
            "reads": sum(v["reads"] for v in d.values()),
            "bytes": sum(v["bytes"] for v in d.values()),
            "lseeks": sum(v["lseeks"] for v in d.values()),
            "zero_reads": sum(v["zero_reads"] for v in d.values()),
        }

    s, m = roll(shard), roll(meta)
    sizes = collections.Counter()
    for v in shard.values():
        sizes.update(v["sizes"])

    print(f"=== syscall view: {stats['files']} trace file(s), "
          f"{stats['lines']:,} lines, {stats['resumed']:,} resumed ===")
    print(f"shard files touched   {s['files']:,}  ({s['opens']:,} opens)")
    print(f"read syscalls         {s['reads']:,}   ({s['zero_reads']:,} returned 0 = EOF probes)")
    print(f"bytes read            {s['bytes']/1e9:.3f} GB")
    print(f"lseek syscalls        {s['lseeks']:,}")
    if s["reads"]:
        print(f"mean read size        {s['bytes']/max(s['reads']-s['zero_reads'],1)/1e3:.1f} KB")
    print(f"\n.metadata             {m['opens']} opens, {m['reads']} reads, "
          f"{m['bytes']/1e6:.2f} MB")

    small = sum(c for n, c in sizes.items() if n <= 65536)
    tot = sum(sizes.values())
    print(f"\nreads <= 64 KB        {small:,} of {tot:,} ({100*small/max(tot,1):.1f}% of reads)")
    print(f"bytes in those        "
          f"{sum(n*c for n, c in sizes.items() if n <= 65536)/1e6:.2f} MB "
          f"({100*sum(n*c for n,c in sizes.items() if n<=65536)/max(s['bytes'],1):.2f}% of bytes)")
    print("\ntop read sizes:")
    for n, c in sizes.most_common(10):
        print(f"   {n:>12,} B   x{c:,}")

    if args.expect_bytes:
        ratio = s["bytes"] / max(args.expect_bytes, 1)
        print(f"\nvs DCP logical    {args.expect_bytes/1e9:.3f} GB "
              f"-> syscall/logical = {ratio:.2f}x")
        if ratio < 0.98:
            print("  *** TRACE IS INCOMPLETE. Syscall bytes should EXCEED the")
            print("      logical total (zip framing). Re-run strace with -ff.")

    if args.json:
        args.json.write_text(json.dumps(
            {"shard": s, "metadata": m,
             "size_histogram": {str(k): v for k, v in sorted(sizes.items())}}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
