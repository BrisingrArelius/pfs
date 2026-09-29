#!/usr/bin/env python3
"""
schedule_jobs.py — emit a job mix whose eval:restart ratio matches production.

The frequency knob
------------------
A single eval read tells you the access pattern. It does not tell you how much
of the filesystem's year that pattern accounts for -- and that is the claim the
project actually rests on ("eval is the dominant checkpoint reader, restart is
the wrong optimisation target"). That claim needs a rate, and the rate is the
one number here taken directly from production rather than modelled.

[CITED] ByteCheckpoint (Wan et al., NSDI'25, arXiv:2407.20143) reports a
six-month census of checkpoint-loading events across ByteDance's training
fleet:

    pre-training resumption      1,870
    cross-stage (SFT / RL)      13,080
    evaluation                  19,844
                                ------
    total                       34,794

Evaluation outnumbers resumption 10.6 : 1. Over 182.6 days that is about 109
eval loads and 10 resumptions per day.

[ASSUMPTION] Two things this script decides that the paper does not state:

  * how eval events map onto checkpoint *generations*. We fire evaluation
    against every generation, because that is what an eval harness wired to a
    training run does; the paper gives totals, not coverage. `--eval-coverage`
    changes it.
  * what to do with cross-stage (SFT / RL) events. They are neither of the two
    jobs this harness runs, so by default they are excluded and the headline
    ratio is evaluation against pre-training resumption, 19,844 : 1,870 =
    10.61 : 1. Folding them in either direction changes that number a lot --
    `--crossstage-as restart` gives 1.33 : 1, `--crossstage-as eval` gives
    17.61 : 1 -- so the choice is exposed rather than buried, and any result
    should say which was used. Whether an SFT or RL warm start actually reads
    optimizer state is pipeline-dependent; some resume it, some start fresh.

Running the real ratio means 34,794 jobs. `--scale` compresses the campaign
while preserving the ratio, which is the part that matters.

anjuna2 has no batch scheduler, so the generated driver runs jobs back to back
in the shuffled order rather than submitting them. The interleaving is the
point -- it is what puts eval and restart reads in contention for the same
targets, which a run of all-evals-then-all-restarts would not.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

# ByteCheckpoint NSDI'25, six months of production.
CENSUS = {"resumption": 1870, "crossstage": 13080, "evaluation": 19844}
CENSUS_DAYS = 182.6


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1 / 200,
                    help="fraction of the real event count to actually submit")
    ap.add_argument("--generations", type=int, default=20,
                    help="checkpoint generations the training run produced")
    ap.add_argument("--eval-coverage", type=float, default=1.0,
                    help="fraction of generations each eval wave covers")
    ap.add_argument("--crossstage-as", choices=["exclude", "restart", "eval"],
                    default="exclude",
                    help="treat cross-stage (SFT/RL) events as a third workload "
                         "(default), or fold them into either class")
    ap.add_argument("--spacing-min", type=float, default=1.0,
                    help="minutes to pause between jobs (0 = back to back)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    n_eval = CENSUS["evaluation"]
    n_restart = CENSUS["resumption"]
    if args.crossstage_as == "restart":
        n_restart += CENSUS["crossstage"]
    elif args.crossstage_as == "eval":
        n_eval += CENSUS["crossstage"]

    s_eval = max(1, round(n_eval * args.scale))
    s_restart = max(1, round(n_restart * args.scale))

    rng = random.Random(args.seed)
    jobs = (["eval"] * s_eval) + (["restart"] * s_restart)
    rng.shuffle(jobs)

    rows = []
    n_cov = max(1, math.ceil(args.generations * args.eval_coverage))
    for i, kind in enumerate(jobs):
        # Eval sweeps generations round-robin so every generation is covered;
        # restarts land on the newest generation, which is what a resuming run
        # actually reloads.
        gen = (i % n_cov) if kind == "eval" else args.generations - 1
        rows.append({"idx": i, "kind": kind, "generation": gen,
                     "begin_offset_min": round(i * args.spacing_min, 2),
                     "script": "02_eval_read.sh" if kind == "eval"
                               else "03_restart_read.sh"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "schedule.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    sh_path = args.out_dir / "submit_schedule.sh"
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n# generated by schedule_jobs.py -- do not edit\n")
        f.write(f"# eval:restart = {s_eval}:{s_restart} "
                f"({n_eval/n_restart:.2f}:1, ByteCheckpoint NSDI'25 census)\n")
        f.write("#\n# No scheduler on anjuna2, so these run sequentially in the\n"
                "# shuffled order. Run under nohup/tmux -- this takes hours.\n")
        f.write("set -uo pipefail\nHERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n")
        f.write(f'RUN="${{RUN_DIR:-$HERE}}"\nSPACING={int(args.spacing_min * 60)}\n\n')
        f.write(f'echo "campaign: {len(rows)} jobs "\n\n')
        for r in rows:
            f.write(f'echo "[{r["idx"]+1}/{len(rows)}] {r["kind"]} '
                    f'gen={r["generation"]}"\n')
            f.write(f'CKPT_GENERATION={r["generation"]} MODE=load '
                    f'"$RUN/{r["script"]}" || echo "  job {r["idx"]} FAILED"\n')
            if args.spacing_min:
                f.write('sleep "$SPACING"\n')
            f.write("\n")
    sh_path.chmod(0o755)

    span_h = rows[-1]["begin_offset_min"] / 60  # pauses only; jobs add their own
    print(f"census (6 months, ByteCheckpoint NSDI'25):")
    for k, v in CENSUS.items():
        print(f"    {k:<14} {v:>7,}   ({v/CENSUS_DAYS:6.1f}/day)")
    print(f"\ncross-stage counted as: {args.crossstage_as}")
    print(f"real ratio              eval:restart = {n_eval:,}:{n_restart:,} "
          f"= {n_eval/n_restart:.2f}:1")
    print(f"scaled by {args.scale:g}         {s_eval}:{s_restart} "
          f"= {s_eval/s_restart:.2f}:1")
    print(f"generations covered     {n_cov} of {args.generations}")
    print(f"inter-job pauses        {span_h:.1f} h total at {args.spacing_min} min each")
    print(f"                        (plus each job's own runtime -- run under tmux)")
    print(f"\nwrote {csv_path}\nwrote {sh_path}")


if __name__ == "__main__":
    main()
