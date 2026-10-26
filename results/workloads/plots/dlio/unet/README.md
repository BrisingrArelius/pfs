# U-Net 3D: which pool leads?

Source: [nine-run U-Net sweep](../../../runs/runs-20261026-164248-962399/)
and its [native-output analysis](../../../runs/runs-20261026-164248-962399/analysis/report.md).
The [pool-level values](comparison.csv) and [three values per pool](comparison_runs.csv)
accompany the figures.

![U-Net pool comparison](pool_comparison.png)

**Reading this figure:** Default and HDD training-input proxy rates are
effectively tied (271.06 versus 270.99 nominal MiB/s). SSD's lower mean
(265.11) is influenced by one unusually long epoch-1 run (107.25 s).
HDD has the numerically highest rank-0 checkpoint-save mean (38.48 MiB/s),
but there are only three runs. Default and HDD whole-run times are also
nearly identical (~7.38 min). Stars indicate **best observed means**, not
statistically established winners.

U-Net's DLIO 2.0.0 native training metric is absent in epochs 1, 3, and 5.
The input-rate bar is explicitly a **nominal proxy**: processed samples ×
configured mean NPZ record length / training-block time where available,
or epoch wall time in checkpoint-free epochs. It is not measured read bytes.
Checkpoint bars use one rank's file bytes/save duration: **no `fsync`, no
eight-rank aggregate timer**.

Supporting plots:

- [Training across epochs](training_epochs.png) — wall times and available
  native epoch-2/4 sample rates; missing epochs are not plotted as zero.
- [Checkpoint events](checkpoint_rank0.png) — duration and rank-0 rate at
  epochs 2 and 4.
- [Three repetitions](training_repetitions.png) — mean epoch wall time for
  each run in its execution order.
- [Rank loading tail](rank_load_p95.png) — application-level step load times,
  not POSIX read latency.
- [Input file-to-OST assignments](dataset_ost_assignments.png) — file counts,
  not bytes or device traffic.
