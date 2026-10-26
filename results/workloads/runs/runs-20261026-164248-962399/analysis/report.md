# unet3d_v100 · DLIO pool comparison

Nine runs: three per pool, eight MPI ranks per run.

| Pool | Mean epoch wall (s) | Epoch 1 wall (s) | Native epochs 2/4 samples/s | Checkpoint 2 rank-0 (MiB/s) | Checkpoint 4 rank-0 (MiB/s) |
|---|---:|---:|---:|---:|---:|
| 1 | 88.01 ± 0.01 | 83.09 ± 0.01 | 1.94 ± 0.00 | 37.88 ± 0.72 | 36.47 ± 1.12 |
| 3 | 88.04 ± 0.03 | 83.19 ± 0.10 | 1.94 ± 0.00 | 37.40 ± 2.42 | 39.68 ± 0.04 |
| 6 | 89.94 ± 2.77 | 91.42 ± 13.71 | 1.94 ± 0.00 | 38.70 ± 3.13 | 37.43 ± 1.56 |

Mean ± sample SD across three independent runs. Epoch wall time includes
checkpoints in epochs 2 and 4; checkpoint rate is rank-0 save completion
without fsync, not durable or eight-rank aggregate throughput.
Modeled training MiB/s is samples/s × configured mean record length, not
measured POSIX read bytes or backend device traffic.
The pool order was Default → HDD → SSD each repetition; Default uses colva4
while HDD/SSD span colva1–3. These are configured-pool comparisons.

**U-Net DLIO 2.0.0 reporting gap:** training-block duration and throughput
are absent in epochs 1, 3, and 5 although every rank loaded the expected
number of batches. Its native five-epoch mean includes zeros for those
epochs and is not used here. Epoch wall times cover all five epochs; the
native rate column uses only epochs 2 and 4, not a five-epoch mean.
NPZ record lengths vary; mean-size-based MiB/s is only nominal.

Epoch-1 wall-time outlier: pool 6, repetition 2 took 107.25 s (23.70 s above that pool's median). Its cause is not established.

Per-rank hostnames indicate one client; DLIO summary reported eight hosts in 9/9 runs. Use per-rank hostnames for topology.

## Pool comparison at a glance

| Metric | Highest-performing observed mean | Default | HDD | SSD |
|---|---|---:|---:|---:|
| Training input (MiB/s) | Default (colva4) | 271.06 ± 0.03 | 270.99 ± 0.09 | 265.11 ± 8.47 |
| Checkpoint rank 0 (MiB/s) | HDD (colva1–3) | 37.16 ± 0.92 | 38.48 ± 1.31 | 38.04 ± 2.23 |
| Whole-run time (min; lower is better) | Default (colva4) | 7.38 ± 0.00 | 7.38 ± 0.00 | 7.54 ± 0.23 |

Winners here mean only the numerically best mean across **three** runs;
they are not significance claims or isolated HDD/SSD effects.
Input and checkpoint are different I/O phases and must not be added as throughput.
The run timer includes startup, reads, and checkpoint saves.
U-Net training input MiB/s is a **nominal proxy**, not DLIO's invalid
five-epoch mean: validated batches × configured mean NPZ record size
divided by the sum of training-block times where available and epoch
wall times for checkpoint-free epochs 1, 3, and 5. The latter can
include non-read overhead; individual NPZ file sizes vary.
Checkpoint MiB/s is rank-0 file bytes / rank-0 DLIO save time across
two dependent saves within each run; it has no fsync or eight-rank barrier.
