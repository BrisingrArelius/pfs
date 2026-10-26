# ResNet-50 DLIO pool comparison

Nine completed runs: three repetitions per pool, eight MPI ranks per run.

| Pool | DLIO mean training MiB/s | Epoch 1 nominal MiB/s | Epochs 2–5 nominal MiB/s | Checkpoint 2 rank-0 MiB/s | Checkpoint 4 rank-0 MiB/s |
|---|---:|---:|---:|---:|---:|
| 1 | 275.57 ± 0.79 | 275.93 ± 0.97 | 275.48 ± 0.76 | 36.29 ± 2.61 | 36.44 ± 1.54 |
| 3 | 259.79 ± 3.87 | 249.77 ± 1.48 | 262.30 ± 4.48 | 34.60 ± 0.44 | 37.47 ± 1.03 |
| 6 | 276.57 ± 3.33 | 278.79 ± 0.06 | 276.02 ± 4.16 | 35.32 ± 1.57 | 36.05 ± 2.49 |

Values are mean ± sample standard deviation across three independent runs.
Observed training mean versus pool 1: pool 3 -5.7%, pool 6 +0.4%.
Pool 3 versus pool 1: epoch 1 -9.5%, epochs 2–5 -4.8%. These are descriptive comparisons, not media-isolated effects.
Training MiB/s is DLIO's modeled sample rate × configured bytes/record; it is not a measured
POSIX read-byte counter or device throughput. Per-rank load/proc times are application-step
timings, not separate read/write throughput; do not sum them across concurrent ranks.
Checkpoint MiB/s is actual rank-0 file bytes / DLIO's rank-0 save duration; no fsync,
and no synchronized eight-rank completion time. Checkpoint data files were removed after
the runner saved their sizes and placement; other ranks' measured byte totals are unavailable.
Client caches were dropped before each run, but server cache residency was not verified.
Default resides on colva4; HDD/SSD pools span colva1–3, so media and server topology differ.
The run order was pool 1 → 3 → 6 in each repetition; time/order effects are not eliminated.

Actual rank output hostnames: anjuna2 only. DLIO summary num_hosts reports 8 in 9/9 runs; use observed per-rank hostnames instead.

## Timing discrepancy to investigate

Expected samples per epoch are calculated from the fixed profile's file, record,
batch and rank counts. The wall-implied rate is expected samples / DLIO's training
block duration; it is a consistency check, not a measured byte rate. Flagged cases:

- Pool 6, repetition 1, epoch 2: DLIO samples/s versus block wall time -10.70%; block-end to checkpoint-start 60.11 s.

Do not attribute this gap to checkpoint write time: it occurs **before** DLIO's
checkpoint timer starts. Keep the native metric; investigate its attribution.
