# DLIO Experiment Specification

## Objective

Evaluate the effect of BeeGFS storage pooling on representative deep-learning
I/O workloads.

BeeGFS-wide factors such as storage-pool configuration, stripe count, stripe size,
target-selection policy (`randomized`/`roundrobin`), proposed high/low free-space
state on the same devices (pending user confirmation of capacity meaning and
thresholds), five independent repetitions, fixed membership per measured run,
blocked/counterbalanced ordering, and cache handling are inherited from `Global.md`. Storage
configurations are D (all targets in Default, experimentally pooling-off), S
(SSD-only), and H (HDD-only); D still uses a BeeGFS pool.

DLIO emulates workload I/O behavior; it does not execute the neural networks.

This is a separate application-validation suite, not a microbenchmark domain.
Use the microbenchmark results to interpret application behavior without treating
their throughput differences as additive component costs.

## Workloads

| ID | DLIO profile | Framework / loader | Dataset layout | I/O characteristic |
|---|---|---|---|---|
| W1 | `resnet50_v100` | TensorFlow / tf.data | TFRecord; 1,024 files; 1,251 records/file; 114,660 B/record | Records packed into moderately large files |
| W2 | `unet3d_v100` | PyTorch DataLoader | NPZ; 168 files; 1 record/file; ~146.6 MB/file | Small number of very large files |
| W3 | `cosmoflow_v100` | TensorFlow | TFRecord; 524,288 files; 1 record/file; ~2.83 MB/file | Very large file count; metadata/file-open intensive |

The three workloads therefore represent:
- W1 — multi-record container files
- W2 — large-file I/O
- W3 — many independent files

## DL-Specific Factors Varied

| Factor | Values |
|---|---|
| Workload | W1, W2, W3 |
| MPI processes | 1, 2, 4 |
| I/O phase | Training input, checkpoint write, checkpoint recovery |

All BeeGFS-specific factors are inherited from the global experiment matrix.

## Workload Settings

Use the upstream DLIO workload configuration wherever possible.

| Setting | W1 | W2 | W3 |
|---|---:|---:|---:|
| Epochs | 5 | 5 | 5 |
| Emulated compute time | 0 | 0 | 0 |
| Batch size | 64 | 4 | 1 |
| Reader threads | 8 | 4 | 4 |
| Shuffle | Seeded | Seeded | Seeded |
| Prefetch | 2 | 2 | 2 |

Dataset generation is performed before measured benchmark runs.

### Capacity Provisioning and Cache Preparation

Confirm the proposed capacity interpretation and supply the explicit site
high/low free-space bands, inode reserves, and write headroom required by
`Global.md` before execution. Verify allocated dataset
bytes/inodes per target and data volume per server, allowing for peak checkpoint
retention, temporary files, and coexisting dataset copies. Check feasibility for
each D/S/H target set and stripe count, including low free space. Document
infeasible cells; do not shrink datasets only for one capacity or pool condition.

Apply pool/stripe/chooser settings before generating new files and verify actual
placement. Recreate datasets and prepared recovery checkpoints when changing
placement settings; changing settings alone does not relocate existing files.
Preserve dataset contents and working-set size across those recreations.
Dataset generation, recovery checkpoint preparation, and filler
creation/removal occur outside measured phases. Account for dataset allocation
when provisioning filler to reach the starting band; restore it before each
independent repetition. Let provisioning and background activity settle before
measurement, and maintain the declared band using reserved dataset/checkpoint
headroom without timed filler adjustments. Follow the global policy for band
excursions. Within each D, S, or H configuration, high/low comparisons use identical
workloads, datasets/working sets, devices, eligible target IDs and counts, and
other settings. Record per-target free bytes, free percent,
free inodes, and internal Normal/Low/Emergency class before/after each run;
record byte and inode classes separately where exposed, and record installed
capacity and internal classification thresholds separately from experimental bands.

Preserve realistic input caching with the same explicit preparation procedure
for comparable runs. Record client cache mode, RAM per server, data volume per
server, preparation commands/scope, and backend device traffic per epoch/phase.
Client cache dropping or direct I/O alone does not establish cold server caches.
Generation and recovery preparation may warm caches; record their relationship
to measurement. Report epoch 1 separately from epochs 2–5, retaining each epoch's
metrics; call epoch 1 cold only with verified server cache control and backend
traffic evidence. Five epochs do not replace five independent run repetitions.

## Checkpointing

This section describes checkpoint I/O performed by the workload. Durable
experiment-progress checkpoints are specified separately below.

Checkpoint experiments use rank-sharded checkpoint files.

| Setting | Value |
|---|---|
| Parallelism | `model.parallelism.zero_stage: 3` |
| Checkpoint frequency | Every 2 epochs |
| `checkpoint_fsync` | `true` |

Model-state size is taken from the corresponding upstream DLIO profile where
defined. Any required override must be recorded explicitly in the experiment
configuration.

Include checkpoint `fsync` completion in write-phase timing and its throughput
denominator. Record retention/cleanup and recovery cache preparation; perform
cleanup outside timing and reserve headroom for peak retained checkpoints.

## Measurements

For each run collect:

- Epoch / phase completion time
- Effective read or write throughput
- Bytes read and written
- POSIX read/write time
- Metadata time
- `open`, `stat`, `read`, `write`, `seek`, and `fsync` counts where applicable
- Epoch 1 and individual later-epoch metrics, with backend device bytes/traffic
- Per-target capacity/class before and after each run, actual placement, and target traffic
- Client cache mode, server RAM, data volume per server, and preparation/provisioning records

## Execution

### Darshan and phase attribution

Instrument measured DLIO/application processes where supported, validating both
MPI-rank and framework loader/worker coverage. Confirm library/build integration,
enabled modules and log creation in a pilot; preload in a parent alone does not
prove all I/O workers are covered. Record gaps rather than treating missing
records as zero I/O. Keep instrumentation settings fixed across comparisons.

Preserve raw Darshan logs with run/attempt/process association and native DLIO
output. Exclude dataset generation from measured statistics or label it separately.
Standard Darshan summaries do not automatically provide per-epoch measurements:
use DLIO phase/epoch output and aligned telemetry; declare any extra tracing needed
for finer attribution. Darshan alone cannot verify cache residency or backend
traffic. Retain realistic application prefetch/caching and the declared protocol.

### Progress resume and allocation planning

Follow the durable manifest, atomic progress, evidence validation and separate
measurement/analysis status contract for the experiment design. The existing
synthetic runner's simple resume keys do not implement this full-factor contract.
Use stable workload × global configuration × concurrency × repetition × phase
identities and distinct retry attempts.

Treat a dependent five-epoch sequence as a restartable measurement unit: resuming
at epoch 3 after relinquishing nodes does not preserve the intended warm-cache
history. Restart interrupted sequences after restoring preparation. Separately
completed write/recovery measurements may be reused only when their evidence and
prepared-file prerequisites are verified; a progress marker is not a usable
application checkpoint. Parser failure alone must not repeat successful I/O.

Estimate dataset generation, placement recreation, capacity provisioning, all
epochs, checkpoint writes including synchronization, recovery and cleanup from
pilots. Account for five independent repetitions and the full global matrix;
five epochs are not five repetitions. Fit restartable units within allocations
and record session boundaries and restored preparation on every resume.

### Run sequence

For every `(workload, MPI process count)` configuration:

1. Plan feasible capacity-state blocks using site bands/headroom; counterbalance feasible D/S/H and chooser comparisons within blocks and high/low block order across repetitions where feasible. Record constraints and order.
2. Apply BeeGFS settings, recreate dataset/checkpoint files when placement settings change, verify placement, provision filler, let activity settle, and apply the declared cache procedure outside timing.
3. Record starting capacity/class and run training-input measurements, separating epoch 1 from later epochs.
4. Run checkpoint-write measurements including `fsync`, then checkpoint-recovery measurements with declared cache preparation. Restore starting capacity conditions before each separately measured run.
5. Record ending capacity/class and backend traffic for every run; hold settings and pool membership fixed, maintain the confirmed capacity band, and document band excursions/internal class transitions under the global policy.
6. Execute five independent repetitions per measured configuration under the global block-order policy for expensive capacity changes, restoring preparation and starting conditions each time.

DLIO workload parameters remain identical when comparing BeeGFS configurations.
