# DLIO throttled-SSD comparison

The [experiment spec](../../../../docs/specs/DL_Throttled_SSD.md) describes
the five four-target conditions. Run from the repository root on `anjuna2`.
The normal scripts and datasets remain under `normal/` and their existing
cluster paths; this suite stages copies in `dlio/throttled_ssd/` only.

For a single `--pool` run, the four targets must already be in that pool,
and `systemd` must report **45 MiB/s** read/write limits for its fast IDs
and **15 MiB/s** for the other IDs. `pool_common.py` checks pool membership,
systemd settings, and the kernel's actual `io.max` before a run. On `colva1`,
the limit for target 104 must be set on its whole `/dev/nvme1n1` device:
the partition setting appeared in systemd but installed no kernel limit.
Complete existing pool-6 jobs first.

## Entire U-Net or CosmoFlow sweep from the normal pool

With the original nine SSD targets in pool 6, no storage-service bandwidth
limits, and empty pools 14–18, run on `anjuna2`:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --all-pools
```

For reduced CosmoFlow, first prepare its normal pool-1/3/6 datasets with
`python3 scripts/workloads/dlio/normal/generate_data.py --cosmoflow`
and complete its normal pilot/full comparison. When pool 6 again has its
original nine targets and no caps, run its three-condition protocol:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_cosmoflow_pools.py --all-pools
```

CosmoFlow runs **only pools 14 → 16 → 18 (4/0, 2/2, 0/4)**, three repetitions
each, then restores pool 6. It uses **8,192 TFRecords per pool**; the first invocation in each
condition copies the inputs before timing. Like the normal runner, it
inspects a 65-file sample of OST assignments rather than querying all
8,192 files. For a short pilot in a currently configured pool, use
`run_cosmoflow_pools.py --pool 14 --pilot`.

This moves the four selected SSD targets from pool 6 into pool 14, applies
the 4:0 caps, and runs three full U-Net repetitions. It then moves through
pools 15–18 in order, setting their read/write caps and doing three runs in
each; after a successful sweep it returns the targets to pool 6 and clears
the runtime limits. Dataset copies are outside measured runs. Before each
DLIO invocation, the runner syncs and drops Linux page caches on
`colva1–4` over SSH, then drops the client cache on `anjuna2` and waits
30 seconds; the preparation is outside timing and requires noninteractive
remote `sudo`. Full-run checkpoints are removed after their evidence is
recorded, while pilots retain checkpoint files for inspection. This does not clear
DLIO's own buffers or caches within an epoch. If a move,
copy, or run fails, execution stops at that condition **without moving the
targets again**; inspect the pool and any still-running processes before
resuming. `--all-pools` cannot be combined with `--pilot`.
Input staging announces its start and completion. Each run prints a flushed
`pool N, run X/3 complete` line only after its metrics and checkpoint checks
succeed; pilots use `X/1`. The final pool-6 restoration happens after the
last completed condition.

For the currently configured **pool 14** (all four fast), pilot one workload
at a time:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --pool 14 --pilot
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 14 --pilot
```

The first invocation copies the normal input dataset into the experiment
pool before DLIO starts and validates its four-target, 512-KiB pattern. It
can take time; the copy is outside measured runtime. The pilot has two steps
per epoch, two epochs and a 16-MiB checkpoint per rank. Inspect its native
output and retained checkpoint files. These pilots do not establish that
device throttling constrained actual I/O: measure backend and client-network
counters during a sustained probe before a full comparison.

After that validation, omit `--pilot` for **three full runs** per workload:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --pool 14
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 14
```

After each pool's runs finish, use the copyable commands in
[POOL_CHANGES.md](POOL_CHANGES.md) or its one-step equivalent, for example:

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 14 --to-pool 15
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 15
```

Continue through pools 16–18, one pool at a time. `configure_pool.py` moves
targets and sets runtime limits; it never launches DLIO. It verifies the
current pool and caps before moving, and the next condition after applying
limits. A failed move may have partially completed: inspect the pool listing
before another attempt. To return targets to pool 6 after the last run:

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 18 --restore
```

The restore command clears the experiment's runtime limits; use it only if
there were no pre-existing service I/O limits. Moving targets does not
restripe previous files: this suite stages fresh copies under each pool's
pattern. Results go to `~/tejas/dlio/throttled_ssd/`.

Each **new** throttled sweep writes `device_io.csv` beside `metrics.csv`:
physical SSD read/write bytes and whole-invocation MiB/s for each of the
four targets in each repetition. Counter snapshots bracket DLIO itself,
after the cache preparation and before checkpoint-file cleanup. They cover
training **and** checkpoint phases together, and device-wide counters can
include unrelated activity or delayed writeback. They are not per-epoch or
per-rank measurements. Existing sweeps have no retrospective device counts.

To check actual physical SSD read/write MiB/s without flooding the terminal,
run this on `anjuna2` in a second terminal **while DLIO is active**:

```bash
python3 scripts/workloads/dlio/throttled_ssd/check_throughput.py --seconds 10
```

It prints one line per selected target with measured device rates, configured
read/write caps, and whether the limits match the storage service's kernel
`io.max`. The device counters include other I/O on the same drive. A matched
cap with zero device activity means the cap was not exercised; it cannot
explain DLIO's nominal training rate.

For an **active native BeeGFS I/O check** of one currently configured pool,
run between DLIO runs (example: all-slow pool 18):

```bash
python3 scripts/workloads/dlio/throttled_ssd/check_throughput.py --native --pool 18
```

This writes and reads a temporary **2-GiB**, four-target/512-KiB-striped
file using `fio --direct=1`, printing fio throughput and actual backend MiB/s
per target. It removes the probe file after success; on failure it prints
the path for inspection. `fio` must be installed. This is I/O **through
BeeGFS**, so it exercises the throttled storage service; running fio directly
on `/dev/nvme*` would bypass those service limits. A read served from server
cache can still bypass the SSD cap, so compare fio bytes with device-read
bytes before claiming the cap was exercised. Do not run the probe alongside
a timed DLIO repetition.

## Plot copied full runs locally

Copy each completed `runs-pool*-.../` directory from `anjuna2` into
`results/workloads/runs/`. Render the available pools with one command;
add further run directories as each condition finishes:

```bash
python3 scripts/workloads/dlio/throttled_ssd/plot_results.py --unet \
  results/workloads/runs/runs-pool14-20261102-124553-139890
```

When pool 15 finishes, rerun the same command with its copied directory
appended; continue through pool 18. Use `--resnet` or `--cosmoflow` with
completed sweeps of those workloads.
The plotter validates three
full repetitions per supplied pool, writes six PNGs and source CSVs to
`results/workloads/plots/dlio/throttled_ssd/{unet,resnet,cosmoflow}/`, and never runs
DLIO. Pool performance and per-run plots use **end-to-end samples/s**:
total processed samples across all epochs divided by DLIO's full-run
`end - start` seconds. Counts come from full batches and are checked against
each rank's recorded load-step counts. This includes checkpointing and all
overhead within those timestamps; input staging and pre-run cache preparation
are outside them. It is a derived completion-time rate, not an arithmetic
average of DLIO's epoch rates or a physical-device byte rate. Pool error bars
summarize three independent per-run rates. The redundant whole-run-time panel
and inferred pool-summary figure are removed; timestamps and durations remain
in the source data/CSVs.

Per-epoch diagnostic plots retain DLIO-reported samples/s; zero-reported
epochs are gaps, not interpolated. CSVs retain native and inferred fields.
`training_throughput_inferred.png` separately shows the calculated MiB/s
series with dotted lines; it is not a DLIO-reported byte rate. Checkpoint
write throughput times rank 0 without `fsync`.

For U-Net, the per-epoch inferred byte-rate numerator is
5 steps × 4 samples/rank × 8 ranks × 146,600,628
configured bytes/sample = 23,456,100,480 estimated bytes per epoch. This is
160 processed samples, not all 168 input files; the remainder does not form
a full global batch. Record sizes vary, so this is nominal sample volume,
not measured file-read bytes. Its per-epoch denominator is training-block time
where present and otherwise epoch wall time. `epochs.csv` records the
sample count, configured bytes, estimated volume and timing source explicitly.
