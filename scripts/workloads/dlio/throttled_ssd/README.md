# DLIO throttled-SSD comparison

The [experiment spec](../../../../docs/specs/DL_Throttled_SSD.md) describes
the original four-target conditions. Run from the repository root on `anjuna2`.
The normal scripts and datasets remain under `normal/` and their existing
cluster paths; this suite stages copies in `dlio/throttled_ssd/` only.

## Current experiment: 1.2:1

[`experiment_config.py`](experiment_config.py) is the single place to set
`FAST_MIB_S = 45`, `FAST_TO_SLOW_RATIO = 1.2`, and
`RESULTS_SUBDIR = "throttled_ssd_1.2"`. Rates are per physical SSD target,
in MiB/s; the slow rate is calculated as 45 / 1.2 = **37.5 MiB/s** and the
kernel byte/s limits are rounded to whole bytes/s. Both workload runners use pools **14, 16, 18** with **three full
five-epoch invocations per pool**; Darshan is enabled with `--darshan`.

From restored original pool 6 with no caps, run these **sequentially**:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_cosmoflow_pools.py \
  --all-pools --darshan "$HOME/darshan-nompi/lib/libdarshan.so.0.0.0"

python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py \
  --all-pools --darshan "$HOME/darshan-nompi/lib/libdarshan.so.0.0.0"
```

Each workload runs 14→16→18 (4/0, 2/2, 0/4), then restores pool 6 on
success. The runner checks systemd and kernel `io.max` before each case;
Darshan parser checks occur after each invocation. Existing staged datasets
remain under `dlio/throttled_ssd/`; new results go to
`~/tejas/dlio/throttled_ssd_1.2/{cosmoflow,unet3d}/`. A failure stops the
existing sweep at its current pool; inspect and restore it before another run.

Change the three config constants for a future experiment; choose a **new**
`RESULTS_SUBDIR` when rates change. Restart Python processes after editing the
config. `condition.json` records the actual configured rates per result;
resuming into an existing condition with different rates is rejected.

### Both workloads across reservations

The combined runner has one **shared deadline** from invocation, including
staging, cache preparation, DLIO, Darshan validation and restoration:

```bash
scripts/workloads/dlio/throttled_ssd/run.sh --deadline 6hours
```

It runs CosmoFlow first, then U-Net with whatever budget remains, for three
full repetitions in each of pools 14/16/18. Darshan is always enabled; the
default existing runtime is `~/darshan-nompi/lib/libdarshan.so.0.0.0` (or
`DARSHAN_RUNTIME_LIB`); `--darshan-library /path/libdarshan.so` overrides it.
It reserves up to two minutes (10% for short budgets) for stopping owned
processes and restoring pool 6. A deadline/interruption exits with status 75.

Resume in the next reservation using the same command (automatic continuation)
or explicitly require an unfinished study:

```bash
scripts/workloads/dlio/throttled_ssd/run.sh --continue --deadline 6hours
```

Completed repetitions are journaled and skipped; an interrupted repetition
restarts from the beginning, with partial raw output/checkpoints preserved.
The study journal is `~/tejas/dlio/throttled_ssd_1.2/study-*.json` under the
current config. Resume checks the workload, rates, source and runtime hashes;
do not change these mid-study. SIGTERM/SIGHUP/Ctrl-C follow the cleanup path;
SIGKILL cannot, so continuation checks for study-tagged processes and restores
pool 6 first. A failed cleanup is reported for inspection. The legacy
individual U-Net/CosmoFlow commands below remain non-resumable; use `run.sh`
for the shared-deadline workflow.

For a single `--pool` run, the four targets must already be in that pool,
and `systemd` must report the configured read/write limits (currently
**45 MiB/s** fast and **37.5 MiB/s** slow). `pool_common.py` checks pool membership,
systemd settings, and the kernel's actual `io.max` before a run. On `colva1`,
the limit for target 104 must be set on its whole `/dev/nvme1n1` device:
the partition setting appeared in systemd but installed no kernel limit.
The same issue was observed on colva2 for target 205: cap its whole
`/dev/nvme1n1`, not `/dev/nvme1n1p1`. The configuration script clears old
per-device entries before applying the corrected paths. Kernel verification
remains mandatory; matching systemd properties alone is insufficient.
All four caps now use physical devices: colva1/104 and colva2/205 use
`/dev/nvme1n1`; colva3/305 uses `/dev/nvme1n1` and colva3/306 uses
`/dev/nvme2n1`. The partition-backed BeeGFS storage paths are unchanged.
Complete existing pool-6 jobs first.

To return the four targets from pool 14 to the original pool 6 and clear
runtime bandwidth limits, first finish/stop the workload, then run:

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 14 --restore
```

Restoration checks source target membership, not working throttle caps, so it
also handles a transition that moved targets but failed `io.max` verification.
It verifies original pool-6 membership and cleared service caps afterward.

## Optional built-in Darshan profiling

The U-Net and CosmoFlow runners accept `--darshan [LIBRARY]` for the same
[built-in profiler and compatibility fixes](../normal/README.md#optional-built-in-darshan-profiling-u-net-and-cosmoflow)
as the normal runners. The default runtime is the existing
`~/darshan-nompi/lib/libdarshan.so.0.0.0` (or `DARSHAN_RUNTIME_LIB`).
For an already configured pool, first run its existing short pilot:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_cosmoflow_pools.py --pool 14 --pilot --darshan
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 14 --pilot --darshan
```

Use the currently configured pool ID; these single-pool commands do not
configure it. `--all-pools --darshan` also enables capture, retaining the
original three repetitions per condition by default. Use `--repetitions 1`
for one full five-epoch invocation per selected pool. For the plotting study,
select **14, 16 and 18** explicitly for both workloads:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_cosmoflow_pools.py \
  --all-pools --pools 14 16 18 --repetitions 1 --darshan
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py \
  --all-pools --pools 14 16 18 --repetitions 1 --darshan
```

Run the commands sequentially, starting each sweep from restored original
pool 6. Each sweep restores pool 6 after successful completion. `--pools`
requires `--all-pools`; it selects the pool order and does not alter any
individual invocation's workload. Without `--pools`, both workloads use
14/16/18. A single already-configured
condition also accepts `--pool 14 --repetitions 1 --darshan`. Pilots remain one
shortened invocation regardless of the full-run repetition setting.

Logs/configuration/rank identities are saved under each invocation's
`darshan/` directory. Both parsers must pass before checkpoint cleanup and
completion are recorded. Record/file filtering does not turn STDIO aggregates
into exact POSIX events. The ResNet CLI is unchanged.

For A/B/C, pass the completed `rep1-poolN` directory to
`normal/plot_darshan.py`; it does not require three repetitions. The separate
performance comparison plotter retains its repeated-run requirements.

## ResNet across cluster reservations

Start three conditions (4/0, 2/2, 0/4; three repetitions each) from original pool 6:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --all-pools --deadline 6h
```

Continue the newest unfinished managed sweep during the next reservation:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --continue --deadline 6h
```

`--deadline` is a duration from invocation: seconds, `90m`, `6h` or `1h30m`.
It includes staging, checks, cache preparation and DLIO. Up to the final
60 seconds (10% for short budgets) are reserved for stopping the active
process group and restoring pool 6 with no caps. An interrupted repetition
is **restarted**, not resumed mid-epoch; completed repetitions are skipped.
Each retry uses the full cache protocol.

Progress is saved after every completed repetition in
`~/tejas/dlio/<RESULTS_SUBDIR>/resnet50/sweep-*.json`. Continuation reuses the
same `runs-poolN-*` directories. Per-repetition CSV journals rebuild combined
metrics without duplicate rows. Interrupted raw outputs are retained under
`interrupted/`; partial checkpoints are renamed in place on BeeGFS, with
their paths recorded there. Staging uses a temporary file and reuses
completed same-size copies after an interruption.

Ctrl-C, SIGTERM and SIGHUP follow the same stop/save/restore path. If
restoration fails or runs out of time, inspect membership/caps before
releasing the reservation. SIGKILL cannot clean up; ensure old MPI processes
have stopped before continuing after one. Continuation detects whether the
targets are in pool 6 or an experiment pool and reinstalls verified caps;
if necessary it first restores pool 6, then starts/resumes the 14→16→18 sweep.
Split/missing targets stop execution for inspection. A lock prevents two
managed ResNet sweeps running concurrently.

If no unfinished journal exists, `--continue` can adopt the latest legacy
`python3 -c ... run_all("resnet")` sweep. Stop that old process first. It
checks the condition, native completion evidence, eight metrics and four
device rows before preserving each completed repetition. Fully completed
pool 14 is skipped, so the next condition is pool 16; any partial repetition
is restarted. Existing pool-15/17 results are retained but not scheduled.
The old process keeps its original schedule until stopped; updating code
cannot change the process already running.

`--pool N` and `--pilot` keep single-condition behavior; reservation flags
apply to the three-condition sweep (9 full runs). Without a deadline the
sweep runs until complete or interrupted.
A new sweep is refused while an unfinished journal exists; use `--continue`.

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
pools 16 and 18, setting their read/write caps and doing three runs in
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
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 14 --to-pool 16
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 16 --darshan
```

Continue to pool 18, one pool at a time. `configure_pool.py` moves
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
pattern. Results go to `~/tejas/dlio/<RESULTS_SUBDIR>/`.

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
  results/workloads/runs/runs-pool14-TIMESTAMP \
  results/workloads/runs/runs-pool16-TIMESTAMP \
  results/workloads/runs/runs-pool18-TIMESTAMP
```

The plotter also accepts a subset of completed pools. Use `--resnet` or `--cosmoflow` with
completed sweeps of those workloads.
The plotter validates three
full repetitions per supplied pool, writes six PNGs and source CSVs to
`results/workloads/plots/dlio/throttled_ssd_1.2/{unet,resnet,cosmoflow}/`
under the current config, and never runs
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

The plotter rejects mixed target-cap pairs. When plotting historical 45/15 or
45/30 results while the current configuration is 45/37.5, pass `--output` explicitly
to the historical plot location; this prevents silently mixing experiments.

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
