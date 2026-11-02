# DLIO throttled-SSD comparison

The [experiment spec](../../../../docs/specs/DL_Throttled_SSD.md) describes
the five four-target conditions. Run from the repository root on `anjuna2`.
The normal scripts and datasets remain under `normal/` and their existing
cluster paths; this suite stages copies in `dlio/throttled_ssd/` only.

The four targets must already be in the requested pool, and `systemd` must
report **45 MiB/s** read/write limits for its fast IDs and **15 MiB/s** for
the other IDs. `pool_common.py` checks both before a run; it does not move
targets or change limits. Complete existing pool-6 jobs first.

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

## Plot copied full runs locally

Copy each completed `runs-pool*-.../` directory from `anjuna2` into
`results/workloads/runs/`. Render the available pools with one command;
add further run directories as each condition finishes:

```bash
python3 scripts/workloads/dlio/throttled_ssd/plot_results.py --unet \
  results/workloads/runs/runs-pool14-20261102-124553-139890
```

When pool 15 finishes, rerun the same command with its copied directory
appended; continue through pool 18. Use `--resnet` with completed ResNet sweeps.
The plotter validates three
full repetitions per supplied pool, writes five PNGs and source CSVs to
`results/workloads/plots/dlio/throttled_ssd/{unet,resnet}/`, and never runs
DLIO. U-Net's throughput is an estimated sample-size/time proxy where native
rates are missing; checkpoint write throughput times rank 0 without `fsync`.
