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

Only then move the targets to pool 15 and change per-device limits for its
3:1 assignment; repeat with `--pool 15`, then pools 16–18. Moving targets
does not restripe previous files: this suite copies a fresh dataset under
each pool's pattern. Restore the original I/O limits and pool-6 membership
at the end. Outputs are written to `~/tejas/dlio/throttled_ssd/`.
