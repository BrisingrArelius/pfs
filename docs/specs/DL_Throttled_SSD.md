# DLIO throttled-SSD pooling experiment

This is a new, separate speed-mix experiment; the completed normal
Default/HDD/SSD runs remain unchanged. Compare DLIO ResNet-50 and U-Net 3D
on `anjuna2`, eight ranks, five epochs, synthetic all-rank checkpoints after
epochs 2 and 4, three repetitions per condition. Do not include CosmoFlow
until its normal comparison has completed.

## Controlled conditions

Keep **the same four physical SSD targets (104, 205, 305, 306)** in each
condition; each file still has **four stripes of 512-KiB chunks**, exactly as
in the normal suite. Cap both read and write throughput per target to
45 MiB/s (fast) or 15 MiB/s (slow), a 3:1 ratio. The nominal aggregate
four-fast ceiling is 180 MiB/s, below the ~280 MiB/s measured client TCP
path. These limits must be verified on backend devices; merely configuring
`systemd` limits is not proof they affected application reads.

| Pool | Fast | Slow | Fast target IDs |
|---:|---:|---:|---|
| 14 | 4 | 0 | 104,205,305,306 |
| 15 | 3 | 1 | 104,205,305 |
| 16 | 2 | 2 | 104,205 |
| 17 | 1 | 3 | 104 |
| 18 | 0 | 4 | none |

Each target can belong to only one BeeGFS pool. Pools 14–18 already exist;
move all four targets between them **sequentially**, verifying membership
after each removal and addition. Finish the original pool-6 jobs first:
pool 6 temporarily loses these targets. Restore all four to pool 6 after
the experiment. Do not reuse existing files across conditions: moving
targets does not rewrite file pool metadata. Copy the same prepared normal
input bytes under each new pool's four-stripe directory before timing.

## Execution and evidence

On `colva1`, `colva2`, and `colva3`, verify the target-to-device mapping and
`systemctl show beegfs-storage.service` read/write bandwidth limits for the
current condition. The limits are runtime properties of the storage service
on each backing device, not a setting on a BeeGFS pool. They also affect
older files on those devices until cleared. Record original properties and
restore them when finished. Never change membership or caps during a run.

For each condition: stage identical inputs outside timing; pilot both DLIO
workloads; measure a sustained read/write probe with backend device counters
and client-network counters; then run the three full repetitions. The new
runner verifies configured pool members and limits, preserves native DLIO
evidence, and reports the same rank-0 checkpoint metric as the normal suite.
**It does not itself measure backend bytes or network traffic.** Add those
independent counters before claiming that the speed cap, rather than client
or server caching or the network, caused any application difference.

All four targets are striped into every file. Mixed conditions may therefore
be limited by a single slow member rather than improve monotonically with
the number of fast targets. DLIO's nominal training throughput is not
measured device traffic; U-Net has missing native odd-epoch rates; rank-0
checkpoint completion does not include `fsync` or an eight-rank barrier.
Three repetitions describe variance, not statistical significance.

The normal scripts are in `scripts/workloads/dlio/normal/`. This variant is
in `scripts/workloads/dlio/throttled_ssd/` with one runner per workload and
one shared helper. The old runners and historical results are not modified.
