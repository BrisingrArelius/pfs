# DLIO throttled-SSD pool changes

> **Current experiment:** use the [10:1 configuration and commands](README.md#current-experiment-101).
> The detailed five-pool 45/15 examples below are historical and must not be
> used as current run instructions. Both current workload runners use
> pools 14, 16 and 18; current caps come from `experiment_config.py`.

To run the **entire U-Net sweep in one command**, start on `anjuna2` with
the original nine targets in pool 6, no storage-service bandwidth limits,
and empty pools 14–18:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --all-pools
```

After CosmoFlow's normal datasets and three-pool sweep are complete, use
the same original pool-6 state for its three-condition sweep (4/0, 2/2, 0/4):

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_cosmoflow_pools.py --all-pools
```

Each command runs three full repetitions per condition and restores pool 6
after all selected conditions succeed. Both move directly 6→14→16→18→6.
If any step fails, it stops at the current
pool without restoring or launching further runs; inspect processes and pool
membership before retrying.

ResNet can span reservations using three conditions (4/0, 2/2, 0/4;
pools 14, 16, 18), three repetitions each:

```bash
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --all-pools --deadline 6h
# During the next reservation:
python3 scripts/workloads/dlio/throttled_ssd/run_resnet50_pools.py --continue --deadline 6h
```

The managed ResNet runner reserves up to the final minute for stopping and
restoring pool 6. Continuation skips completed repetitions and restarts the
partial one, automatically moving from pool 6 to the required condition.
If restoration is incomplete, it reports that explicitly; inspect membership
and caps before releasing the reservation. See [README.md](README.md#resnet-across-cluster-reservations)
for saved-state locations and interruption behavior.

For manual condition-by-condition execution, run the commands below on
`anjuna2` **between** completed DLIO runs. All five
conditions use the same targets `104,205,305,306`, with four targets and
512-KiB chunks per file. Only one condition pool can contain them at a time.
`--rmtargets` is the removal flag on this cluster. Inspect the pool listing
after each move; if either operation fails, stop rather than running DLIO
against a partially configured pool. The cap changes are runtime-only
`beegfs-storage.service` settings on the storage nodes.

Target 104 on `colva1` needs its cap on the **whole** `/dev/nvme1n1`:
setting `/dev/nvme1n1p1` appeared in systemd but installed no kernel
`io.max` rule. If both paths remain in systemd after a previous attempt,
clear the duplicate before continuing a pool-14 run:

```bash
ssh colva1 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=" "IOWriteBandwidthMax=" && sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1 47185920" "IOWriteBandwidthMax=/dev/nvme1n1 47185920"'
```

The transition script clears old per-device entries before installing each
condition and verifies the resulting kernel limits on all four devices.

For each pool, copy the **script commands** first. The manual BeeGFS/systemd
commands immediately below them are an alternative; **do not run both**.
Each workload command performs three full repetitions; the first invocation
for a new pool also stages a copy of its inputs outside measurement.

## 14 → 15: 104, 205, 305 fast; 306 slow

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 14 --to-pool 15 &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 15
```

Manual alternative:

```bash
sudo beegfs-ctl --modifystoragepool --id=14 --rmtargets=104,205,305,306 &&
sudo beegfs-ctl --modifystoragepool --id=15 --addtargets=104,205,305,306 &&
sudo beegfs-ctl --liststoragepools &&
ssh colva3 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1p1 47185920" "IOReadBandwidthMax=/dev/nvme2n1p1 15728640" "IOWriteBandwidthMax=/dev/nvme1n1p1 47185920" "IOWriteBandwidthMax=/dev/nvme2n1p1 15728640"' &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 15
```

## 15 → 16: 104, 205 fast; 305, 306 slow

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 15 --to-pool 16 &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 16
```

Manual alternative:

```bash
sudo beegfs-ctl --modifystoragepool --id=15 --rmtargets=104,205,305,306 &&
sudo beegfs-ctl --modifystoragepool --id=16 --addtargets=104,205,305,306 &&
sudo beegfs-ctl --liststoragepools &&
ssh colva3 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1p1 15728640" "IOReadBandwidthMax=/dev/nvme2n1p1 15728640" "IOWriteBandwidthMax=/dev/nvme1n1p1 15728640" "IOWriteBandwidthMax=/dev/nvme2n1p1 15728640"' &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 16
```

## 16 → 17: 104 fast; 205, 305, 306 slow

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 16 --to-pool 17 &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 17
```

Manual alternative:

```bash
sudo beegfs-ctl --modifystoragepool --id=16 --rmtargets=104,205,305,306 &&
sudo beegfs-ctl --modifystoragepool --id=17 --addtargets=104,205,305,306 &&
sudo beegfs-ctl --liststoragepools &&
ssh colva2 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1p1 15728640" "IOWriteBandwidthMax=/dev/nvme1n1p1 15728640"' &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 17
```

## 17 → 18: all four slow

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 17 --to-pool 18 &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 18
```

Manual alternative:

```bash
sudo beegfs-ctl --modifystoragepool --id=17 --rmtargets=104,205,305,306 &&
sudo beegfs-ctl --modifystoragepool --id=18 --addtargets=104,205,305,306 &&
sudo beegfs-ctl --liststoragepools &&
ssh colva1 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=" "IOWriteBandwidthMax=" && sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1 15728640" "IOWriteBandwidthMax=/dev/nvme1n1 15728640"' &&
python3 scripts/workloads/dlio/throttled_ssd/run_unet3d_pools.py --pool 18
```

The workload runner checks all three nodes' read **and** write limits before
starting. `47185920` bytes/s is 45 MiB/s; `15728640` is 15 MiB/s.

## Restore pool 6 after the last run

These runtime caps were absent before this experiment. If different limits
were configured before your session, restore those instead of clearing them.

```bash
python3 scripts/workloads/dlio/throttled_ssd/configure_pool.py --from-pool 18 --restore
```

Manual alternative:

```bash
sudo beegfs-ctl --modifystoragepool --id=18 --rmtargets=104,205,305,306
sudo beegfs-ctl --modifystoragepool --id=6 --addtargets=104,205,305,306
sudo beegfs-ctl --liststoragepools
for host in colva1 colva2 colva3; do
    ssh "$host" 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=" "IOWriteBandwidthMax="'
    ssh "$host" 'systemctl show beegfs-storage.service -p IOReadBandwidthMax -p IOWriteBandwidthMax'
done
```
