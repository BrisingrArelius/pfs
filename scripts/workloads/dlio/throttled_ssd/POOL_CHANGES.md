# DLIO throttled-SSD pool changes

Run these commands on `anjuna2` **between** completed DLIO runs. All five
conditions use the same targets `104,205,305,306`, with four targets and
512-KiB chunks per file. Only one condition pool can contain them at a time.
`--rmtargets` is the removal flag on this cluster. Inspect the pool listing
after each move; if either operation fails, stop rather than running DLIO
against a partially configured pool. The cap changes are runtime-only
`beegfs-storage.service` settings on the storage nodes.

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
ssh colva1 'sudo systemctl set-property --runtime beegfs-storage.service "IOReadBandwidthMax=/dev/nvme1n1p1 15728640" "IOWriteBandwidthMax=/dev/nvme1n1p1 15728640"' &&
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
