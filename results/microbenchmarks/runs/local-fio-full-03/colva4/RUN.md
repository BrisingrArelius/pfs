# Local FIO run on colva4

> Generated from `manifest.json`; the JSON manifest and native FIO files are authoritative.

## Identity

- Run ID: `dfbd5a8cad2a4dac9c368e87f09c47dc`
- Mode: `full`
- Protocol version: `5`
- FIO version: `fio-3.36`

## Measurement protocol

- Parallel jobs per OST: **4**
- Per-job region: **10 GiB**
- Prepared file per OST: **40 GiB**
- I/O engine: `libaio` with `direct=1`
- Queue depth: **32 per job**, up to **128 aggregate**
- Timing: **5 s ramp + 60 measured s**, `time_based=1`
- Repetitions: **5** per workload
- Each job uses a separate non-overlapping region of the same prepared file.
- Reported bandwidth and IOPS are sums across jobs for one OST.

## Workloads

| Name | Pattern | Block size |
|---|---|---:|
| `seq_read` | `read` | `1m` |
| `seq_write` | `write` | `1m` |
| `rand_read_4k` | `randread` | `4k` |
| `rand_write_4k` | `randwrite` | `4k` |
| `rand_read_128k` | `randread` | `128k` |

## Targets

| OST ID | Media | Mount | Device |
|---:|---|---|---|
| 401 | hdd | `/mnt/hdd1` | `/dev/sda1` |
| 402 | hdd | `/mnt/hdd3` | `/dev/sdc1` |
| 403 | hdd | `/mnt/hdd4` | `/dev/sdd1` |
| 404 | nvme | `/mnt/nvme0` | `/dev/nvme0n1p1` |
| 405 | nvme | `/mnt/nvme1` | `/dev/nvme1n1` |
| 406 | nvme | `/mnt/nvme2` | `/dev/nvme2n1` |
| 407 | nvme | `/mnt/nvme3` | `/dev/nvme3n1` |

## Evidence

Raw `job.fio`, `fio.json`, stdout and stderr are under `raw/`. 
Session progress, completion state, file identity and device snapshots are in `manifest.json`.
