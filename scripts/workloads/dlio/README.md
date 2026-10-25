# DLIO BeeGFS pool runs

Run these commands on `anjuna2` from the repository root with DLIO 2.0.0.
They compare the existing BeeGFS pools: 1 (Default on `colva4`), 3 (HDD),
and 6 (SSD). Each file uses four targets and 512-KiB chunks. The workload
data lives under `/mnt/beegfs/pfs/tejas/dlio/<workload>/`; logs and CSVs go
under `~/tejas/dlio/<workload>/`.

## One workload at a time

| Workload | Data preparation | Short pilot | Full comparison |
|---|---|---|---|
| ResNet-50 | `python3 scripts/workloads/dlio/generate_data.py --resnet` | `python3 scripts/workloads/dlio/run_resnet50_pools.py --model-size-bytes 100000000 --pilot` | `python3 scripts/workloads/dlio/run_resnet50_pools.py --model-size-bytes 100000000` |
| U-Net 3D | `python3 scripts/workloads/dlio/generate_data.py --unet` | `python3 scripts/workloads/dlio/run_unet3d_pools.py --pilot` | `python3 scripts/workloads/dlio/run_unet3d_pools.py` |
| CosmoFlow | `python3 scripts/workloads/dlio/generate_data.py --cosmoflow` | `python3 scripts/workloads/dlio/run_cosmoflow_pools.py --pilot` | `python3 scripts/workloads/dlio/run_cosmoflow_pools.py` |

First prepare the data, then inspect the pilot's checkpoint files and native
`per_epoch_stats.json` before launching its full comparison. The pilot uses
pool 1, two steps in each of two epochs, and retains its checkpoint files.
The full runner uses eight MPI ranks and five epochs; it executes three
repetitions of pools 1, 3, and 6 sequentially, checkpointing after epochs
2 and 4. `sync` and a client cache drop precede each measured run, and its
checkpoint files are removed only after the raw results and OST placements
have been recorded. OST and cache commands require `sudo` access.

`generate_data.py` uses DLIO to generate pool 1's data, then copies the same
bytes to pools 3 and 6 under their BeeGFS patterns. It reuses existing
complete datasets (including the previously generated ResNet copies) and
stops if a dataset is partial. It never regenerates data during a timed run.
CosmoFlow has 524,288 files of roughly 2.83 MB **per pool**; confirm time,
free space, and inode availability before preparing it.

The full-run `metrics.csv` contains DLIO training rates and rank-0 checkpoint
save rates computed from rank-0 output-file bytes and DLIO's rank-0 duration.
Its `raw_output` column identifies the preserved DLIO output directory.
Checkpoint saves in DLIO 2.0.0 do **not** include `fsync`: these are
application-observed save-completion rates, not durable-write or eight-rank
aggregate rates. The CosmoFlow OST listing samples 65 input files per pool;
ResNet and U-Net list every input file. None of these commands runs the full
factor matrix or checkpoint-recovery suite in `docs/specs/DL_Experiment.md`.

`pool_common.py` contains the mechanics shared by the three small runners.
Each checkpoint class remains inside its workload runner so DLIO can import
it in every MPI rank without changing the installed DLIO package.
