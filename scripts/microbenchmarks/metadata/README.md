# BeeGFS metadata benchmark

`run_mdtest.py` is the only metadata runner. Run it as `pfs` from `~/pfs` on
**anjuna3**. It uses the existing `/mnt/beegfs` mount; it does not change
BeeGFS configuration, storage pools, caches, or NetBench mode. Only the user
runs cluster commands.

The call chain is `run_mdtest.py` → MPICH `mpirun` → one `mdtest` process per
rank → Linux/BeeGFS client → BeeGFS metadata service and backing storage.
Anjuna3 also hosts the metadata service, so its local-client case is labelled
separately. These are end-to-end zero-byte namespace operations, not pure
metadata-server CPU or data-throughput measurements.

| Mode | Cases | Items per rank | Repetitions |
|---|---:|---:|---:|
| Pilot | 4 | 1,000 | 1 per selected case |
| Full | 90 | Set `FULL_ITEMS_PER_RANK` after pilot timing review | 5 |

The pilot cases are anjuna2/1-rank/flat, anjuna3/1-rank/flat,
both-clients/1-rank/per-rank, and both-clients/4-rank/flat. The full plan is
3 placements × 3 rank counts per participating client (1, 4, 16) × 2 layouts
(flat, per-rank) × 5 repetitions, shuffled and rotated deterministically.
The full count is intentionally unset; the runner refuses `--full` until the
pilot establishes a useful per-phase duration. The pilot is a functional and
timing check, not a replicated performance result.

Each case is one MPI invocation of the installed
`/home/pfs/ior-main/src/mdtest` with `-d <owned-workdir> -n <items> -i 1
-w 0 -e 0 -N 0 -P`, plus `-u` for the per-rank layout. MPICH is fixed at
`/mnt/nfs_shared/mpich-install/bin/mpirun` and launches with explicit host
slots and core binding. The workload includes directory create/stat/remove
and zero-byte file create/stat/read/remove; exact phase labels and rates will
be checked against the pilot's native output later. Do not treat a zero exit
alone as a validated performance result.

After placing the current code in `~/pfs` on anjuna3, start the pilot with a
new ID:

```bash
python3 scripts/microbenchmarks/metadata/run_mdtest.py \
  --pilot --run-id metadata-pilot-01
```

The script first checks the live mount, mdtest/MPICH help and binary hashes,
NetBench off, and that both MPI clients see the BeeGFS marker and the same
mdtest binary. It launches one case at a time and captures unchanged native
`stdout.txt` and `stderr.txt`, the exact `command.json`, preflight and
before/after mount records, and `result.json` under
`~/pfs/results/microbenchmarks/runs/<run-id>/cases/<case-id>/`. `plan.json`
and `remote_clients.json` pin the run protocol. The working directory is an
exact marker-owned descendant of `/mnt/beegfs/pfs/.metadata-mdtest/<run-id>/`.
Successful cases are cleaned from BeeGFS; raw local evidence remains. A rerun
with the same ID skips intact completed cases.

If mdtest fails, times out, or is interrupted, the runner preserves that
case's BeeGFS namespace because remote MPI ranks might still exist. It will
not silently retry that case. Check remote processes and the preserved path
before manual recovery, then use a fresh run ID. Pilot cases have a five-minute
per-case timeout; the full timeout is a module constant to review alongside
`FULL_ITEMS_PER_RANK` after the pilot. There is no allocation-wide deadline
or automatic recovery of a failed case in this narrowly scoped runner. No
metadata cluster run or plot exists yet. The old offline visualizer was
removed because it could not read any runner-produced raw format;
interpretation and plotting follow the pilot.
