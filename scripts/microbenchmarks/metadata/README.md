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
| Full | 90 | 10,000 | 5 |

The pilot cases are anjuna2/1-rank/flat, anjuna3/1-rank/flat,
both-clients/1-rank/per-rank, and both-clients/4-rank/flat. The full plan is
3 placements × 3 rank counts per participating client (1, 4, 16) × 2 layouts
(flat, per-rank) × 5 repetitions, shuffled and rotated deterministically. The
four pilot MPI-case durations (`ended_at - started_at`) sum to 28.84 seconds;
this excludes preflight and whole-run overhead. The fastest reported phase was
anjuna3 directory stat at 0.001 ms/op, about 1 ms for 1,000 items per rank.
The full count is provisionally 10,000 per rank, targeting about 10 ms for a
similarly fast phase. That is a workload-sizing choice, not a validated timer
noise floor or a variance estimate. Each invocation still has `-i 1`; the five
full repetitions are separate cases in the matrix. The full suite has no
reliable runtime estimate yet: the pilot did not include 16 ranks per host, and
the full matrix may take hours at 10,000 items per rank.

Each case is one MPI invocation of the installed
`/home/pfs/ior-main/src/mdtest` with `-d <owned-workdir> -n <items> -i 1
-w 0 -e 0 -N 0 -P`, plus `-u` for the per-rank layout. MPICH is fixed at
`/mnt/nfs_shared/mpich-install/bin/mpirun` and launches with an explicit
shared BeeGFS working directory, host slots and core binding. The workload
includes directory create/stat/rename/remove and zero-byte file
create/stat/open-close/remove, plus tree create/remove. mdtest warns that with
`-e 0`, its “File read” phase only opens and closes files; it transfers no file
payload. The pilot confirmed these phase labels and native rate/time summaries
in `stdout.txt`. Each pilot case had one iteration, so its reported zero
standard deviation is not a variance estimate. Treat the pilot as a functional
and timing check, not a replicated performance result.

After placing the current code in `~/pfs` on anjuna3, start the pilot:

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
per-case timeout; full cases have a one-hour timeout. There is no allocation-
wide deadline or automatic recovery of a failed case in this narrowly scoped
runner.

## After the pilot

`metadata-pilot-01` completed all four cases with return code 0, empty stderr,
and successful cleanup. The native mdtest output is captured unchanged; the
runner does not parse or plot phase summaries. Review those raw summaries
before treating the full run as a performance result.

To run the full matrix on anjuna3:

```bash
python3 -B scripts/microbenchmarks/metadata/run_mdtest.py \
  --full --run-id metadata-full-01
```

The successful final line should report `full: 90/90 mdtest exits and
cleanups`. Confirm there are 90 case result records with:

```bash
find "$HOME/pfs/results/microbenchmarks/runs/metadata-full-01/cases" -mindepth 2 -maxdepth 2 -type f -name result.json | wc -l
```

No metadata visualizer exists yet; analysis/plotting is a later
step, using the native stdout files as the source of phase rates and times.
Keep the run directory intact. From the laptop, copy it directly from anjuna3
into the matching local `results/microbenchmarks/runs/` directory:

```bash
RUN_ID=metadata-full-01
rsync -a --ignore-existing -e 'ssh -J dashlab@lab.dashlab.in' \
  "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" \
  /home/arelius/projects/pfs/results/microbenchmarks/runs/
rsync -ai --checksum --dry-run -e 'ssh -J dashlab@lab.dashlab.in' \
  "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" \
  /home/arelius/projects/pfs/results/microbenchmarks/runs/
```
