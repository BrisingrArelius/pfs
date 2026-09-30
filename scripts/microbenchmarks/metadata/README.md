# BeeGFS metadata benchmark

`run_mdtest.py` is the only metadata runner. Run it as `pfs` from `~/pfs` on
**anjuna3**. It uses the existing `/mnt/beegfs` mount and does not change
global BeeGFS pool membership, caches, or NetBench mode. For each owned case,
it sets and verifies that case's work-directory pattern to one configured
singleton pool. Only the user runs cluster commands.

The call chain is `run_mdtest.py` → set one-target directory pattern → MPICH
`mpirun` → one `mdtest` process per rank → Linux/BeeGFS client → BeeGFS
metadata service and backing storage.
Anjuna3 also hosts the metadata service, so its local-client case is labelled
separately. These are end-to-end zero-byte namespace operations, not pure
metadata-server CPU or data-throughput measurements.

| Mode | Cases | Items per rank | Repetitions |
|---|---:|---:|---:|
| Pilot | 8 | 1,000 | 1 per selected case |
| Full | 180 | 10,000 | 5 |

The pilot repeats four client/rank/layout configurations once for each target
class: anjuna2/1-rank/flat, anjuna3/1-rank/flat,
both-clients/1-rank/per-rank, and both-clients/4-rank/flat, each on HDD and
SSD singleton pools. The full plan is 3 placements × 3 rank counts per
participating client (1, 4, 16) × 2 layouts (flat, per-rank) × 2 target classes
× 5 repetitions, shuffled and rotated deterministically. Pair order is balanced
within each repetition (nine HDD-first and nine SSD-first pairs); because the
configuration order rotates, each full configuration gets both orders across
its five repetitions (a 2/3 split that alternates by configuration). Pilot
pair order alternates across its four configurations. The original four-case
pilot and 90-case full run were unpinned historical baselines; their pilot MPI
durations (`ended_at - started_at`) summed to 28.84 seconds. Correction: in the installed
mdtest output, the section labelled `time (in ms/op)` actually contains phase
elapsed times in seconds, not time per operation. The visualizer derives
throughput-normalized time per operation from each invocation's aggregate
`ops/s` (`1000 / ops/s` gives ms/op), then averages those per-invocation values
across repetitions. For example, 806,442 ops/s corresponds to 0.00124 ms/op
(about 1.24 microseconds/op); the raw `0.001` in the mislabeled time section
is about 0.001 seconds for the phase. The full run sets `-n 10000` items per
rank; each mdtest phase reports its own aggregate operation rate. Five
separate invocations (`-i 1` each) repeat each configuration.

Before running, set `HDD_POOL_NAME` and `SSD_POOL_NAME` near the top of
`run_mdtest.py` to the exact, single-token descriptions of the singleton pools
you created. The runner requires the HDD pool to contain only target 101 and
the SSD pool to contain only target 104. It discovers their live pool IDs;
there is no pool ID to edit. Each case applies RAID0, one target, 512-KiB
chunks, and that pool to its owned work directory, then confirms the resulting
pattern with `beegfs-ctl --getentryinfo`. `placement.json` preserves the command,
pool membership and verified directory pattern. The runner never creates,
deletes, or changes global pool membership.

This factor compares metadata-operation rates in work directories whose
inherited file-layout pattern requests the singleton pool for target 101 or
104. The runner verifies the directory's inherited pattern and the pool's
singleton membership before each case, then snapshots membership again after
mdtest. It does not inspect every mdtest-created file's realized target. More
importantly, this assignment controls storage-target layout for file data; it
does not move BeeGFS inode/dentry metadata from the metadata service onto an
HDD or SSD storage target. With `-w 0`, no file payload is written. Therefore
do not interpret the comparison as direct HDD-versus-SSD metadata-media
performance, data bandwidth, or durable-write latency. The inode/dentry work
still uses the metadata service and its backing filesystem.

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
  --pilot --run-id metadata-pilot-targeted-01
```

The script first checks the live mount, mdtest/MPICH help and binary hashes,
NetBench off, that both MPI clients see the BeeGFS marker and same mdtest
binary, and that the named pools are singleton pools containing the expected
targets. It invokes `/usr/sbin/beegfs-ctl` through `sudo -n` for pool listing,
directory-pattern assignment and verification, so the `pfs` account must have
non-interactive sudo permission for that command. Check it before starting:

```bash
sudo -n /usr/sbin/beegfs-ctl --cfgFile=/etc/beegfs/beegfs-client.conf --liststoragepools
```

It launches one case at a time and captures unchanged native `stdout.txt` and
`stderr.txt`, exact `command.json`, verified `placement.json`, preflight,
before/after mount and storage-pool records, and `result.json` under
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

The earlier unpinned `metadata-pilot-01` completed all four cases, and the
un-pinned 90-case `metadata-full-01` completed and was plotted. They remain
valid as the original baseline, but are not part of the new target-assignment
comparison. The targeted pilot/full runs must use new run IDs.

To run the full matrix on anjuna3:

```bash
python3 -B scripts/microbenchmarks/metadata/run_mdtest.py \
  --full --run-id metadata-full-targeted-01
```

The successful final line should report `full: 180/180 mdtest exits and
cleanups`. Confirm there are 180 case result records with:

```bash
find "$HOME/pfs/results/microbenchmarks/runs/metadata-full-targeted-01/cases" -mindepth 2 -maxdepth 2 -type f -name result.json | wc -l
```

After copying a completed run to the laptop, plot its raw native summaries:

```bash
python3 -B scripts/microbenchmarks/metadata/visualize_results.py results/microbenchmarks/runs/metadata-full-targeted-01
```

The visualizer refuses incomplete or failed runs, checks that the exact pilot
or full matrix and each saved MPI/mdtest command match the plan, and verifies
the stdout/stderr hashes against `result.json`. It writes
`plots/mean_rate.png` and `plots/mean_time_per_op.png` below the run directory.
Both figures have ten panels, one per mdtest phase, and one point per
configuration and target class. The rate figure averages mdtest's reported
aggregate rate across that configuration's invocations. The time figure converts each
invocation's aggregate rate to `1000 / ops/s` ms/op, then averages those
per-invocation values across repetitions. This is throughput-normalized time
per operation, not an individual syscall latency or per-rank latency. X-axis
labels use `a2`/`a3`/`dual`, rank count, and `F` (flat) or `R` (per-rank).
Colors identify client placement, marker shape identifies layout, and open vs
filled markers identify HDD vs SSD target assignment. The raw
`time (in ms/op)` table is retained and validated but is not used for the
time plot because this installed mdtest reports phase elapsed seconds there
under a misleading heading. A single pilot point is just one invocation; the
five full-run invocations provide the across-run mean. Axes are logarithmic.
The visualizer leaves all raw files unchanged and warns if any case has
nonempty stderr.

Keep the run directory intact; the native stdout remains the authoritative
source for later interpretation.

For results produced on anjuna2, first stage the run onto anjuna3. Run these
commands on anjuna3; set `RUN_ID` to the exact run directory name:

```bash
RUN_ID=cache-native-full-01
rsync -a --ignore-existing "pfs@anjuna2:/home/pfs/tejas/pfs/results/microbenchmarks/runs/$RUN_ID" "$HOME/pfs/results/microbenchmarks/runs/"
rsync -ai --checksum --dry-run "pfs@anjuna2:/home/pfs/tejas/pfs/results/microbenchmarks/runs/$RUN_ID" "$HOME/pfs/results/microbenchmarks/runs/"
```

The metadata runner writes its raw results on anjuna3, so metadata runs skip
that first hop. From the laptop, copy the run from anjuna3 into the local
`results/microbenchmarks/runs/` directory, then verify with a checksum dry run:

```bash
RUN_ID=metadata-full-targeted-01
rsync -a --ignore-existing -e 'ssh -J dashlab@lab.dashlab.in' "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" /home/arelius/projects/pfs/results/microbenchmarks/runs/
rsync -ai --checksum --dry-run -e 'ssh -J dashlab@lab.dashlab.in' "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" /home/arelius/projects/pfs/results/microbenchmarks/runs/
```

An empty checksum dry-run means the destination matches the source. If it lists
differences, inspect them before interpreting the local copy; `--ignore-existing`
intentionally does not replace files already present.
