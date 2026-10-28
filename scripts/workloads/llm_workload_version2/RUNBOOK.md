# Runbook: throttled F/S rank-balance experiment

How to run the eval checkpoint read matrix 3 times, then average and plot it.
The reasoning behind the design is in [experiment_context.md](experiment_context.md).

Everything runs on **anjuna2**, from this folder:

```bash
cd ~/advay/pfs/scripts/workloads/llm_workload_version2
```

## What gets measured

`llama31_8b` (Adam, `per-rank`, saved as TP4/PP2/DP1, giving 8 files of ~10 GB each) is read
as an eval job with `TP2`. Each rank reads 4 files in a fixed order. Each file comes from the
**F** copy (pool 11, 20 MiB/s per SSD), the **S** copy (pool 12, 5 MiB/s per SSD) or the
**D** copy (striped over all 12 SSDs):

| Config | Rank 0 (files 4 5 0 1) | Rank 1 (files 6 7 2 3) | Files 0–7 come from | Estimate |
|---|---|---|---|---|
| `allF` | F F F F | F F F F | F F F F F F F F | ~2–2.5 min |
| `allS` | S S S S | S S S S | S S S S S S S S | ~8–10 min |
| `unbalanced` | F F F F | S S S S | F F S S F F S S | ~8–10 min |
| `balanced` | F F S S | F F S S | S S S S F F F F | ~5–6.3 min |
| `onefast` | F S S S | S S S S | S S S S F S S S | ~8–10 min |
| `D` | mixed | mixed | D D D D D D D D | ~7.5–8.5 min |

The matrix is defined once, in `CONFIGS` / `PLACEMENT` in [env_v2.sh](env_v2.sh).

## Scripts

| Script | Does | Time |
|---|---|---|
| `00_preflight.sh` | read-only checks: pools, throttles, cache drop, Darshan, copies, other load | 1 min |
| `01_populate_FS.sh` | writes copy_F into pool 11 and copy_S into pool 12 | ~15–20 min |
| `02_populate_D.sh` | builds a temporary `llm_D` pool, writes copy_D into it, then puts the targets back in 11 and 12 | ~10 min |
| `03_placement.sh` | records each file's real targets and checks F files are only on F targets, and so on | 1 min |
| `04_make_configs.sh` | builds the 6 symlink config dirs plus `configs.json` and per-config target maps | seconds |
| `05_run_matrix.sh` | runs every config × `REPS`, shuffled within each rep, with a cache drop before each run; resumable | ~2.5–3 h for 3 reps |
| `06_analyze.sh` | averages over reps, writes tables and 8 figures | 1 min |
| `99_restore.sh` | clears throttles and returns pools to their pre-experiment state | 1 min |

Where things go (`BEEGFS_ROOT` defaults to `/mnt/beegfs/$USER/ckpt_experiment`):

```
$BEEGFS_ROOT/llmv2_ckpt/copy_{F,S,D}/        the three checkpoint copies (~80 GB each)
$BEEGFS_ROOT/llmv2_ckpt/cfg_<config>/        .metadata + 8 symlinks into the copies
$BEEGFS_ROOT/llmv2_results/                  runs.tsv, per-run traces, configs.json, target maps
$BEEGFS_ROOT/llmv2_results/analysis/         averaged tables and figures
```

---

## Step 0: Before you start

1. **Agree a time window** with the other cluster users. The throttles slow every reader of
   those 12 SSDs. Pool 6 currently holds only 206, and Default holds only 401–403 (HDD).
2. **Use tmux**, because the matrix takes hours and must survive an ssh drop:
   ```bash
   tmux new -s llmv2        # reattach later with: tmux attach -t llmv2
   ```
3. **Set up passwordless cache dropping (one time).** The cache must be dropped on anjuna2 and
   on colva1–4 before every run. If only the client is dropped, the colvas serve the
   checkpoint from their RAM, which **bypasses the throttle**. Install the wrapper on
   anjuna2 *and* on each colva:
   ```bash
   install_drop='
     printf "#!/bin/sh\nsync; echo 3 > /proc/sys/vm/drop_caches\n" | sudo tee /usr/local/sbin/drop-caches >/dev/null
     sudo chmod 755 /usr/local/sbin/drop-caches
     echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/drop-caches" | sudo tee /etc/sudoers.d/dropcaches >/dev/null
     sudo chmod 440 /etc/sudoers.d/dropcaches'
   bash -c "$install_drop"                                        # anjuna2
   for h in colva1 colva2 colva3 colva4; do ssh -t $h "$install_drop"; done
   ```
   Then check that it works without a password prompt:
   ```bash
   for h in colva1 colva2 colva3 colva4; do
     ssh -o BatchMode=yes $h 'grep ^Cached: /proc/meminfo; sudo -n /usr/local/sbin/drop-caches && grep ^Cached: /proc/meminfo'
   done
   ```
   The second `Cached:` figure on each host should be much smaller than the first.

### Dropping caches by hand

`05_run_matrix.sh` does this automatically before every run, and aborts if any host fails.
To do it yourself:

```bash
# client (anjuna2)
sync; sudo /usr/local/sbin/drop-caches          # or: echo 3 | sudo tee /proc/sys/vm/drop_caches

# servers (colva1-4): this is the part that matters for the throttle
for h in colva1 colva2 colva3 colva4; do
  ssh $h 'sync; sudo /usr/local/sbin/drop-caches; echo "$(hostname): $(grep ^Cached: /proc/meminfo)"'
done
```

Without the wrapper, run `ssh -t $h 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'` on
each host.

## Step 1: Check the throttles and preflight

The throttles are `--runtime` settings, so a reboot or a `beegfs-storage` restart on any colva
removes them. Check them at the start of every session:

```bash
./throttle.sh show      # 6 lines rbps=20971520 (F) and 6 lines rbps=5242880 (S)
./throttle.sh apply     # only if any are missing
./00_preflight.sh
```

Preflight must end with `preflight OK`. Copies and configs show as `not populated` /
`not built` until Steps 2–5 are done, and that's expected. It also samples
`beegfs-storage` read traffic on each colva for 5 s. When nobody is reading, all four should
show ~0 MiB/s.

## Step 2: Populate the F and S copies

```bash
./01_populate_FS.sh
```

This writes ~80 GB into pool 11 (`copy_F`) and ~80 GB into pool 12 (`copy_S`), then verifies a
sample of tensors from each. Writes are not throttled. The verify reads are throttled, so
the S verify takes a few minutes. A copy that already exists is skipped.

To redo just one copy: `rm -rf $BEEGFS_ROOT/llmv2_ckpt/copy_S && COPIES=S ./01_populate_FS.sh`.

## Step 3: Populate the D (mixed pool) copy

```bash
./02_populate_D.sh       # asks for confirmation; YES=1 skips the prompt
```

This removes all 12 targets from pools 11 and 12, creates `llm_D` with all 12, writes
`copy_D`, and then moves the targets back to 11 and 12. The move-back runs on exit even if
the populate fails. Files never move when pool membership changes, so `copy_F` and `copy_S`
are unaffected.

Check that the pools are back afterwards:

```bash
sudo beegfs-ctl --liststoragepools    # 11 llm_F 104,105,205,305,404,405 | 12 llm_S 106,107,306,307,406,407 | llm_D empty
```

If the script was killed hard (`kill -9`, lost node) and the trap didn't run, restore the pools by hand:

```bash
sudo beegfs-ctl --modifystoragepool --id=<llm_D id> --rmtargets=104,105,205,305,404,405,106,107,306,307,406,407
sudo beegfs-ctl --modifystoragepool --id=11 --addtargets=104,105,205,305,404,405
sudo beegfs-ctl --modifystoragepool --id=12 --addtargets=106,107,306,307,406,407
```

## Step 4: Verify the placement

```bash
./03_placement.sh
```

Every line should end in `ok`: each F file is on 4 F targets and each S file on 4 S targets.
For D, the script prints how many of the 8 files have no S target. Expect 0 or 1, because only
~3% of random 4-wide stripes avoid S. The target maps are saved as
`llmv2_results/targets_copy_{F,S,D}.json`.

## Step 5: Build the configurations

```bash
./04_make_configs.sh
```

This checks that the three `.metadata` files are identical (`cmp`), then builds
`cfg_allF … cfg_D`. The last block prints which copy each config's files 0–7 resolve to. It
must match the "Files 0–7 come from" column of the table above.

## Step 6: Smoke test (one run, ~2.5 min)

```bash
RUN_CONFIGS=allF REPS=1 ./05_run_matrix.sh
```

Then check:

```bash
R=$BEEGFS_ROOT/llmv2_results
T=$(awk -F'\t' '$1=="allF" && $6=="ok" {t=$3} END {print t}' $R/runs.tsv)
# 1. Darshan saw the checkpoint reads (should be ~16 GB, 8 shard files)
python3 -c "import json;d=json.load(open('$R/$T.darshan.json'));print(d['bytes_read']/1e9,'GB',d['files_touched'],'files')"
# 2. Reads resolved to real F target IDs (104 105 205 305 404 405; not stripe slots 0-3)
python3 -c "import csv,collections;c=collections.Counter();[c.update(r['targets'].split(',')) for r in csv.DictReader(open('$R/$T.ops.csv'))];print(c)"
# 3. Bandwidth sane: all-F should be <= ~120 MiB/s (6 SSDs x 20). ~280 means a throttle or cache drop failed.
./06_analyze.sh
```

If check 1 shows ~0 bytes, Darshan didn't match the paths. Look at the file names in
`$R/$T.darshan.json` and report it. The smoke run is kept as `allF` rep 1. To discard it
instead, delete its line from `runs.tsv`.

## Step 7: Run the full matrix

```bash
./05_run_matrix.sh                  # REPS=3 by default
```

- Each rep runs all 6 configs in a new random order, which is printed and logged in
  `llmv2_results/matrix_<date>.log`.
- Before **each** run it checks that all 12 throttles are present, drops caches on anjuna2 and
  colva1–4, and aborts if either step fails.
- A failed read is recorded as `fail(rc)` in `runs.tsv` and the matrix continues.
- **Resuming:** run the same command again. `(config, rep)` pairs already marked `ok` are
  skipped.
- More reps later: `REPS=5 ./05_run_matrix.sh` only runs reps 4 and 5.

Progress:

```bash
column -t -s $'\t' $BEEGFS_ROOT/llmv2_results/runs.tsv
```

## Step 8: Analyze and plot

On anjuna2:

```bash
./06_analyze.sh
```

On a laptop, if anjuna2 lacks pandas/matplotlib:

```bash
scp -r anjuna2:/mnt/beegfs/pfs/ckpt_experiment/llmv2_results ./
scripts/workloads/llm_workload_version2/06_analyze.sh llmv2_results
```

Outputs in `llmv2_results/analysis/`:

| File | Content |
|---|---|
| `matrix_summary.md` / `.csv` | per config: mean ± sd (also min/max/count) of load time, per-rank times, bandwidth, latency |
| `run_metrics.csv` | one row per run |
| `latency_by_medium.csv` | per run × read class × medium (F / S / F+S) |
| `file_times.csv` | per run × rank × file: start, duration, MiB/s |
| `fig1_load_time.png` | load time per config, mean ± sd with a dot per rep, over the estimated range |
| `fig2_rank_completion.png` | rank 0 vs rank 1 finish time (balance) |
| `fig3_bandwidth.png` | aggregate and per-rank read bandwidth, against the F and S pool ceilings |
| `fig4_latency.png` | mean and p95 latency of framing reads and payload reads |
| `fig5_latency_by_medium.png` | the same split by whether a read touched F, S or both |
| `fig6_latency_cdf.png` | payload latency CDF, one line per config |
| `fig7_throughput_timeline.png` | client MiB/s over time, each rep plus the mean |
| `fig8_file_gantt.png` | which file each rank read when, for the median rep |

Definitions:
- **Load time:** the slowest rank's `dcp.load` time, which is what the job sees.
- **Rank N time:** from the first read to rank N's last read.
- **Framing read:** a read of one 512 KiB chunk or less, served by a single target.
- **Payload read:** anything larger, striped across the file's 4 targets.
- **Latency:** the duration of one blocking read in the Darshan DXT trace.

## Step 9: Restore the cluster

When all runs are done and analyzed:

```bash
./99_restore.sh            # clears throttles, returns 104-107/205/305-307 to pool 6, 404-407 to Default,
                           # removes the empty llm_F/llm_S/llm_D pools (REMOVE_POOLS=0 to keep them)
sudo beegfs-ctl --liststoragepools
# expected: 1 Default 401-407 | 3 hdd ... | 4 offline_pool 207 | 6 ssd 104-107,205,206,305-307
```

Delete the copies when you no longer need them (~240 GB):

```bash
rm -rf $BEEGFS_ROOT/llmv2_ckpt
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FAIL: throttles: n/6 F …` | A colva rebooted or `beegfs-storage` restarted. Run `./throttle.sh apply`, then rerun `05` (it resumes). |
| `FAIL: cache drop did not succeed everywhere` | Wrapper or sudoers rule missing on a host (Step 0.3). `ALLOW_WARM=1` overrides, but the numbers are then invalid. |
| All F runs at ~260–280 MiB/s | Reads are not hitting the disks. Either the throttle is missing or the server cache was not dropped. |
| `pool 11 has [...], want [...]` | Pools are not in the expected state, for example after an interrupted `02_populate_D.sh`. Restore them by hand (Step 3). |
| `03_placement.sh` prints `NO TARGETS` | `beegfs-ctl --getentryinfo` failed; try it by hand on that file. |
| `beegfs-ctl` refuses pool changes | The Enterprise trial for storage pools may have expired. |
| All S slower than ~10 min | Server read-ahead may be using up throttle budget on bytes nobody asked for. See Known issues in experiment_context.md. |
