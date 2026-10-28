# Throttled F/S pool experiment: rank balance for LLM eval checkpoint reads

Status as of 2026-10-28: throttles applied; pools llm_F (11) and llm_S (12)
created (the D phase was skipped at pool-creation time; `02_populate_D.sh` builds
llm_D temporarily instead). Run scripts written; no experiment runs yet. How to
run: [RUNBOOK.md](RUNBOOK.md).

## Question

During an eval checkpoint read, does it matter **how** a limited amount of fast
storage is distributed across the files of a checkpoint?

Hypotheses, from the existing eval-read traces (`../llm_workload/run/results/`):

1. **Balanced beats unbalanced.** Total load time is `max(rank 0 time, rank 1 time)`.
   Ranks do not synchronise between files, so fast media only helps if every rank
   gets an equal share of it. Giving all fast files to one rank gains nothing.
2. **A mixed-media pool performs close to all-slow.** With stripe width 4, a
   payload read finishes when its slowest target does. A random stripe drawn from
   a mixed pool almost always contains a slow target, so per-file speed is set by
   the slow media. Randomness does not "average out" to balanced.
3. Random per-file placement (each file wholly fast or wholly slow) is expected to
   be ~15% worse than balanced on average, because random assignments often leave
   one rank with fewer fast files.

## Why throttled NVMe instead of real HDD vs NVMe

- The client (anjuna2) has a single 2.5 GbE link: iperf ~2.35 Gbit/s ≈ **280 MiB/s**.
  Existing eval reads on the Default and HDD pools both ran at 264–269 MiB/s,
  i.e. network-bound and blind to media.
- Filling an HDD to make it "slow" is uncontrolled (track position and
  fragmentation depend on the allocator), slow to set up, and pushes the target
  into BeeGFS's low/emergency capacity class.
- cgroup v2 read-bandwidth limits on the `beegfs-storage` service give an exact,
  reversible fast/slow ratio on identical hardware, and keep the whole experiment
  well below the network cap.

## Workload being measured

Same as the earlier eval-read runs:

| Setting | Value |
|---|---|
| Model | `llama31_8b`, Adam optimizer state, `per-rank` layout |
| Saved as | TP4/PP2/DP1, giving 8 files `__0_0`..`__7_0` of ~10 GB each, plus `.metadata` |
| Loaded as | `TP2` (2 ranks), eval read set (weights only, ~20% of each file) |
| Stripe | 4 targets, 512 KiB chunks |

What each rank reads (file numbers are the `__N_0` index; the order is the same every run):

| Rank | Files in read order |
|---|---|
| 0 | 4, 5, 0, 1 |
| 1 | 6, 7, 2, 3 |

Per file: ~2.25 GB read, ~576 framing reads of 512 KiB (one target each) and
113 payload reads averaging ~17 MB (striped across all 4 targets). Every file has
the same access pattern; no file is intrinsically a better candidate for fast
media.

## Schema

### Throttles (read bandwidth only; writes are unthrottled)

| Class | Limit per SSD | Targets |
|---|---|---|
| **F** | 20 MiB/s (20971520 B/s) | 104, 105, 205, 305, 404, 405 |
| **S** | 5 MiB/s (5242880 B/s) | 106, 107, 306, 307, 406, 407 |
| excluded, unthrottled | — | 206 (dropped to keep F and S equal), 207 (in `offline_pool`) |

Target-to-device mapping as observed on 2026-10-28. `throttle.sh` resolves
devices on each host at apply time, because the numbering does not follow target
order and can change after a reboot.

| Host | Target → device (MAJ:MIN) |
|---|---|
| colva1 | 104 → nvme1n1 (259:0) F · 105 → nvme0n1 (259:1) F · 106 → nvme2n1 (259:2) S · 107 → nvme3n1 (259:3) S |
| colva2 | 205 → nvme1n1 (259:7) F |
| colva3 | 305 → nvme1n1 (259:0) F · 306 → nvme2n1 (259:2) S · 307 → nvme3n1 (259:1) S |
| colva4 | 404 → nvme0n1 (259:1) F · 405 → nvme1n1 (259:3) F · 406 → nvme2n1 (259:0) S · 407 → nvme3n1 (259:2) S |

Why 20 and 5 MiB/s:

- **Network headroom:** the worst case is 2 ranks × 4 stripe targets, all in payload reads on F
  targets, which is 8 × 20 = 160 MiB/s, about 57% of the 280 MiB/s link.
- **Latency:** a 512 KiB read takes at least ~25 ms on F and ~100 ms on S. The
  network adds about 2 ms, so the throttle sets latency as well; no separate
  latency throttle is needed.
- **Ratio:** 4:1 separates the predictions clearly without making runs very long.

### Storage pools

| Pool | Targets | Purpose |
|---|---|---|
| `llm_D` | the 12 F + S targets | mixed pool, used to populate the "default pool" copy |
| `llm_F` | 104, 105, 205, 305, 404, 405 | fast copy |
| `llm_S` | 106, 107, 306, 307, 406, 407 | slow copy |

A target can belong to only one pool, so D and F/S exist at different times.
Pool membership only affects where **new** files are placed, and throttles apply
per device regardless of pool. So the D copy keeps its mixed placement after the
pools are split.

Pool state before the experiment (2026-10-28), needed for the restore step:

| Pool ID | Name | Targets |
|---|---|---|
| 1 | Default | 401, 402, 403, 404, 405, 406, 407 |
| 3 | hdd | 101–103, 201–204, 301–304 |
| 4 | offline_pool | 207 |
| 6 | ssd | 104, 105, 106, 107, 205, 206, 305, 306, 307 |

Pools 2, 5, 7, 8, 9 and 10 are empty.

## Experiment matrix

| Config | Rank 0 files (4, 5, 0, 1) | Rank 1 files (6, 7, 2, 3) | Predicted run time |
|---|---|---|---|
| All F | F F F F | F F F F | ~2–2.5 min |
| All S | S S S S | S S S S | ~8–10 min |
| Unbalanced | F F F F | S S S S | ≈ All S |
| Balanced | F F S S | F F S S | ~5–6.3 min |
| One file fast | F S S S | S S S S | ≈ All S |
| Default (D copy) | mixed | mixed | ~7.5–8.5 min (≈ All S) |

How these estimates were made:

- **Per-file times:** F ≈ 30–38 s and S ≈ 121–150 s. The range comes from whether
  repeated framing reads hit the client cache.
- **D:** only C(6,4)/C(12,4) = 15/495 ≈ 3% of random 4-wide stripes have no S
  target, so almost every D file runs at S speed for its payload reads.
- **Contention not included:** with 6 targets per pool and stripe width 4, the two
  ranks' concurrent files always share at least 2 targets, which splits those
  targets' limits. Expect extra time, worst in All F and All S.

## Procedure (all from anjuna2)

Scripts are in this folder. The workload scripts are in `../llm_workload/run/`.

### 0. Coordinate first

These steps are cluster-wide changes:

- **Throttles** slow every reader of those SSDs.
- **Removing targets from pool 6** leaves `ssd` with only 206.
- **Moving 404–407 out of Default** leaves Default HDD-only, so other people's new
  files land differently.
- **Other users' data is on these targets:** 307 has ~390 GB; 104–106 and 306 also
  hold data.

Agree a time window with the other users.

### 1. Throttles

```bash
./throttle.sh show     # expect no rbps= lines
./throttle.sh apply    # check every "target -> device rbps=" line
./throttle.sh show     # rbps=20971520 / 5242880 against the right MAJ:MIN
```

Verified on 2026-10-28. Reading a 200 MiB single-target file through BeeGFS
with cold caches gave 20.0 MiB/s on 105 (F) and 5.0 MiB/s on 106 (S).

### 2. Pools: D phase

```bash
sudo beegfs-ctl --modifystoragepool --id=6 --rmtargets=104,105,106,107,205,305,306,307
sudo beegfs-ctl --addstoragepool --desc=llm_D \
  --targets=104,105,106,107,205,305,306,307,404,405,406,407
sudo beegfs-ctl --liststoragepools     # record llm_D's ID
```

Targets removed from a pool go to Default. The `addstoragepool` command takes
them from there.

### 3. Populate the D copy

```bash
cd ../llm_workload/run
export MODEL=llama31_8b SAVE_CONFIG=TP4/PP2/DP1 LOAD_CONFIG=TP2 \
       OSS_HOSTS="colva1 colva2 colva3 colva4"
POOL_ID=<llm_D id> TAG_BASE=llama31_8b_adam_per-rank_llmD ./01_populate.sh
```

The previous 80 GB populate took ~7 minutes.

### 4. Pools: split into F and S

```bash
sudo beegfs-ctl --modifystoragepool --id=<llm_D id> \
  --rmtargets=104,105,106,107,205,305,306,307,404,405,406,407
sudo beegfs-ctl --addstoragepool --desc=llm_F --targets=104,105,205,305,404,405
sudo beegfs-ctl --addstoragepool --desc=llm_S --targets=106,107,306,307,406,407
sudo beegfs-ctl --liststoragepools     # record llm_F and llm_S IDs
```

### 5. Populate the F and S copies

```bash
POOL_ID=<llm_F id> TAG_BASE=llama31_8b_adam_per-rank_llmF ./01_populate.sh
POOL_ID=<llm_S id> TAG_BASE=llama31_8b_adam_per-rank_llmS ./01_populate.sh
```

Check placement on a few files with `sudo beegfs-ctl --getentryinfo <file>`.
The target lines in that output are indented (`  + 105 @ colva1`).

### 6. Build the configurations as symlink directories

Each configuration is a directory holding the `.metadata` file and 8 symlinks,
each pointing at the F or S copy of that file. DCP opens files by name. **This is
untested:** confirm on one configuration that the eval read works and that the
traced reads resolve to the intended copy.

```bash
R=/mnt/beegfs/$USER/ckpt_experiment
Fd=$R/llama31_8b_adam_per-rank_llmF; Sd=$R/llama31_8b_adam_per-rank_llmS
mkcfg() {   # mkcfg <name> <pool for file 0> ... <pool for file 7>   (F or S)
  local d=$R/cfg_$1; shift; local p=("$@"); mkdir -p "$d"; ln -sf "$Fd/.metadata" "$d/.metadata"
  for i in 0 1 2 3 4 5 6 7; do
    local src=$Fd; [[ ${p[i]} == S ]] && src=$Sd
    ln -sf "$src/__${i}_0.distcp" "$d/__${i}_0.distcp"
  done
}
#           files: 0 1 2 3 4 5 6 7
mkcfg allF         F F F F F F F F
mkcfg allS         S S S S S S S S
mkcfg unbalanced   F F S S F F S S   # rank 0 (0,1,4,5) F, rank 1 (2,3,6,7) S
mkcfg balanced     S S S S F F F F   # each rank: first two files F, last two S
mkcfg onefast      S S S S F S S S   # rank 0's file 4 only
```

With these patterns:

- **balanced:** each rank gets two F files and two S files (rank 0: 4, 5 F and 0, 1 S;
  rank 1: 6, 7 F and 2, 3 S).
- **unbalanced:** all of rank 0's files are F and all of rank 1's are S.

The two `.metadata` files should be identical, because the F and S copies are the
same checkpoint. Verify with `cmp $Fd/.metadata $Sd/.metadata`.

### 7. Run

```bash
cd ../llm_workload/run
export MODEL=llama31_8b SAVE_CONFIG=TP4/PP2/DP1 LOAD_CONFIG=TP2 \
       OSS_HOSTS="colva1 colva2 colva3 colva4"
for rep in 1 2 3; do
  for cfg in $(printf '%s\n' allF allS unbalanced balanced onefast D | shuf); do
    if [[ $cfg == D ]]; then dir=$R/llama31_8b_adam_per-rank_llmD; else dir=$R/cfg_$cfg; fi
    CKPT_DIR=$dir TAG_BASE=llmv2_$cfg ./02_eval_read.sh
  done
done
```

- **Cache drops:** `02_eval_read.sh` drops caches on the client and on `OSS_HOSTS`
  before each read. Without `OSS_HOSTS`, reads come from the servers' RAM and skip
  the throttle.
- **Order:** shuffle the configurations within each repetition so drift does not
  line up with one configuration.
- **Repetitions:** 3 reps take ~2.5–3 h and 5 reps ~4–4.5 h, plus ~20–25 min of populate.

### 8. Restore

```bash
./throttle.sh clear
sudo beegfs-ctl --modifystoragepool --id=<llm_F id> --rmtargets=104,105,205,305,404,405
sudo beegfs-ctl --modifystoragepool --id=<llm_S id> --rmtargets=106,107,306,307,406,407
sudo beegfs-ctl --modifystoragepool --id=6 --addtargets=104,105,106,107,205,305,306,307
# 404-407 end up back in Default (pool 1), where they started; 206 never left pool 6
```

Then delete the checkpoint copies and the `cfg_*` directories when they are no
longer needed.

## Validity checklist

- [ ] Throttles show the expected `rbps` on all 12 devices just before each session.
      They are `--runtime`, so a reboot or service restart clears them.
- [ ] Caches are dropped on the client and on colva1–4 before every run.
- [ ] Each All F run keeps the client at or below ~160 MiB/s. Seeing ~280 MiB/s
      means a throttle or a cache drop failed.
- [ ] Each copy's real target placement is recorded with `make_target_map.sh`,
      once its regex is fixed (see Known issues).
- [ ] No other heavy users are on the throttled SSDs during runs.
- [ ] The analysis reports per-rank completion time as well as the total.

## Known issues

- **`make_target_map.sh` never matches.** The script in `../llm_workload/run/` uses
  `awk '/^\+ *[0-9]+ @/'`, but `beegfs-ctl --getentryinfo` indents target lines
  (`  + 105 @ colva1`). It therefore returns no targets, which is why the earlier
  stats show stripe slots instead of target IDs. A pattern that works is
  `grep -oP '^\s*\+\s*\K[0-9]+(?=\s*@)'`.
- **BeeGFS licence.** Storage pools are an Enterprise feature; `beegfs-ctl` warns
  about a 60-day trial. Pool changes may be refused if the trial has expired.
- **Server read-ahead.** Read-ahead on the storage servers may consume throttle
  budget on bytes nobody asked for, which would make S slower than predicted.
- **Earlier write-bandwidth results are wrong.** The legacy Darshan comparison
  divides bytes written by the span between the first and last write timestamp.
  For multi-phase profiles that inflates write bandwidth about 2×. Read bandwidth
  is unaffected. This is not part of this experiment, but don't reuse those write
  numbers.
