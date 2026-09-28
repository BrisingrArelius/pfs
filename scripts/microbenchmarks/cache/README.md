# BeeGFS cache microbenchmark

`run_cache.py` is the single cache benchmark program. It prepares two 8-GiB
files on BeeGFS, one on HDD target 101 and one on NVMe target 104 on `colva1`,
then measures reads from `anjuna2` through its existing BeeGFS mount. `--mode`
must match that mount's effective cache mode; the script does not change it.

| Client mode | Measured states on each medium | Pilot | Full run |
|---|---|---:|---:|
| `buffered` | Backend, server RAM | 4 reads | 20 reads (5 per medium/state) |
| `native` | Backend, server RAM, client RAM | 8 reads (includes a second client-RAM read per medium) | 30 reads (5 per medium/state) |

Each measured read uses one IOR rank, a 1-MiB transfer and an existing 8-GiB
file. Unmeasured warm-up reads are separate from IOR results. The 8-GiB
client-RAM case is **not** claimed in `buffered` mode: BeeGFS's small buffered
client cache cannot hold the complete file.

```text
                       client on anjuna2
                              │
             ┌────────────────┴────────────────┐
             │ client RAM hit (native)          │ miss
             ▼                                  ▼
        application                     network → colva1
                                                │
                                  ┌─────────────┴─────────────┐
                                  │ server RAM hit             │ miss
                                  ▼                            ▼
                              application                HDD 101 / SSD 104
                                                             │
                                                         application
```

## Why IOR?

IOR is the **measured read workload**. It performs a single sequential POSIX
read of the existing 8-GiB BeeGFS file and reports read bandwidth. The IOR
command uses one MPI rank and does **not** use `O_DIRECT`: direct I/O would
change the cache path whose effect this experiment is measuring. In `native`
mode, Linux page cache can hold the file; in `buffered` mode, the script tests
only backend and server-RAM paths. The script checks the chosen mode and never
edits the client configuration or mounts another instance. `dd`
creates the files with direct writes and performs unmeasured warm-up reads;
its timing is not reported as cache benchmark throughput. This measures a
fixed synthetic read, not application performance or pure device speed.

The actual measured command, with paths filled in for each case, is:

```text
/usr/bin/mpirun -np 1 /usr/local/bin/ior \
  -a POSIX -r -E -k -g -t 1m -b 8g -s 1 -i 1 \
  -o <existing-8-GiB-file> \
  -O summaryFormat=JSON -O summaryFile=<case-directory>/ior.json
```

`-r` selects a read; `-E` tells IOR to use the file already prepared by the
script; `-k` keeps that file. `-t 1m` is the transfer size, `-b 8g` the block
per rank, `-s 1` one segment, and `-i 1` one IOR iteration. `-g` enables IOR's
intra-test barriers. Each case is a **separate IOR invocation**; IOR's own
iteration count does not stand for the five experiment repetitions. The native
IOR JSON provides the reported read MiB/s, while host counters establish the
data path. There is no `--posix.odirect` option on the measured command.

## What happens when it runs

1. Check that the existing `/mnt/beegfs` client on `anjuna2` matches the chosen
   `--mode`, that NetBench is off, and that RAM, target health and the
   HDD/NVMe mappings match the experiment.
2. Create one run-owned directory per medium with a one-target stripe pattern.
   Check each newly created file's actual target **before** filling it with
   8 GiB. No cluster-wide storage-pool membership changes occur.
3. For each case, drop client and server caches. An unmeasured read warms the
   file for `server_ram` and, in native mode, `client_ram`. Then drop **only
   the client** cache for `server_ram`, or **only the server** cache for
   `client_ram`. Run IOR once and capture client network bytes, server network
   bytes, and target-device read bytes. Client page residency is sampled in
   `native` mode only.
4. Compare that traffic with the intended path. A read whose path cannot be
   verified is recorded as `unverified`, not called a cache hit. Remove only
   this run's BeeGFS files on normal exit.

The 8-GiB files are filled with direct `dd` writes and an fsync *before* the
read cases. The script sets one desired stripe on each run-owned directory,
creates empty candidate files, checks BeeGFS's **actual target ID** for each,
and fills only a file assigned to 101 or 104. It does not move targets between
storage pools. Normal IOR reads and non-direct `dd` warm-ups use the **existing**
BeeGFS mount; the script does not change its caching policy.

### What each cache state means

Every case starts with a fresh cache drop on `anjuna2` and `colva1`. Warm-ups
are outside the measured IOR invocation.

| State in `results.json` | Preparation before IOR | Data path supported by the measurements |
|---|---|---|
| `backend` | Drop both hosts' caches, then measure | Client and server network traffic **and** backing-device reads |
| `server_ram` | Drop both, read the file once to warm it, drop **client only**, then measure | Client and server network traffic, few backing-device reads |
| `client_ram` (`native` only) | Drop both, read the file once to warm it, drop **server only**, then measure | File resident in client RAM, little bulk network or backing-device activity |

Traffic ratios use **IOR's 8-GiB logical read** as the denominator. A verified
backend case requires at least 80% of that amount in each of client-received,
server-sent and device-read bytes. A verified server-RAM case requires at least
80% on both network counters and at most 20% device reads. In `native` mode,
both cases also require at most 10% client page residency. The script does not
use Linux `mincore` to claim how much of BeeGFS's **internal buffered cache** is
resident; `client_residency` is `null` in `buffered` results. A verified
client-RAM case (`native` only) requires at least 95% client residency and at
most 20% on each network and
device counter. The script also checks a one-second pre-read window for
background traffic. These are *observed classifications*, not labels inferred
from the order of cache-drop commands alone.

### Functions in `run_cache.py`

| Function | Role |
|---|---|
| `main` | Parse the run ID, `--mode` and `--pilot`, create a results directory, and call `benchmark`. |
| `benchmark` | Hold the run lock, prepare two files on the existing mount, measure the cases, and clean up. |
| `check_cluster` | Check the selected mode on the existing mount, target health/mapping, NetBench state, sudo access and available RAM before preparing data. |
| `targets` | Ask BeeGFS which storage target actually holds a given file. |
| `make_file` | Select a file on target 101 or 104, fill it with direct writes, and recheck size and placement. |
| `drop` | Clear both hosts' caches at the start of a case, then optionally clear only one host after warm-up. |
| `residency` | In native mode, use Linux `mincore` to check how much of the file is in the client's page cache without reading it. |
| `counters` | Read client-received network bytes, server-sent network bytes and backing-device read bytes. |
| `cases` | Return buffered (4/20) or native (8/30) pilot/full cases; full runs rotate their configuration order. |
| `measure` | Prepare one cache state, run IOR, compare IOR and host evidence, and save the achieved label. |
| `cleanup` | Remove only files and directories bearing this run's owner marker. |
| `run`, `remote`, `ctl` | Execute commands locally or over SSH; `ctl` uses sudo and the installed client's configuration for all `beegfs-ctl` calls. Capture measured-command output. |
| `record` | Atomically publish a JSON record such as the per-case result or run summary. |

## Running it and reading the results

Run from the project checkout on **`anjuna2` as the normal `pfs` user**. Select
the mode already active on the existing mount. The script does not restart or
restore the BeeGFS client. It invokes non-interactive sudo to set the stripe
pattern on its own directories, read BeeGFS target information through
`beegfs-ctl`, and drop the client cache. The CLI is given
`--cfgFile=/etc/beegfs/beegfs-client.conf`, matching the configuration that
works on `anjuna2`; the script does not change its authentication settings.
The server cache drop uses SSH and sudo on `colva1`. A run drops host-wide page
caches, so its measurements require exclusive use of those hosts. Its file lock
only prevents two copies of `run_cache.py` from running at once; it does not
block other cluster workloads.

Buffered pilot (four measured reads) with the current `buffered` mount:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-buffered-pilot-02 --mode buffered --pilot
```

Buffered full run (20 measured reads, new run ID):

```bash
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-buffered-full-01 --mode buffered
```

With the **existing** client separately configured and verified as `native`,
the same script accepts `--mode native`. Its pilot has eight measured reads
and its full run has 30, each using a new run ID:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-native-pilot-01 --mode native --pilot
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-native-full-01 --mode native
```

Each run ID is used once. The terminal prints a bandwidth and `achieved` label
after every completed IOR read. Output is saved under
`results/microbenchmarks/runs/<run-id>/`:

```text
owner.json                  run/user identity, selected mode and pilot flag
results.json                completed cases and their verified/unverified labels
01-hdd-backend/             one example measured case
  command.json              exact IOR argument list
  ior.json                  native IOR summary
  stdout.txt, stderr.txt    native IOR output
  counters.json             before/after network and target-device counters
  result.json               mode, throughput, residency (native only), ratios and achieved label
  warmup/                   present for states with an unmeasured warm-up
```

`results.json` and completed case directories remain if a run stops early;
missing cases do not appear as completed. On normal exit the script removes
its run-owned BeeGFS files while preserving these raw results. There is no
resume command or separate plotting script in this folder.

**Execution status:** the program has not been run or calibrated against the
installed BeeGFS and IOR versions on the cluster. There are no verified cache
results or observed full-run timings in this repository. A run stops without
reporting a verified cache path when its layout, mount, IOR output or traffic
evidence does not match the protocol.

This cache runner does not load Darshan. Its results come from native IOR JSON
and client/server Linux counters; Darshan cannot identify which cache level
served the read.
