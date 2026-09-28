# BeeGFS cache microbenchmark

`run_cache.py` is the single cache benchmark program. It prepares two 8-GiB
files on BeeGFS, one on HDD target 101 and one on NVMe target 104 on `colva1`,
then measures backend, storage-server RAM and client RAM reads from `anjuna2`.
The full protocol is 2 media × 3 states × 5 repetitions = 30 measured reads;
`--pilot` runs eight. Each measured read uses one IOR rank, a 1-MiB transfer,
and an existing file. Unmeasured warm-up reads are separate from IOR results.

```text
                       client on anjuna2
                              │
             ┌────────────────┴────────────────┐
             │ client RAM hit                   │ miss
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
bypass the client page cache whose effect this experiment is measuring. `dd`
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

1. Check the client mount, RAM, NetBench mode, and the health and HDD/NVMe
   mapping of targets 101 and 104.
2. Mount BeeGFS in `native` client-cache mode within a private Linux mount
   namespace. The existing `buffered` mount stays in place.
3. Create one run-owned directory per medium with a one-target stripe pattern.
   Check each newly created file's actual target **before** filling it with
   8 GiB. No cluster-wide storage-pool membership changes occur.
4. For each case, drop client and server caches. An unmeasured read warms the
   file for the server-RAM and client-RAM cases; the server-RAM case then drops
   only the client cache. Run IOR once and capture client network bytes,
   server network bytes, target-device read bytes and client page residency.
5. Compare that traffic with the intended path. A read whose path cannot be
   verified is recorded as `unverified`, not called a cache hit. Remove only
   this run's BeeGFS files and the private mount on normal exit.

The 8-GiB files are filled with direct `dd` writes and an fsync *before* the
read cases. The script sets one desired stripe on each run-owned directory,
creates empty candidate files, checks BeeGFS's **actual target ID** for each,
and fills only a file assigned to 101 or 104. It does not move targets between
storage pools. Normal IOR reads and buffered `dd` warm-ups use the temporary
native-cache mount; that mount is separate from the existing `buffered` mount.

### What each cache state means

Every case starts with a fresh cache drop on `anjuna2` and `colva1`. Warm-ups
are outside the measured IOR invocation.

| State in `results.json` | Preparation before IOR | Data path supported by the measurements |
|---|---|---|
| `backend` | Drop both hosts' caches, then measure | Client and server network traffic **and** backing-device reads |
| `server_ram` | Drop both, read the file once to warm it, drop **client only**, then measure | Client and server network traffic, few backing-device reads |
| `client_ram` | Drop both, read the file once to warm it, leave caches intact, then measure | File resident in client RAM, little bulk network or backing-device activity |

Traffic ratios use **IOR's 8-GiB logical read** as the denominator. A verified
backend case requires at least 80% of that amount in each of client-received,
server-sent and device-read bytes, with at most 10% client residency. A verified
server-RAM case requires at least 80% on both network counters, at most 20%
device reads and at most 10% client residency. A verified client-RAM case
requires at least 95% client residency and at most 20% on each network and
device counter. The script also checks a one-second pre-read window for
background traffic. These are *observed classifications*, not labels inferred
from the order of cache-drop commands alone.

### Functions in `run_cache.py`

| Function | Role |
|---|---|
| `main` | Parse the run ID and `--pilot`, create a results directory, and start the same script with sudo in a private mount namespace. |
| `benchmark` | Hold the run lock, set up the mounted client and two files, measure the cases, and clean up. |
| `check_cluster` | Check BeeGFS mounting, target health/mapping, NetBench state and available RAM before preparing data. |
| `native_mount` | Make a run-local copy of the client config with `native` caching, mount it, and check the effective mode. |
| `targets` | Ask BeeGFS which storage target actually holds a given file. |
| `make_file` | Select a file on target 101 or 104, fill it with direct writes, and recheck size and placement. |
| `drop` | Clear the client's caches, and the server's too except in the server-RAM preparation step. |
| `residency` | Use Linux `mincore` to check how much of the file is in the client's page cache without reading it. |
| `counters` | Read client-received network bytes, server-sent network bytes and backing-device read bytes. |
| `cases` | Return the eight-case pilot or five rotated repetitions of the six media/state combinations. |
| `measure` | Prepare one cache state, run IOR, compare IOR and host evidence, and save the achieved label. |
| `cleanup` | Remove only files and directories bearing this run's owner marker. |
| `run`, `remote`, `ctl` | Execute commands locally, over SSH or through `beegfs-ctl`; capture measured-command output. |
| `record` | Atomically publish a JSON record such as the per-case result or run summary. |

## Running it and reading the results

Run from the project checkout on **`anjuna2` as the normal `pfs` user**. The
script invokes non-interactive sudo for its private mount and client cache
drop; the server cache drop uses SSH and sudo on `colva1`. A run drops host-wide
page caches, so its measurements require exclusive use of those hosts.

Pilot (eight measured reads):

```bash
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-pilot-01 --pilot
```

Full protocol (30 measured reads, new run ID):

```bash
python3 -B scripts/microbenchmarks/cache/run_cache.py \
  --run-id cache-full-01
```

Each run ID is used once. The terminal prints a bandwidth and `achieved` label
after every completed IOR read. Output is saved under
`results/microbenchmarks/runs/<run-id>/`:

```text
owner.json                  run/user identity
results.json                completed cases and their verified/unverified labels
01-hdd-backend/             one example measured case
  command.json              exact IOR argument list
  ior.json                  native IOR summary
  stdout.txt, stderr.txt    native IOR output
  counters.json             before/after network and target-device counters
  result.json               throughput, residency, ratios and achieved label
  warmup/                   present for states with an unmeasured warm-up
```

`results.json` and completed case directories remain if a run stops early;
missing cases do not appear as completed. On normal exit the script removes
its run-owned BeeGFS files and private mount, while preserving these raw
results. There is no resume command or separate plotting script in this folder.

**Execution status:** the program has not been run or calibrated against the
installed BeeGFS and IOR versions on the cluster. There are no verified cache
results or observed full-run timings in this repository. A run stops without
reporting a verified cache path when its layout, mount, IOR output or traffic
evidence does not match the protocol.
