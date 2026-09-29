# BeeGFS cache microbenchmark

This benchmark measures how BeeGFS caching affects **reads and writes** through
the existing mount on `anjuna2`. It has two narrowly scoped runners:
`run_cache_read.py` measures reads from prepared 8-GiB files on HDD target
101 and NVMe target 104 on `colva1`; `run_cache_write.py` measures write
completion under the existing fsync policy and, in native mode, client-side
write acceptance. Neither runner changes the mount or its configuration.
Both import `cache_common.py` for the same preflight, target selection,
cache drops, lock, run ownership and cleanup. `visualize_results.py` handles
completed **read** runs only; it never touches the cluster.
Buffered and native read pilots/full runs are complete; the write runner is
implemented but has not yet had a cluster pilot.

## Read benchmark

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
   bytes, and target-device read bytes. The runner does not map or probe the
   file between warm-up and IOR.
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
| `client_ram` (`native` only) | Drop both, read the file once to warm it, drop **server only**, then measure | Little bulk network or backing-device activity during the 8-GiB IOR read, consistent with a client-cache hit |

Traffic ratios use **IOR's 8-GiB logical read** as the denominator. A verified
backend case requires at least 80% of that amount in each of client-received,
server-sent and device-read bytes. A verified server-RAM case requires at least
80% on both network counters and at most 20% device reads. A verified
client-RAM case (`native` only) requires at most 20% on each network and device
counter. The script also checks a one-second pre-read window for background
traffic. These are *observed traffic classifications*, not labels inferred
from cache-drop order alone. The counters cover whole interfaces and a whole
device, so unrelated traffic can make a real hit `unverified`; they do not
measure the exact number of resident client pages. New results identify this
rule as `traffic_counters_v1` and contain no `client_residency` field.

### Functions and call chain

| Function | Role |
|---|---|
| `run_cache_read.main` | Parse read arguments and call `benchmark`. |
| `run_cache_read.benchmark` | Prepare two files, measure read cases, and save results. |
| `cache_common.new_run/prepared_run` | Create the run, lock out both cache runners, preflight, verify ownership, prepare the BeeGFS namespace, and clean up. |
| `cache_common.check_cluster/cache_config_matches` | Check the existing mount's mode/native threshold, target health/mapping, NetBench, sudo and RAM. |
| `cache_common.target_file/targets` | Select and verify one storage target for a new file. |
| `make_file` | Select a file on target 101 or 104, fill it with direct writes, and recheck size and placement. |
| `cache_common.drop` | Clear both hosts' caches at the start of a case, then optionally clear only one host after warm-up. |
| `counters` | Read client-received network bytes, server-sent network bytes and backing-device read bytes. |
| `cases` | Return buffered (4/20) or native (8/30) pilot/full cases; full runs rotate their configuration order. |
| `measure` | Prepare one cache state, run IOR, compare IOR and host evidence, and save the achieved label. |
| `cache_common.cleanup` | Remove only files and directories bearing this run's owner marker. |
| `cache_common.run/remote/ctl/record` | Run commands and save native output or JSON. |

## Running the read benchmark and interpreting results

Run from the project checkout on **`anjuna2` as the normal `pfs` user**. Select
the mode already active on the existing mount. The script does not restart or
restore the BeeGFS client. It invokes non-interactive sudo to set the stripe
pattern on its own directories, read BeeGFS target information through
`beegfs-ctl`, and drop the client cache. The CLI is given
`--cfgFile=/etc/beegfs/beegfs-client.conf`, matching the configuration that
works on `anjuna2`; the script does not change its authentication settings.
The server cache drop uses SSH and sudo on `colva1`. A run drops host-wide page
caches, so its measurements require exclusive use of those hosts. Its file lock
only prevents these two cache runners from running at once; it does not
block other cluster workloads.

For another buffered pilot (four measured reads), use an unused run ID with
an active `buffered` mount:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache_read.py \
  --run-id cache-buffered-pilot-05 --mode buffered --pilot
```

For another buffered full run (20 measured reads), use a new run ID:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache_read.py \
  --run-id cache-buffered-full-02 --mode buffered
```

With the **existing** client separately configured and verified as `native`,
the same script accepts `--mode native`. Its pilot has eight measured reads
and its full run has 30, each using a new run ID. Before a native run, check
the live mount configuration on `anjuna2`:

```bash
sudo grep -E 'tuneFileCacheType|tuneFileCacheBufSize' /proc/fs/beegfs/*/config
```

It must show `native` and a `tuneFileCacheBufSize` of at least `2097152`
bytes. The installed client configuration describes this value as a threshold
for direct I/O in native mode. Its original `524288`-byte value is below the
benchmark's 1-MiB transfer; 2 MiB permits the page-cache path. The current
runner checks the effective value and stops if it is too small. To repeat the
pilot, use a fresh ID, for example:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache_read.py \
  --run-id cache-native-pilot-07 --mode native --pilot
```

Inspect all eight pilot cases and their raw traffic counters before deciding
whether a native full run is valid. Only after a pilot verifies both media and
all intended states, use a fresh full-run ID, for example:

```bash
python3 -B scripts/microbenchmarks/cache/run_cache_read.py \
  --run-id cache-native-full-02 --mode native
```

Each run ID is used once. The terminal prints a bandwidth and `achieved` label
after every completed IOR read. Output is saved under
`results/microbenchmarks/runs/<run-id>/`:

```text
owner.json                  run/user identity, selected mode and pilot flag
live_config.json            effective mode, threshold and fsync policy at preflight
results.json                completed cases and their verified/unverified labels
01-hdd-backend/             one example measured case
  command.json              exact IOR argument list
  ior.json                  native IOR summary
  stdout.txt, stderr.txt    native IOR output
  counters.json             pre-read idle, before and after traffic counters (new runs)
  result.json               mode, throughput, traffic ratios and achieved label
  warmup/                   present for states with an unmeasured warm-up
```

`results.json` and completed case directories remain if a run stops early;
missing cases do not appear as completed. On normal exit the script removes
its run-owned BeeGFS files while preserving these raw results. There is no
resume command.

## Visualizing a completed run

`visualize_results.py` needs only the copied run directory and matplotlib; it
runs on any machine and starts no MPI or BeeGFS command:

```bash
RUN_ID=cache-native-full-01
python3 scripts/microbenchmarks/cache/visualize_results.py \
  "results/microbenchmarks/runs/$RUN_ID"
```

It revalidates every case before plotting: the native IOR summary must be one
POSIX read of the whole 8-GiB file by one rank with 1-MiB transfers, its rate
must match the recorded result, the recorded command must be the documented
one without `O_DIRECT`, and the traffic ratios in `result.json` must be
recomputable from the raw `counters.json`. For new runs it also recomputes the
one-second quiet check from the saved idle and before samples. It then
re-derives each case's `achieved` label from the same thresholds the runner
applied and refuses to plot if the recorded label disagrees with its own
evidence. It also accepts transitional native traffic-only results without the
newer marker or idle sample, such as pilot `-06`, while preserving that evidence
limit. For older mapping-probe investigation runs, it preserves their original
labels and plots their legacy probe values as diagnostics; it does not
reinterpret them as valid residency.

| Output | Contents |
|---|---|
| `plots/throughput.png` | One dot per measured read, grouped by medium and cache state, with each group's min-to-max span and median bar |
| `plots/traffic_evidence.png` | One panel per configuration: client-received, server-sent and device-read ratios for each case against the 0.8 and 0.2 thresholds |
| `plots/client_residency.png` | Failed legacy probe values, only when those values exist in an older run |
| `plots/plot_manifest.json` | The complete figure set this visualizer owns for the run |

The program also prints one line per group with its count, minimum, median and
maximum rate, plus the run's verified and `unverified` totals.

**Execution status:** buffered pilot `cache-buffered-pilot-04` achieved 4/4 and
buffered full `cache-buffered-full-01` achieved 20/20. Native pilot
`cache-native-pilot-06` achieved 8/8 and native full `cache-native-full-01`
achieved 30/30. The visualizer rechecked every completed case's IOR summary,
command, recorded label and raw traffic counters. The full run has five reads
per medium/state; mean IOR throughput was:

| Medium and state | Buffered full (MiB/s) | Native full (MiB/s) |
|---|---:|---:|
| HDD backend | 182.9 | 163.4 |
| HDD server RAM | 239.0 | 235.9 |
| HDD client RAM | — | 9586.3 |
| SSD backend | 237.8 | 244.4 |
| SSD server RAM | 240.5 | 243.9 |
| SSD client RAM | — | 9550.0 |

The ten native full-run client-RAM reads ranged from 9515.3 to 9626.5 MiB/s.
Their client-received traffic was at most 0.0000312 of the 8-GiB logical read,
server-sent traffic at most 0.0000017, and target-device reads zero. The
near-zero bulk traffic directly supports client-cache hits. Both native runs
used the transitional traffic-only result format, without the later explicit
`verification` marker or saved idle-window sample. Their recorded
`quiet_before` flags cannot be independently recomputed. Neither run saved a
live configuration snapshot. Shortly after the full run and before restoration,
the live mount showed `native` and `2097152`; this observation is not a
during-run snapshot. The mount was subsequently restored and verified as
`buffered` with `524288`. Native pilots `cache-native-pilot-01` through `-03`
remain investigation artifacts with unverified client-RAM cases.

The preserved [buffered full raw run](../../../results/microbenchmarks/runs/cache-buffered-full-01/)
and [native full raw run](../../../results/microbenchmarks/runs/cache-native-full-01/)
include their throughput and traffic-evidence plots.

Layout, mount, or IOR protocol failures stop a run; a traffic mismatch leaves
the affected case labeled `unverified`.

### Native investigation record

Native caching itself worked in manual tests with the 2-MiB threshold: after a
separate `dd` process warmed the 8-GiB file and exited, IOR read it at
9053 MiB/s; other warm IOR reads reached 11425–11863 MiB/s. Closing the
warm-up file therefore did not necessarily invalidate the cache. The original
runner's mapping-based `residency()` probe coincided with a destructive loss of
cached pages: a writable mapping was followed by system `Cached` falling from
9,007,872 to 624,860 kB, and the later **read-only** `libc.mmap` plus `mincore`
version returned 0.0 while `Cached` fell from 8,978,276 to 596,132 kB. The
precise BeeGFS/kernel mechanism is unknown. The current runner removes that
probe and uses the independent traffic counters above. Rates from the earlier
native pilots' `client_ram` cases are preserved as investigation data, not
validated client-cache throughput. Their roughly 240-MiB/s results do not
establish a buffer-capacity or network ceiling; the same-path iperf3 result
reached 280.5 MiB/s and native pilot rates varied.

The `anjuna2` checkout is separate from this coding checkout. Pilot `-06` and
full `-01` used the transitional traffic-only format; transfer the current
runner to `anjuna2` before any future run so its threshold guard and raw idle
sample are active.

### Copying raw results through `anjuna3`

Run these commands yourself after a run. On **`anjuna3`**, copy one complete
run directory from `anjuna2` into the existing results directory, retaining
the same run ID:

```bash
RUN_ID=cache-native-full-01
ls -ld "$HOME/pfs/results/microbenchmarks/runs"
rsync -a --ignore-existing "pfs@anjuna2:/home/pfs/tejas/pfs/results/microbenchmarks/runs/$RUN_ID" "$HOME/pfs/results/microbenchmarks/runs/"
rsync -ai --checksum --dry-run "pfs@anjuna2:/home/pfs/tejas/pfs/results/microbenchmarks/runs/$RUN_ID" "$HOME/pfs/results/microbenchmarks/runs/"
```

The final dry run should print no file changes. On **your laptop**, copy that
run directory from `anjuna3` into the coding checkout's existing results
directory:

```bash
RUN_ID=cache-native-full-01
ls -ld /home/arelius/projects/pfs/results/microbenchmarks/runs
rsync -a --ignore-existing -e 'ssh -J dashlab@lab.dashlab.in' "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" /home/arelius/projects/pfs/results/microbenchmarks/runs/
rsync -ai --checksum --dry-run -e 'ssh -J dashlab@lab.dashlab.in' "pfs@anjuna3.dashlab.in:/home/pfs/pfs/results/microbenchmarks/runs/$RUN_ID" /home/arelius/projects/pfs/results/microbenchmarks/runs/
```

The second dry run should also print no file changes. `--ignore-existing`
protects existing files; a nonempty checksum dry run means a file is missing or
differs and should be inspected before relying on the copy. Change `RUN_ID` to
copy another full or pilot run, including the early investigation artifacts.
These commands copy raw files; run `visualize_results.py` on the laptop
afterward.

The read runner does not load Darshan. Its results come from native IOR JSON
and client/server Linux counters; Darshan cannot identify which cache level
served the read.

## Write benchmark (implemented, not yet run)

`run_cache_write.py` uses the same anjuna2/colva1 preflight, cache drops,
one-target placement, exclusive lock and run-owned cleanup as the read runner.
It does **not** edit or remount BeeGFS. Check the live client configuration
first. Procfs may print `1`/`0`; pass the equivalent `true`/`false`
to the runner:

```bash
sudo grep -E 'tuneFileCacheType|tuneFileCacheBufSize|tuneRemoteFSync' /proc/fs/beegfs/*/config
python3 -B scripts/microbenchmarks/cache/run_cache_write.py \
  --run-id cache-write-buffered-true-pilot-02 \
  --mode buffered --remote-fsync true --pilot
```

The example applies only when the live values are `buffered` and `1` (true).
Use a fresh run ID for each invocation, including after a failed preflight.
The script refuses a mismatched live
mode, native threshold or fsync policy. Run only after arranging exclusive
host use: each case drops caches on anjuna2 and colva1. The four live
configurations and matrix sizes are:

| Client mode | `tuneRemoteFSync` | Measured states per medium | Pilot | Full |
|---|---|---|---:|---:|
| buffered | false | server-cache acknowledgement | 2 | 10 |
| buffered | true | server-disk fsync completion | 2 | 10 |
| native | false | server-cache acknowledgement | 2 | 10 |
| native | true | server-disk fsync completion; client write acceptance | 4 | 20 |

Across the four configurations this is a 10-case pilot and 50-case full
matrix, not a single run of either size. The native client case is run only
with remote fsync enabled so its post-timing fsync has the disk-completion
policy. Do not compare results across configurations without recording how
the mount was changed externally; this runner never changes it.

For the server cases, the runner creates a new empty, verified one-target
file, drops caches, then invokes one IOR rank with:

```text
/usr/bin/mpirun -np 1 /usr/local/bin/ior -a POSIX \
  -w -E -k -g -e -t 1m -b 8g -s 1 -i 1 \
  -o <new-empty-file> -O summaryFormat=JSON \
  -O summaryFile=<case-directory>/ior.json
```

The reported IOR MiB/s is the whole write workload, including the requested
final fsync; it is **not** an isolated RAM, network or device bandwidth
([IOR timing and `-e`](https://ior.readthedocs.io/en/latest/userDoc/tutorial.html)).
`tuneRemoteFSync=false` makes fsync acknowledge server-side cache, while
`true` requires server-side disk completion
([BeeGFS client tuning](https://doc.beegfs.io/7.4.4/advanced_topics/client_tuning.html)).
The `server_ram` result label
therefore denotes the *acknowledgement policy*, not proof that no data hit
disk during the run. In particular, false does not prevent concurrent
writeback. The `server_disk` label means that fsync returned under the disk
policy; target-device sectors written corroborate traffic but cannot time the
exact commit of this particular file.

The native-only `client_ram` case instead writes 1 GiB with 1-MiB POSIX
`write()` calls and times only those calls while the descriptor stays open.
It records client-transmitted bytes across that interval; then it calls
`fsync()` and closes **outside** the timer. It is application write-acceptance
throughput, not pure memory bandwidth. The case is marked `unverified` if
more than 20% of the logical data crossed the client interface during the
timed region, or if post-fsync client/server traffic and target-device
write counters do not support transfer. Host-wide counters can include
unrelated traffic, so a mismatch is diagnostic rather than proof of a
specific BeeGFS internal behavior.

Both runners save `live_config.json` at preflight; it is a start-of-run
snapshot, not proof that a mount could not change later. The call chain is
`main → cache_common.new_run → benchmark →
cache_common.prepared_run → measure → ior_write/client_write`. Each case
saves its exact command or POSIX workload description, native IOR output
where applicable, idle and measurement counters, and a `result.json`
with intended/achieved state. A failed run preserves completed raw results.
`visualize_results.py` does not accept write runs; inspect their
`results.json` and raw case directories until a write-specific validator
is implemented. No cluster write pilot or full run has yet been executed
with this runner.
