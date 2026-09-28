# Cache: HDD and SSD reads

**Goal:** Read an 8-GiB file on one HDD and another 8-GiB file on one SSD,
with both targets on the **same storage server**. For each file, compare where
the bytes came from. This is a cache experiment, not a storage-pool sweep.

```text
anjuna2 reader
    ├─ client RAM hit ──────────────────────────────> application
    └─ client miss -> network -> server RAM hit ───> application
                              └─ server miss -> HDD or SSD -> application
```

| State | Before the measured read | Evidence of success |
|---|---|---|
| Backend | Drop client and server caches | Network and device reads |
| Server RAM | Drop caches, warm the file, drop client cache only | Network but few device reads |
| Client RAM | Drop caches, warm the file, keep client cache | Very little bulk network or device reads |

The measured command reads the existing file once with one IOR MPI rank and
normal buffered POSIX reads. No direct I/O, NetBench or setup I/O is included in
the reported throughput. Each case starts with its own cache preparation. The
full experiment is **2 media × 3 states × 5 repetitions = 30 reads**; the pilot
is eight reads (all six cases and one extra client hit per medium).

## What works now

- `cache_config.json` defines the fixed workload and evidence thresholds.
- `run_cache.py` **only writes a plan** to a pre-existing, marker-owned project
  run directory (`--plan-out .../plan.json`, optionally `--pilot`). Its helper
  functions build an IOR command and validate **saved** inventory and evidence.
  Running this script does **not** run a cache test.
- `probe_native_mount.py` tests **only** whether `anjuna2` can create a second,
  `native` BeeGFS mount. It uses a private Linux mount namespace and does not
  change the existing `buffered` mount, the system config, storage pools, or
  caches. After the current checkout is present on `anjuna2`, run there:

  ```bash
  python3 -B scripts/microbenchmarks/cache/probe_native_mount.py \
    --run-id cache-native-probe-01
  ```

  It saves `probe.json` in the named project run directory. An interrupted
  namespace disappears when its process exits. Use a new run ID for each retry.
- `visualize_results.py` reads completed raw IOR output and telemetry directly
  and plots throughput and path evidence by medium and verified state. For an
  existing complete run:

  ```bash
  python3 scripts/microbenchmarks/cache/visualize_results.py \
    results/microbenchmarks/runs/<cache-run-id>
  ```

The checked-in `cache_inventory.json` is **unreviewed**. Live target/layout
verification, dataset creation, cache controls and evidence capture, IOR launch,
checkpoint/resume, and watchdog-backed cache-mode/pool restoration are **not
implemented**. Do not run a full cache experiment from this checkout yet. Read
[`DESIGN.md`](DESIGN.md) and
[`IMPLEMENTATION_RULES.md`](../IMPLEMENTATION_RULES.md) before implementing the
cluster runner. The plots are published under `plots/generations/<id>/` and the
active generation is selected by `plots/plot_manifest.json`.

## How long?

The full run reads **240 GiB measured** plus **160 GiB of unmeasured warm-ups**;
80 GiB of measured client hits should be local, so roughly **320 GiB** of the
reads should cross the client network. At an ideal 2.5-Gbit/s link, that alone
requires **at least ~18 minutes**. Add two 8-GiB file preparations, HDD/SSD
backend reads, 30 cache preparations, telemetry, validation, restoration and
cleanup. This is a lower bound, **not a wall-time estimate**; obtain a full-run
estimate from the eight-case pilot, including all overhead. The pilot itself
must fit in 20 minutes or the pilot protocol must be revised before execution.
