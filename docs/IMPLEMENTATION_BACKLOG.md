# Experiment implementation backlog

This checklist distinguishes completed benchmark coverage from outstanding
experiment work. Completed repository-path repairs are recorded separately in
[COMPLETED_PATH_REPAIRS.md](COMPLETED_PATH_REPAIRS.md).

## Configuration and inventory

- [ ] Complete the remaining [cluster topology](CLUSTER_TOPOLOGY.md) details:
  exact drive models for `colva2`-`colva4` and upstream PCIe/switch topology.
  Refresh the dated inventory before benchmark execution.
- [ ] Execute the planned D/S/H, chooser, stripe, concurrency, and repetition
  matrix. The placement code generates a 6,480-unit baseline-capacity plan but
  does not execute it. Add capacity only after site bands are agreed.
- [ ] Confirm the designed request sizes, datasets, concurrency, timing, setup,
  and success checks against installed tools in bounded pilots. Keep both 4-KiB
  and 128-KiB local random-read tests.
- [ ] Record skipped or impossible matrix cases and explain why they cannot run.

## Six benchmark domains

- [x] Complete the per-target [FIO protocol-5 run](../scripts/microbenchmarks/fio/README.md):
  700 native measurements across four hosts. Simultaneous-target experiments
  are outside this runner's scope. FIO 3.28 and 3.36 figures are kept separate.
- [ ] Demonstrate and time forced FIO interruption/recovery on the live cluster;
  fake-process tests cover resume but do not establish this live behavior.
- [x] Capture both directions and simultaneous TCP paths in the iperf3 full run:
  190 restartable units and 320 path sessions, with raw-to-plot visualization.
- [ ] Add RDMA transport measurements only if a reviewed active BeeGFS path uses RDMA.
- [ ] Add IOR + NetBench tests for BeeGFS communication. Check server traffic and
  always restore NetBench mode after success, failure, or interruption.
- [ ] Add a multi-process normal-IOR runner for the full placement matrix. Verify
  fresh file layouts and include write synchronization in measured time.
- [ ] Add controlled read-cache tests for client miss/server miss, client
  miss/server hit, and client hit. Verify states with network and device evidence.
- [ ] Add mdtest coverage for selected operations, directory layouts, and process
  counts. Distinguish metadata-only work from operations that also write data.

## Durable progress for every experiment

- [ ] Add an allocation-aware wall-clock budget to every top-level runner. Accept
  a relative duration such as `--time-limit 5h` and an absolute deadline for the
  remaining reservation time.
- [ ] Reserve configurable shutdown time for result flushing, cleanup, state
  restoration, and scheduler exit. Do not start a measurement that cannot finish
  inside the remaining budget plus that buffer.
- [ ] Store the active deadline in the run manifest and reread it between atomic
  measurements. Support an atomic deadline update while the runner is active so
  an extended reservation can be used without restarting completed work.
- [ ] On budget expiry, finish or safely interrupt the current atomic unit, record
  its status, restore temporary cluster state, and exit successfully as a resumable
  stop rather than reporting an experiment failure.
- [ ] Give every configuration, repetition, phase, and retry a stable ID.
- [ ] Save progress safely with `pending`, `running`, `completed`, `failed`, and
  `interrupted` states. Record the cluster allocation and execution order.
- [ ] Skip only runs whose required output was successfully saved and validated.
  Retry interrupted measurements after restoring their setup.
- [ ] Track measurement, owned cleanup, and plotting separately across all
  domains. A visualization failure must not repeat a successful benchmark.
- [ ] On resume, recheck topology, pool membership, chooser, stripes, capacity,
  transport, and NetBench mode. Recreate required files and cache preparation.
- [ ] Never treat warm RAM as saved progress or resume a dependent epoch sequence
  in the middle after releasing the cluster nodes.

## Instrumentation and artifacts

- [ ] Check that Darshan captures every intended MPI rank and application worker.
  Measure its overhead before enabling it for the full matrix.
- [ ] Match every raw log to a run, retry, and process ID. Do not select logs only
  because they are newest, and do not delete raw logs after parsing.
- [ ] Keep setup I/O separate from measured I/O. Use native benchmark output and
  system monitoring where Darshan cannot identify an epoch or cache state.
- [ ] Store commands, settings, raw logs, monitoring data, and directly generated
  figures in each run directory. Never write into historical result directories.

## Workload profiles and classification

- [ ] Replace the two-profile current set with an evidence-backed profile set.
  Define size, direction, access-pattern, frequency, sharing, and phase dimensions
  only where the available counters support them.
- [ ] Preserve Darshan file/rank records when sharing-pattern classification is
  required. The current parser aggregates rank away.
- [ ] Decide whether HDF5/PnetCDF dimensional counters are in scope and add their
  parser support before using them to classify multidimensional striding.
- [ ] Connect supported strided/nd-strided profiles to an executable runner. Wire
  `block_size` and `nd_dims` through configuration, and resolve classifier order
  and dimensionality before adopting profile thresholds.
- [ ] Fix the standalone C workload compile path on current glibc, including the
  `_GNU_SOURCE` requirement for `O_DIRECT`.
- [ ] Validate any HDD/SSD placement classifier against measurements. No placement
  rules or validated thresholds currently exist.
- [ ] Repeat contiguity and operation-rate characterization on a broader corpus;
  current findings cover a limited nine-day Polaris sample.
- [ ] Replace stale `/mnt/beegfs/advay` defaults with verified, authorized cluster
  paths before workload or BeeGFS FIO execution.

## Application validation and scheduling

- [ ] Add DLIO input, checkpoint-write, and recovery tests with realistic caching
  and separate epoch results. Workload checkpoints and runner progress are different.
- [ ] Use pilots to estimate setup, run, synchronization, and cleanup time for the
  full matrix. Update estimates as measurements finish.
- [ ] Split the matrix across allocations without dropping cases. Restart dependent
  epoch sequences as one unit after interruption.
- [ ] Test manifests and resume behavior with fake interruptions before cluster
  runs. Use instrumented pilots to verify placement, cache, and path assumptions.
