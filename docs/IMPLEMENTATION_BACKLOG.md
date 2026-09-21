# Experiment implementation backlog

These are the experiment features that are not implemented yet. Completed
repository-path work is recorded separately in
[COMPLETED_PATH_REPAIRS.md](COMPLETED_PATH_REPAIRS.md).

## Configuration and inventory

- [ ] Check the real cluster setup: hosts, devices, targets, mounts, pool IDs,
  network topology, and active TCP/RDMA transport. Do not trust historical IDs.
- [ ] Build the full D/S/H, chooser, stripe, concurrency, and repetition matrix.
  Add capacity only after its meaning and thresholds are agreed.
- [ ] Define request sizes, dataset sizes, concurrency, run time, setup, and success
  checks for each domain. Keep both 4-KiB and 128-KiB local random-read tests.
- [ ] Record skipped or impossible matrix cases and explain why they cannot run.

## Six benchmark domains

- [ ] Complete local FIO coverage, including useful simultaneous-target tests.
  Use StorageBench only as an optional cross-check.
- [ ] Add TCP/RDMA tests for both directions and simultaneous network traffic.
- [ ] Add IOR + NetBench tests for BeeGFS communication. Check server traffic and
  always restore NetBench mode after success, failure, or interruption.
- [ ] Add a multi-process normal-IOR runner for the full placement matrix. Verify
  fresh file layouts and include write synchronization in measured time.
- [ ] Add controlled read-cache tests for client miss/server miss, client
  miss/server hit, and client hit. Verify states with network and device evidence.
- [ ] Add mdtest coverage for selected operations, directory layouts, and process
  counts. Distinguish metadata-only work from operations that also write data.

## Durable progress for every experiment

- [ ] Give every configuration, repetition, phase, and retry a stable ID.
- [ ] Save progress safely with `pending`, `running`, `completed`, `failed`, and
  `interrupted` states. Record the cluster allocation and execution order.
- [ ] Skip only runs whose required output was successfully saved and validated.
  Retry interrupted measurements after restoring their setup.
- [ ] Track measurement, parsing, and analysis separately. A parser failure must
  not repeat a successful benchmark.
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
- [ ] Store commands, settings, raw logs, monitoring data, and derived results in
  each run directory. Never write into historical result directories.

## Application validation and scheduling

- [ ] Add DLIO input, checkpoint-write, and recovery tests with realistic caching
  and separate epoch results. Workload checkpoints and runner progress are different.
- [ ] Use pilots to estimate setup, run, synchronization, and cleanup time for the
  full matrix. Update estimates as measurements finish.
- [ ] Split the matrix across allocations without dropping cases. Restart dependent
  epoch sequences as one unit after interruption.
- [ ] Test manifests and resume behavior with fake interruptions before cluster
  runs. Use instrumented pilots to verify placement, cache, and path assumptions.
