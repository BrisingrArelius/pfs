# Experiment implementation backlog

These are future requirements from the documented experiment methodology.
Mechanical relocation repairs are tracked separately in
[TODOS_SCRIPT_CHANGES.md](../TODOS_SCRIPT_CHANGES.md).

## Configuration and inventory

- [ ] Verify live host/device/target mapping, mounts, pool IDs, topology and active
  transport. Reconcile historical hard-coded inventories before implementing
  placement administration; historical labels are not authoritative live IDs.
- [ ] Represent the full D/S/H × chooser × stripe × concurrency × repetition
  matrix and the capacity factor once its meaning/bands are confirmed. Validate
  eligibility and headroom, and record infeasible cells with reasons.
- [ ] Define per-domain request sizes, dataset geometry, concurrency, durations,
  preparation and completion criteria. Preserve the 128-KiB random-read point
  alongside 4-KiB local FIO tests. Record exact commands and tool versions.

## Six benchmark domains

- [ ] Extend local FIO coverage to identified gaps and relevant concurrent target
  combinations. Reuse compatible evidence; StorageBench is an optional cross-check.
- [ ] Add active-transport TCP/RDMA tests for relevant paths, directions and
  simultaneous traffic; capture native metrics and transport evidence.
- [ ] Add IOR + NetBench communication runs with explicit client/server participation,
  request sizes and stripe fan-out. Validate synthetic read setup and server
  traffic; record/restore mode on all clients, including after interruptions.
- [ ] Add a dedicated multi-process normal-IOR placement runner for the complete
  matrix, verified fresh-file layouts, synchronization-inclusive write timing,
  declared cache preparation and client/server/backend telemetry.
- [ ] Add controlled normal-read cache experiments with feasible miss/miss,
  miss/hit and client-hit states. Verify achieved state with network/backend
  evidence; cache mode and residency are separate fields.
- [ ] Add mdtest operations, directory layouts and concurrency with explicit
  namespace-only versus data-bearing scope and native phase metrics.

## Durable progress for every experiment

- [ ] Create a persistent suite manifest and scientific-configuration fingerprint;
  use stable configuration × repetition × phase IDs and separate retry attempts.
- [ ] Atomically persist pending/running/completed/failed/interrupted status,
  required evidence and allocation/session/block/order information in durable storage.
- [ ] Validate completion before skipping a case; preserve interrupted artifacts
  and rerun interrupted measurement units after restoring preparation.
- [ ] Separate measurement, parsing and analysis status so parser failures can be
  retried without repeating successful expensive I/O.
- [ ] Revalidate topology, transport, membership, chooser, stripes, capacity and
  NetBench state on resume. Recreate cache/filler/data prerequisites; do not treat
  warm RAM as durable progress or resume dependent epochs mid-sequence.
- [ ] Replace the legacy workload runner's limited profile/storage/repetition keys
  and non-atomic progress writes with the full contract; prevent incompatible
  configuration reuse and test recovery at interruption boundaries.

## Instrumentation and artifacts

- [ ] Validate Darshan integration, enabled modules, MPI-rank and worker/loader
  coverage and overhead for normal IOR and supported applications. Document
  optional synthetic NetBench coverage and supplementary mdtest coverage.
- [ ] Associate logs with explicit run/attempt/process IDs instead of selecting
  the newest shared-directory log. Preserve raw logs after parsing; replace the
  legacy runner's attempted raw-log deletion with durable artifact retention.
- [ ] Keep preparation and measured phases attributable; use native phase output
  and aligned telemetry where Darshan summaries cannot supply epoch/state metrics.
- [ ] Store manifests, commands, raw outputs/logs, telemetry and derived artifacts
  under run-specific results directories; protect historical evidence from reuse
  as mutable run output. Define required evidence validation per domain.

## Application validation and scheduling

- [ ] Implement separately specified DLIO training-input, synchronized checkpoint-
  write and recovery phases with realistic prefetch/caching and epoch attribution.
  Experiment-progress checkpoints must remain distinct from workload checkpoints.
- [ ] Pilot setup, measurement, synchronization and cleanup costs per domain;
  estimate full-matrix duration and uncertainty, updating from completed cases.
- [ ] Schedule recorded/counterbalanced blocks across allocations while preserving
  the matrix. Bound restartable units; dependent epoch sequences restart as a
  unit. Obtain adequate allocation time or justify segmentation in advance.
- [ ] Verify manifests/resume/evidence handling with fixtures and simulated
  interruptions before cluster pilots; validate scientific path/cache/placement
  assumptions with instrumented pilots on the confirmed topology.
