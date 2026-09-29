# Microbenchmark implementation rules

Use this file when designing or implementing any microbenchmark domain in this
repository. It records the owner's requirements; it does not replace the domain
README or the research specifications.

## Required references

- Read [`Global.md`](../../docs/specs/Global.md) and
  [`MicroBenchmarks.md`](../../docs/specs/MicroBenchmarks.md) while implementing
  any domain, plus its current README or design and the applicable topology in
  [`CLUSTER_TOPOLOGY.md`](../../docs/CLUSTER_TOPOLOGY.md).
- Trace each executable choice (matrix, command, dataset, validation, cleanup)
  back to the applicable spec and current domain README. Record protocol deviations and
  pilot-derived changes explicitly, with a new fingerprint.

## Scope and evidence

- Prefer narrowly scoped, inflexible scripts over a universal benchmark runner.
  Keep each experimental factor fixed unless the domain design explicitly varies
  it. Small, justified duplication is preferable to a general-purpose control
  plane; share only genuinely identical low-risk primitives.
- A **completed** domain must expose a benchmark script producing **raw** run
  artifacts and a visualizer that reads those artifacts directly to make plots.
  Keep domain code
  in its own folder. Do not add a standalone raw-to-CSV parser or require an
  intermediate normalized-data command before plotting.
- An initial pilot should fit within **20 minutes**, including preparation,
  measurements, validation, restoration, and cleanup. Establish the estimate
  from actual pilot timing; refuse to claim or schedule a full run without it.
  If the design's proposed pilot cannot fit, record a bounded pre-pilot and revise
  the full pilot protocol before collecting production data. Do not silently
  lower the scientific workload or reuse pre-pilot data as full-run data.
- At the end of design work, ask one independent subagent to critique the design;
  resolve material flaws and repeat the independent critique until the design is
  clean. Check for unsafe mutations, missing evidence, ambiguous validity,
  incomplete interruption handling, and unnecessary generality.

## Write boundary and cleanup

- All local temporary files and result artifacts belong under the project
  directory, inside an explicitly selected run directory. No implicit writes to
  `$HOME`, `/tmp`, other project trees, or an unscoped shared location.
- The **only** permitted exceptions are the reviewed, marker-owned BeeGFS run
  namespace needed for the measurement and explicitly approved cluster-state
  control endpoints (for example, NetBench mode, cache control, pools, chooser).
  Enumerate and validate each exception before use; no other paths are writable.
  The existing FIO and iperf3 raw-generation runners have documented historical
  host-side output paths; their raw generation is unchanged at the owner's
  request. This does not grant new domains an external temporary-file location.
- Tests must verify path containment and symlink/traversal rejection, and that
  cleanup removes only run-owned created files, including after failure and
  interruption. Preserve captured evidence in the selected run directory until
  analysis is finished, then make artifact disposal explicit. Never delete an
  unrelated file to satisfy cleanup.
- Privileged state changes require exclusive authorization, independent
  restoration on coordinator loss, and verified restoration before a subsequent
  domain runs. A `finally` clause alone is insufficient.

## Progress and timing

- Every runner needs `--deadline`, `--time-limit`, and `--extend-deadline` with
  durable, atomic deadline updates. Reserve enough time for cleanup and cluster
  restoration. Expiry stops admission, interrupts a running attempt safely, and
  leaves a resumable record.
- Keep a locked, fingerprinted, checkpointed manifest with canonical plan and
  stable unit IDs. Attempts have unique IDs; validate saved evidence before
  skipping completed units on resume. Retrying an interrupted unit repeats its
  entire setup and measurement, but never reruns a valid measurement merely
  because later cleanup or analysis failed.
- Checkpoint cleanup, restoration, and plotting separately. The visualizer is
  read-only with respect to measurement and cluster state and rejects missing,
  stale, or corrupt native evidence.

## Current scope

Implement `cache/`, `communication/`, and `metadata/` against their
current protocols. Do not change FIO or iperf3 **raw-result generation**; their
visualizers may be updated to consume their existing raw output directly. No
cluster pilot or privileged state transition may be claimed complete without
live, reviewed evidence.
At present the other four domains have only offline plan/command code and
raw-to-plot visualizers. Their live benchmark executors and pilots remain open.
