# Network Transport Benchmark

This directory implements the TCP transport domain specified in
[`DESIGN.md`](DESIGN.md). It measures memory-to-memory network throughput with
iperf3; it does not access BeeGFS files or storage targets.

## Files

| File | Purpose |
|---|---|
| `capture_beegfs_paths.py` | One-time, read-only capture used to establish the path source of truth |
| `network_inventory.json` | Reviewed immutable hosts, interfaces, addresses and paths |
| `network_config.json` | Protocol, matrix, timing, ports, validation and operational limits |
| `run_iperf3.py` | Single-coordinator pilot/full runner with durable resume |
| `parse_results.py` | Read-only native-JSON validation and CSV/Markdown reporting |

## Safety gate

The checked-in inventory is based on the completed six-host capture at
`$HOME/pfs-results/network/path-sot-01`. The runner still validates every live
address, interface, route, MTU, link speed and clock before starting traffic.

The capture completed on all six hosts at `2026-09-22T18:26:52Z`. It confirms
synchronized clocks, MTU 1500, 2.5-Gbit/s physical links, and active TCP paths for
all eight client-to-OSS pairs. Both addresses on each multihomed client use the
same physical interface (`eno1` on `anjuna2`, `enp4s0` on `anjuna3`). Every OSS
uses `enp7s0`.

Treat the reviewed inventory as immutable for a run. The runner fingerprints and
embeds it in every manifest; resume rejects a changed inventory. If topology or
link configuration changes, create a new capture and inventory rather than
editing an active run.

## Execution model

Invoke `run_iperf3.py` exactly once on `anjuna3`. It runs locally for the
`anjuna3` endpoint and uses non-interactive SSH for `anjuna2` and the Colva
servers. Do not invoke a separate runner on each client.

For each restartable unit the coordinator:

1. Captures interface counters on participating hosts.
2. Starts attempt-owned one-shot servers on the configured addresses and ports.
3. Verifies listeners with `ss` without making a connection that would consume
   `--one-off`.
4. Starts clients behind one future timestamp barrier.
5. Validates actual recorded process-start skew.
6. Retrieves native client/server JSON and process evidence.
7. Validates every member before completing the unit.
8. Terminates only recorded process groups and removes only the exact run-specific
   remote temporary directories.

Remote temporary artifacts use `/tmp/pfs-network-bench/<run-id>/...`. If a
measurement succeeds but cleanup fails, resume retries cleanup without repeating
network traffic. If any member of a concurrent epoch fails, the complete epoch is
retried under a new attempt ID.

## Prerequisites

On `anjuna3`:

- Python 3.9 or newer.
- `iperf3`, `ip`, `ss`, `timeout`, `ssh`, and `scp`.
- Non-interactive SSH with existing verified host keys to `anjuna2` and
  `colva1`-`colva4`.

On every endpoint:

- iperf3 3.x with server, one-off, JSON, bind, port, parallel, omit, time and
  reverse options.
- `ip`, `ss`, `timeout`, `python3`, and a synchronized clock.
- Configured ports 5201 and 5202 available on every OSS.

The runner never uses `sudo`, changes BeeGFS configuration, selects fallback
addresses, or kills an unrecorded process.

## Pilot

The pilot contains four isolated `anjuna2`/`colva2` measurements: one and four
streams in both directions.

```bash
cd "$HOME/pfs"
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-pilot-01" \
  --pilot --time-limit 30m
```

Resume an interrupted or planned-stop pilot:

```bash
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-pilot-01" \
  --resume --time-limit 30m
```

Parse on `anjuna3` or after retrieval:

```bash
python3 scripts/microbenchmarks/network/parse_results.py \
  "$HOME/pfs-results/network/network-iperf3-pilot-01" \
  --output-dir "$HOME/pfs-results/network/network-iperf3-pilot-01/analysis"
```

The pilot must report four units, four path sessions, no validation errors, and
complete remote cleanup before scheduling the full run.

## Full run

```bash
cd "$HOME/pfs"
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-full-01" \
  --time-limit 3h
```

Full coverage is:

```text
160 isolated units and path sessions
20 one-client/four-OSS epochs containing 80 path sessions
10 two-client/four-OSS epochs containing 80 path sessions
190 restartable units, 320 path sessions total
```

If the runner reports `budget_stop`, resume the same directory. Do not create a
new run or add `--pilot`:

```bash
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-full-01" \
  --resume --time-limit 3h
```

An active deadline can be extended without changing scientific settings:

```bash
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-full-01" \
  --extend-deadline 1h
```

## Analysis

```bash
python3 scripts/microbenchmarks/network/parse_results.py \
  "$HOME/pfs-results/network/network-iperf3-full-01" \
  --output-dir "$HOME/pfs-results/network/network-iperf3-full-01/analysis"
```

Analysis produces:

- `measurements.csv`: one row per path session.
- `epochs.csv`: aggregate receiver throughput per restartable unit.
- `interface_counters.csv`: per-host/interface counter deltas per unit.
- `summary.csv` and `summary.md`: repeated per-path results.
- `run_configuration.md`: readable fixed paths and tool versions.
- `parse_report.json`: completeness and validation status.

Receiver `end.sum_received.bits_per_second` is the principal delivered-bandwidth
metric. Sender throughput, bytes, retransmissions, endpoint CPU utilization and
native JSON are retained. Isolated and simultaneous modes remain separate.

## Retrieval

Create a checksummed archive on `anjuna3` after completion:

```bash
RUN="network-iperf3-full-01"
ROOT="$HOME/pfs-results/network"
tar -C "$ROOT" -czf "$ROOT/$RUN.tar.gz" "$RUN"
(cd "$ROOT" && sha256sum "$RUN.tar.gz" > "$RUN.tar.gz.sha256")
```

From the PC, use the required jump host explicitly:

```bash
RUN="network-iperf3-full-01"
mkdir -p "$HOME/network-result-downloads/$RUN"
scp -J dashlab@lab.dashlab.in \
  "pfs@anjuna3.dashlab.in:pfs-results/network/$RUN.tar.gz" \
  "pfs@anjuna3.dashlab.in:pfs-results/network/$RUN.tar.gz.sha256" \
  "$HOME/network-result-downloads/$RUN/"
```

Verify before extraction:

```bash
(
  cd "$HOME/network-result-downloads/network-iperf3-full-01"
  sha256sum -c -- network-iperf3-full-01.tar.gz.sha256
)
tar -xzf "$HOME/network-result-downloads/network-iperf3-full-01/network-iperf3-full-01.tar.gz" \
  -C results/microbenchmarks/runs/
```

## Tests

The development PC does not need iperf3 for fixture-based tests:

```bash
python3 -B -m unittest discover \
  -s scripts/microbenchmarks/network/tests -v
```
