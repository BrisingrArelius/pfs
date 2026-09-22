# Network Transport Benchmark Design

## Purpose

Measure the sustainable transport bandwidth of the network paths used between
the BeeGFS clients and storage servers. This domain isolates the transport,
NICs, and network from BeeGFS request processing and storage-device I/O.

The benchmark answers:

> What can each relevant TCP path sustain in both directions, and what aggregate
> throughput remains when clients and storage servers communicate concurrently?

This design implements the requirements applicable to network transport from:

- [`Global.md`](../../../docs/specs/Global.md)
- [`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md), especially
  section 4
- [`CLUSTER_TOPOLOGY.md`](../../../docs/CLUSTER_TOPOLOGY.md)

`DL_Experiment.md` does not define this benchmark domain.

## Scope

### Included

- The two registered BeeGFS clients: `anjuna2` and `anjuna3`.
- The four BeeGFS storage servers: `colva1` through `colva4`.
- TCP traffic in both directions.
- One and four parallel TCP streams.
- Isolated client-to-storage-server paths.
- Single-client and dual-client simultaneous traffic.
- Five independent repetitions.
- Native iperf3 JSON and endpoint/network provenance.

### Excluded

- BeeGFS request processing, files, target filesystems, and storage devices.
- Storage pools, target selection, stripe settings, media type, and target
  fullness.
- UDP characterization.
- Simultaneous bidirectional traffic in the initial matrix.
- `anjuna2` to `anjuna3` as a main matrix path. It is not a normal BeeGFS bulk
  data path, but may be used as a separately labelled diagnostic.
- RDMA performance while no usable RDMA link exists.

## TCP and RDMA

iperf3 measures TCP or UDP socket transport. It cannot measure RDMA. An RDMA
case requires an RDMA-specific tool such as `ib_write_bw` and active compatible
RDMA links at both endpoints.

The recorded BeeGFS configuration permits RDMA and TCP fallback, but the current
inventory reports no RDMA links and shows TCP as the selected BeeGFS transport.
The one-time source-of-truth capture confirms this before the benchmark inventory
is finalized. If no RDMA path exists, that inventory records RDMA as infeasible
with the observed evidence. It must not represent an iperf3 result as an RDMA
result.

Enabling `connUseRDMA` only allows BeeGFS to attempt RDMA. Actual use additionally
requires compatible hardware, active links, client-module support, server support,
and a successful connection. With TCP fallback enabled, configuration intent is
not proof of the active transport.

## BeeGFS Communication Strategy and Path Evidence

### Architectural paths

For ordinary non-mirrored BeeGFS operation, the documented communication strategy
is:

```text
Mount and control discovery
  client -> management service on anjuna3

Namespace and file-layout operations
  client -> metadata service on anjuna3 -> metadata target

Bulk file reads and writes after layout lookup
  client -> selected storage server(s) -> target filesystem(s)
```

The management service supplies registry and control information. It is not an
inline data proxy. The metadata service supplies namespace and layout information.
Normal bulk file data then moves directly between the client and the storage
servers holding the file's stripes. A striped request can therefore create
simultaneous client-to-OSS traffic without sending its bulk payload through
`anjuna3`'s management or metadata services, except when `anjuna3` itself is the
client.

TCP or RDMA is the transport for these BeeGFS service connections. Normal RDMA
would optimize the network leg; it would not remove the BeeGFS storage service,
target filesystem, or backend storage from a normal data operation.

### Confirmed dated observations

The inventory collected on 2026-09-21 confirms the following for that observation
window:

- `anjuna3` was a BeeGFS client and hosted management and metadata services.
- `anjuna2` was a registered BeeGFS client.
- `colva1` through `colva4` were storage servers.
- All four storage links reported 2,500 Mbit/s, full duplex, on `enp7s0`.
- No storage server reported an RDMA link.
- `anjuna3` reported no RDMA link and its observed BeeGFS connections used TCP.
- Storage configurations set `connUseRDMA=true`; the captured client configuration
  also set `connTCPFallbackEnabled=true`.
- `colva1` through `colva3` restricted BeeGFS advertisement to `enp7s0` through an
  interface file. `colva4` had no effective interface-file restriction in the
  captured state.

The selected routes observed from `anjuna3` were:

| BeeGFS service | Observed transport and destination |
|---|---|
| Management on `anjuna3:8008` | TCP to `192.168.0.7:8008` |
| Metadata on `anjuna3:8005` | TCP to `192.168.0.7:8005` |
| Storage on `colva1:8003` | TCP to `192.168.0.2:8003` via `enp7s0` |
| Storage on `colva2:8003` | TCP to `10.1.19.76:8003` via `enp7s0` |
| Storage on `colva3:8003` | TCP to `10.1.19.77:8003` via `enp7s0` |
| Storage on `colva4:8003` | TCP to `192.168.0.5:8003` via `enp7s0` |

These are explicit historical observations, not assumptions about the state at
benchmark execution time.

### Not yet confirmed

The preserved evidence does not establish:

- The current route and active transport from either client after configuration
  or network changes.
- `anjuna2`'s per-OSS BeeGFS-selected source address, destination address, and
  interface.
- Whether the `192.168.0.0/24` and `10.1.19.0/24` addresses traverse identical
  physical switch paths despite appearing on the same host interfaces.
- Upstream switch capacity, oversubscription, and the physical client uplinks.
- Per-connection TCP versus RDMA state during a future BeeGFS workload.
- The effect of mirroring, if enabled later; mirrored data would add paths and
  traffic not represented by this non-mirrored model.

### Required one-time source-of-truth capture

Before finalizing the iperf3 configuration, run the read-only path capture once
and preserve raw output that establishes:

1. Registered client, metadata, management, and storage nodes with advertised
   interfaces and transport capabilities.
2. Active BeeGFS connection routes and transport from both `anjuna2` and
   `anjuna3` to every storage server.
3. Address/interface routing with `ip route get` from each client to each selected
   storage address and in the return direction.
4. Link state, MTU, speed, duplex, and RDMA-link state on every endpoint.
5. BeeGFS client and server network configuration relevant to interface selection,
   RDMA enablement, and TCP fallback.

Configuration, advertisement, kernel route selection, and active BeeGFS
connection state are separate evidence. The run report must state which level
supports each conclusion. If active connection information cannot be obtained,
label the route as configured or inferred rather than confirmed.

Use the captured evidence to create a reviewed `network_inventory.json` containing
the fixed hosts, roles, transport, and intended paths. Address, interface, MTU and
link speed are recorded separately for both endpoints of every path; they are not
host-wide properties because a client can be multihomed. The benchmark runner
consumes this file; it does not choose different addresses for later repetitions.
Every case in a run uses the same inventory fingerprint.

## Execution Topology

### Coordinator

Invoke the benchmark runner exactly once on `anjuna3`:

```bash
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-full-01" \
  --time-limit 3h
```

`anjuna3` is the coordinator because it has the repository, current topology
evidence, and access to the cluster nodes. The Python process itself generates
negligible traffic compared with iperf3. Running it on a benchmark endpoint is
acceptable, but its CPU use and host role must be recorded.

The user does not invoke the Python runner separately on each client. The
coordinator performs the following operations:

1. Validates local and non-interactive SSH access to all required hosts.
2. Runs commands locally when `anjuna3` is the selected client.
3. Uses SSH to run client commands on `anjuna2`.
4. Uses SSH to start one-shot iperf3 servers on the selected Colva hosts.
5. Waits for server readiness before releasing clients for a measurement.
6. Starts concurrent clients behind a barrier for simultaneous cases.
7. Collects client and server JSON, stderr, exit status, and telemetry.
8. Terminates only server processes belonging to the current attempt.
9. Validates artifacts before marking the measurement complete.

The runner must use non-interactive SSH with host-key checking left enabled. It
must fail preflight rather than prompt for a password or trust decision during a
measurement.

### Why one coordinator

A runner launched independently on each client cannot safely coordinate the
dual-client simultaneous cases or maintain one authoritative execution order.
One coordinator provides:

- A single durable manifest and configuration fingerprint.
- Stable case and attempt identifiers.
- A common start barrier for concurrent traffic.
- Unambiguous ownership of server ports and processes.
- Consistent retry and cleanup behavior.
- One place from which to resume after an interruption.

### Server lifecycle and ports

Each measurement uses one-shot iperf3 servers rather than persistent unmanaged
daemons. Isolated and single-client fan-out cases need one server process on each
participating OSS. Dual-client cases need two independent server processes per
OSS because iperf3 servers may serialize tests; use distinct configured ports,
initially `5201` and `5202`.

Before use, the runner verifies that each configured address still belongs to the
expected host and that each required port is available. It does not replace the
configured path with a newly discovered path.
Every remote process receives a run-specific identity. Cleanup must target only
the recorded process and port, never all iperf3 processes on a host.

Client and server native output must both be retained. Remote artifacts are
written to a run-specific temporary directory and transferred to the coordinator
after the measured interval so that result transfer is not mixed into measured
throughput. Successful transfer and validation precede remote cleanup.

## Fixed Inventory and Path Selection

Run `capture_beegfs_paths.py` once before the pilot to establish the source of
truth. Record at least:

- Hostnames and resolved addresses.
- The exact local and remote socket addresses.
- `ip route get` output for each planned connection.
- Source and destination interface name, state, MTU, speed, and duplex for every
  planned path.
- TCP congestion-control algorithm and relevant socket settings.
- iperf3 version on every endpoint.
- BeeGFS-advertised interfaces and active connection transport.
- `rdma link show` output.
- Interface byte, packet, drop, and error counters before and after each attempt.
- CPU identity and iperf3 local/remote CPU utilization.

Known addresses are starting evidence, not immutable configuration:

| Host | Recorded addresses |
|---|---|
| `anjuna2` | `10.1.19.73` historically; refresh required |
| `anjuna3` | `192.168.0.7`, `10.1.19.74` |
| `colva1` | `192.168.0.2` |
| `colva2` | `192.168.0.3`, `10.1.19.76` |
| `colva3` | `192.168.0.4`, `10.1.19.77` |
| `colva4` | `192.168.0.5` |

The reviewed inventory fixes the route selected for BeeGFS from each client. The
runner binds the configured source and destination addresses and rejects a
connection whose native JSON reports different endpoints. Alternate-subnet tests
are a separate experiment and are not silently added to the primary matrix.

## Isolated Matrix

| Factor | Values |
|---|---|
| Client | `anjuna2`, `anjuna3` |
| Storage server | `colva1`, `colva2`, `colva3`, `colva4` |
| Direction | client to OSS, OSS to client |
| Parallel streams | 1, 4 |
| Repetitions | 5 |
| Warm-up omitted by iperf3 | 5 seconds |
| Measured duration | 30 seconds |
| Protocol | TCP |

The isolated matrix contains:

```text
2 clients x 4 storage servers x 2 directions x 2 stream counts x 5 repetitions
= 160 measurements
```

Only one isolated measurement runs at a time. This prevents another planned test
from competing for the client NIC, OSS NIC, or shared network during a path
baseline.

## Simultaneous Matrix

Simultaneous cases use four streams per iperf3 session because the historical
evidence shows that four streams can saturate a 2.5-Gbit/s link. They are reported
separately from isolated cases.

### One client to four OSSs

For each client, run one iperf3 session to every OSS at the same time:

```text
2 clients x 2 directions x 5 repetitions = 20 concurrent epochs
```

Each epoch contains four path sessions. This exposes the selected client's NIC or
uplink limit and any shared network contention across the four destinations.

### Two clients to four OSSs

Run both clients against every OSS at the same time:

```text
2 directions x 5 repetitions = 10 concurrent epochs
```

Each epoch contains eight path sessions. Every OSS receives one session from each
client on separate ports. This exposes shared switch limits and contention at the
OSS links.

The coordinator starts all servers first, verifies readiness, prepares all client
commands, and releases clients at a common future timestamp using the confirmed
synchronized endpoint clocks. It records actual process-start timestamps and
reports start skew. Excessive skew invalidates the epoch.

## Measurement Order

Each case has a stable ID derived from mode, client, server, direction, stream
count, and repetition. The manifest records the actual order.

Use a recorded deterministic seed to counterbalance isolated case order across
repetitions. Do not run all repetitions of one direction first. Concurrent epochs
remain indivisible restartable units: if one member fails or the coordinator is
interrupted, preserve all partial artifacts and retry the entire epoch under a new
attempt ID.

## Native Commands

The runner constructs commands equivalent to the following. Exact addresses,
ports, paths, and options come from the validated run configuration.

Server:

```bash
iperf3 --server --one-off --json --bind <server-address> --port <port>
```

Client-to-OSS:

```bash
iperf3 --client <server-address> --bind <client-address> \
  --port <port> --parallel <streams> --omit 5 --time 30 --json
```

OSS-to-client uses iperf3 reverse mode:

```bash
iperf3 --client <server-address> --bind <client-address> \
  --port <port> --parallel <streams> --omit 5 --time 30 --reverse --json
```

The direction label refers to bulk data flow, not which endpoint initiated the
control connection.

## Metrics

Preserve native fields and derive at least:

- Receiver and sender throughput in bit/s, Gbit/s, and MiB/s.
- Bytes transferred.
- TCP retransmissions from the sender.
- Requested and actual stream count.
- Per-stream throughput and aggregate throughput.
- Local and remote CPU utilization reported by iperf3.
- Measured duration and interval stability.
- Interface byte, packet, drop, and error deltas.
- Link speed and delivered throughput as a percentage of nominal line rate.
- Start skew and per-session completion skew for concurrent epochs.
- Median, minimum, maximum, and variation across five repetitions.

Receiver throughput is the principal delivered-bandwidth metric. Sender and
receiver values are both retained. Do not subtract network throughput from FIO or
BeeGFS throughput to claim component overhead.

## Completion Validation

An isolated measurement is complete only when:

- Client and server processes exited successfully.
- Client and server JSON parse successfully and contain no native error.
- Reported endpoints, port, direction, and stream count match the case.
- Omitted warm-up and measured duration match the configured values within a
  declared tolerance.
- Receiver bytes and throughput are nonzero.
- Required telemetry and command records exist.
- Both endpoint artifacts have been copied to durable coordinator storage.
- The one-shot server exited and no owned process remains.

A simultaneous epoch is complete only when every member passes the same checks
and start skew is within the configured tolerance. Partial success does not become
a mixture of sessions from different attempts.

## Artifacts and Durable Progress

The cluster run writes outside the Git checkout, for example:

```text
$HOME/pfs-results/network/network-iperf3-full-01/
```

After retrieval, preserve it under:

```text
results/microbenchmarks/runs/network-iperf3-full-01/
```

The run directory contains:

```text
manifest.json
RUN.md
inventory/
raw/<case-id>/attempt-<n>/
analysis/
```

The manifest contains a configuration fingerprint, protocol version, inventory,
execution seed and order, sessions, cases, epochs, attempts, commands, timestamps,
exit states, validation states, and cleanup outcomes. Updates are atomic and
durable.

States include `pending`, `running`, `completed`, `failed`, and `interrupted`.
On resume, abandoned running attempts become interrupted. Only validated completed
measurements are skipped. The saved inventory fingerprint must still match the
run; resume never substitutes newly discovered paths. Parser and plotting status
are independent of measurement status, so analysis failures never repeat valid
network traffic.

## Time Budget and Interruption

The runner accepts a relative time limit and an optional absolute deadline. It
reserves shutdown time for artifact transfer, manifest flushing, and remote
process cleanup. It does not start an isolated measurement or concurrent epoch
that cannot finish before the usable deadline.

On interruption, the runner records the active attempt, stops only its owned
processes, retrieves available partial artifacts, marks the attempt interrupted,
and exits with completed evidence intact. Resume checks required endpoints, the
fixed inventory fingerprint, tool compatibility, and port availability before
continuing; it does not repeat the source-of-truth capture.

## Runtime Estimate

The isolated matrix requires at least:

```text
160 x (5 seconds warm-up + 30 seconds measured) = 1 hour 33 minutes 20 seconds
```

The simultaneous matrix adds about 18 minutes of traffic time. With SSH setup,
readiness checks, telemetry, pauses, durable writes, and cleanup, plan:

| Stage | Expected wall time |
|---|---:|
| Pilot | 5-10 minutes |
| Full isolated and simultaneous suite | 2-2.5 hours |
| Requested allocation/time limit | 3 hours |

Replace these estimates with observed pilot timing before the full run.

## Pilot

The pilot uses `anjuna2` and `colva2`, which have compatible historical evidence.
It runs one repetition for:

- One stream in both directions.
- Four streams in both directions.

It must verify SSH behavior, server readiness, port ownership, address binding,
route evidence, native JSON fields, CPU metrics, counters, timing, artifact
transfer, cleanup, manifest durability, interruption recovery, and resume.

Example future invocation:

```bash
python3 scripts/microbenchmarks/network/run_iperf3.py \
  --results-dir "$HOME/pfs-results/network/network-iperf3-pilot-01" \
  --pilot --time-limit 30m
```

The runner implements this interface and refuses to execute until the reviewed
network inventory is explicitly confirmed.

## Operational Safety

- Do not run concurrently with FIO, BeeGFS I/O experiments, result transfer, or
  production workloads.
- Expect SSH latency while links are saturated.
- `anjuna3` also hosts BeeGFS management and metadata services; use a quiet and
  authorized window.
- Do not use `sudo` for iperf3 or bind privileged ports.
- Do not change BeeGFS transport configuration for this baseline.
- Do not kill pre-existing iperf3 processes; fail preflight on a port conflict.
- Preserve historical network evidence without writing into its directory.

## Historical Evidence

The repository contains historical outputs, not an archived runner. Existing
30-second, four-stream TCP measurements reached approximately 2.34-2.35 Gbit/s,
or about 280 MiB/s, on recorded 2.5-Gbit/s links. The `anjuna2` to `colva2`
forward result recorded substantially more retransmissions than the reverse
result. These data motivate repeated direction-specific measurements but do not
replace current topology, provenance, and simultaneous-path coverage.
