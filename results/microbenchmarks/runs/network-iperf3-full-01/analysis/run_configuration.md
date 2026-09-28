# iperf3 run configuration

> Derived from `manifest.json`; the manifest and native JSON are authoritative.

## Protocol

- Protocol version: `1`
- Transport: `tcp`
- Timing: 5 s omitted + 30 s measured
- Isolated streams: `[1, 4]`
- Concurrent streams: `4`
- Repetitions: **5**
- Inventory fingerprint: `d2f48206e2eab3b899af5cad91705c7964b54eff0dd5294579d151e0b03c392a`

## Fixed paths

| Client | OSS | Source endpoint | Destination endpoint | Evidence |
|---|---|---|---|---|
| anjuna2 | colva1 | `192.168.0.6` (`eno1`) | `192.168.0.2` (`enp7s0`) | active_confirmed |
| anjuna2 | colva2 | `10.1.19.73` (`eno1`) | `10.1.19.76` (`enp7s0`) | active_confirmed |
| anjuna2 | colva3 | `10.1.19.73` (`eno1`) | `10.1.19.77` (`enp7s0`) | active_confirmed |
| anjuna2 | colva4 | `192.168.0.6` (`eno1`) | `192.168.0.5` (`enp7s0`) | active_confirmed |
| anjuna3 | colva1 | `192.168.0.7` (`enp4s0`) | `192.168.0.2` (`enp7s0`) | active_confirmed |
| anjuna3 | colva2 | `10.1.19.74` (`enp4s0`) | `10.1.19.76` (`enp7s0`) | active_confirmed |
| anjuna3 | colva3 | `10.1.19.74` (`enp4s0`) | `10.1.19.77` (`enp7s0`) | active_confirmed |
| anjuna3 | colva4 | `192.168.0.7` (`enp4s0`) | `192.168.0.5` (`enp7s0`) | active_confirmed |

## Tool versions

| Host | iperf3 |
|---|---|
| anjuna2 | `iperf 3.9 (cJSON 1.7.13)` |
| anjuna3 | `iperf 3.9 (cJSON 1.7.13)` |
| colva1 | `iperf 3.16 (cJSON 1.7.15)` |
| colva2 | `iperf 3.9 (cJSON 1.7.13)` |
| colva3 | `iperf 3.9 (cJSON 1.7.13)` |
| colva4 | `iperf 3.16 (cJSON 1.7.15)` |
