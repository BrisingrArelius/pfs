# FIO run configuration

> Derived from host manifests; `manifest.json` and native FIO output remain authoritative.

## Hosts

| Host | Run ID | Mode | Protocol | FIO |
|---|---|---|---:|---|
| colva1 | `2e90dcabc5f04ad598eec9a51499f661` | full | 5 | `fio-3.36` |
| colva2 | `279c400c3cb54df3b70568b49747a3c0` | full | 4 | `fio-3.28` |
| colva3 | `cdd562023e52431ab5ddfd1a5366f72b` | full | 4 | `fio-3.28` |
| colva4 | `579d447a01424e71a2b290f90d08a571` | full | 4 | `fio-3.36` |

## Compatibility

**INCOMPATIBLE:** host manifests use different scientific configurations. Do not interpret them as one experiment.

| Host | Jobs | Size/job GiB | Ramp s | Runtime s | Time based |
|---|---:|---:|---:|---:|---:|
| colva1 | 4 | 10 | 5 | 60 | 1 |
| colva2 | 1 | 10 | 0 | 60 | 0 |
| colva3 | 1 | 10 | 0 | 60 | 0 |
| colva4 | 1 | 10 | 0 | 60 | 0 |
