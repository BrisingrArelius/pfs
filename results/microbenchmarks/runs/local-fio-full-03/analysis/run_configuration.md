# FIO run configuration

> Derived from host manifests; `manifest.json` and native FIO output remain authoritative.

## Hosts

| Host | Run ID | Mode | Protocol | FIO |
|---|---|---|---:|---|
| colva1 | `79a3fd0b23684e1495a4786e040acaba` | full | 5 | `fio-3.36` |
| colva2 | `e4085b7e352d422fbc578954eed94c9d` | full | 5 | `fio-3.28` |
| colva3 | `62a417b0c65345bfa5e7689ac54b78fa` | full | 5 | `fio-3.28` |
| colva4 | `dfbd5a8cad2a4dac9c368e87f09c47dc` | full | 5 | `fio-3.36` |

## Measurement protocol

- Jobs per OST: **4**
- Per-job size/region: **10 GiB**
- Prepared file per OST: **40 GiB**
- Engine/direct I/O: `libaio`, `direct=1`
- Queue depth: **32 per job**
- Timing: **5 s ramp + 60 s runtime**, `time_based=1`
- Repetitions: **5**

## Workloads

| Name | Pattern | Block size |
|---|---|---:|
| `seq_read` | `read` | `1m` |
| `seq_write` | `write` | `1m` |
| `rand_read_4k` | `randread` | `4k` |
| `rand_write_4k` | `randwrite` | `4k` |
| `rand_read_128k` | `randread` | `128k` |

For multi-job protocol 5 runs, bandwidth and IOPS are summed across jobs. 
Mean completion latency is operation-weighted; percentile columns use the worst per-job percentile.
