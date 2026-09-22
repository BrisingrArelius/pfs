# FIO run configuration

> Derived from host manifests; `manifest.json` and native FIO output remain authoritative.

## Hosts

| Host | Run ID | Mode | Protocol | FIO |
|---|---|---|---:|---|
| colva1 | `a711ca58c8ef48d0b95e7903975f6576` | full | 4 | `fio-3.36` |
| colva2 | `7103b0b2110d40aba7b77428e4acfee1` | full | 4 | `fio-3.36` |
| colva3 | `80f163c26e8b4f5eb3c12eea782069ab` | full | 4 | `fio-3.36` |
| colva4 | `53de53e105eb4d65b19bfaed802af211` | full | 4 | `fio-3.36` |

## Measurement protocol

- Jobs per OST: **1**
- Per-job size/region: **10 GiB**
- Prepared file per OST: **10 GiB**
- Engine/direct I/O: `libaio`, `direct=1`
- Queue depth: **32 per job**
- Timing: **0 s ramp + 60 s runtime**, `time_based=0`
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
