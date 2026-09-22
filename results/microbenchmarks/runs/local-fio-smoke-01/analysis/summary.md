# Local-storage FIO results

Validation: **PASS**
Parsed measurements: **10**

| Host | Target | Media | Workload | Runs | Median MiB/s | Range MiB/s | Median IOPS | Mean clat ms | p99 clat ms | Stops |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| colva1 | 101 | hdd | rand_read_128k | 1 | 40.2 | 40.2–40.2 | 322 | 99.386 | 549.454 | 0 byte / 1 time |
| colva1 | 101 | hdd | rand_read_4k | 1 | 1.7 | 1.7–1.7 | 437 | 73.082 | 392.167 | 0 byte / 1 time |
| colva1 | 101 | hdd | rand_write_4k | 1 | 1.8 | 1.8–1.8 | 453 | 70.536 | 96.993 | 0 byte / 1 time |
| colva1 | 101 | hdd | seq_read | 1 | 164.0 | 164.0–164.0 | 164 | 194.899 | 354.419 | 0 byte / 1 time |
| colva1 | 101 | hdd | seq_write | 1 | 166.2 | 166.2–166.2 | 166 | 192.381 | 270.533 | 0 byte / 1 time |
| colva1 | 104 | nvme | rand_read_128k | 1 | 5352.8 | 5352.8–5352.8 | 42823 | 0.737 | 3.293 | 1 byte / 0 time |
| colva1 | 104 | nvme | rand_read_4k | 1 | 1625.1 | 1625.1–1625.1 | 416036 | 0.075 | 0.111 | 1 byte / 0 time |
| colva1 | 104 | nvme | rand_write_4k | 1 | 1189.2 | 1189.2–1189.2 | 304429 | 0.102 | 0.132 | 1 byte / 0 time |
| colva1 | 104 | nvme | seq_read | 1 | 2212.6 | 2212.6–2212.6 | 2213 | 14.262 | 14.483 | 1 byte / 0 time |
| colva1 | 104 | nvme | seq_write | 1 | 4686.5 | 4686.5–4686.5 | 4686 | 6.736 | 10.813 | 1 byte / 0 time |
