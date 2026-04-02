# FIO Hardware Benchmarks

Raw storage hardware benchmarks for BeeGFS HDD and SSD storage pools using [fio](https://fio.readthedocs.io/).  
These tests bypass the page cache (`O_DIRECT`) and Darshan instrumentation — pure disk performance only.

---

## Files

| File | Description |
|------|-------------|
| `hdd.fio` | fio job file for HDD targets — 4 jobs, iodepth 32 |
| `ssd.fio` | fio job file for SSD targets — 5 jobs, iodepth 64 |
| `run_fio.sh` | Orchestration script — drops caches, runs both, saves results |
| `results/` | Auto-created output directory (JSON + text per run) |

---

## Quick Start

```bash
# Run both HDD and SSD benchmarks (requires sudo for cache drop)
sudo ./run_fio.sh \
    --hdd-dir /mnt/beegfs/hdd \
    --ssd-dir /mnt/beegfs/ssd

# HDD only
sudo ./run_fio.sh --hdd-only --hdd-dir /mnt/beegfs/hdd

# SSD only
sudo ./run_fio.sh --ssd-only --ssd-dir /mnt/beegfs/ssd

# Custom results directory
sudo ./run_fio.sh --results-dir /tmp/fio_results \
    --hdd-dir /mnt/beegfs/hdd \
    --ssd-dir /mnt/beegfs/ssd
```

Or run fio directly without the wrapper:

```bash
fio hdd.fio --directory=/mnt/beegfs/hdd/fio_scratch \
    --output-format=json+ --output=results_hdd.json
```

---

## Benchmark Jobs

### HDD (`hdd.fio`) — 4 jobs, iodepth 32, 4 workers each

| Job | Pattern | Block Size | Measures |
|-----|---------|-----------|---------|
| `seq_write` | Sequential write | 1 MiB | Streaming write bandwidth |
| `seq_read` | Sequential read | 1 MiB | Streaming read bandwidth |
| `rand_write_4k` | Random write | 4 KiB | Write IOPS / seek latency |
| `rand_read_4k` | Random read | 4 KiB | Read IOPS / seek latency |

### SSD (`ssd.fio`) — 5 jobs, iodepth 64, 4 workers each

All HDD jobs plus:

| Job | Pattern | Block Size | Measures |
|-----|---------|-----------|---------|
| `rand_mixed_4k` | 70/30 R/W random | 4 KiB | Sustained mixed-IO IOPS |

---

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `direct=1` | O_DIRECT | Bypass page cache — true hardware measurement |
| `runtime=60s` | 60 s per job | Stable average; increase to 120+ for production runs |
| `ramp_time=5s` | 5 s | Discard transient warm-up measurements |
| `filesize=8g` | 8 GiB | Exceeds typical DRAM on a node; forces real disk I/O |
| `numjobs=4` | 4 | Matches typical core count for I/O threads |
| `stonewall` | sequential jobs | One job runs at a time; no cross-job interference |

> **Disk space:** 8 GiB × 4 jobs = **32 GiB** per pool.  
> Adjust `filesize=` in the job files if your PFS has less free space.  
> Scratch files are deleted automatically after each run.

---

## Output

Each run produces two files in `results/`:

```
results/
├── hdd_20260402_153000.json   ← full fio JSON (all metrics)
├── hdd_20260402_153000.txt    ← human-readable output + aggregate summary
├── ssd_20260402_153000.json
└── ssd_20260402_153000.txt
```

The JSON is suitable for further analysis (pandas, plotting, etc.) and follows the same output convention as the rest of the `pfs/` project.

---

## Notes

- **Storage pool setup** must be done first: `../pooling_scripts/configure_pools.sh`
- The BeeGFS pools assign specific disk targets (HDD: `x01–x04`, SSD: `x05–x07` per node) to separate mount points — make sure you pass the correct `--directory` that maps to each pool.
- `sudo` is needed only for `echo 3 > /proc/sys/vm/drop_caches`. Use `--no-drop-cache` to skip it (results may reflect cached data from prior runs).
