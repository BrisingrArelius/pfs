# FIO Hardware Benchmarks

Raw storage hardware benchmarks for BeeGFS HDD and SSD storage pools using [fio](https://fio.readthedocs.io/).  
These tests bypass the page cache (`O_DIRECT`) and Darshan instrumentation — pure disk performance only.

---

## Modes of Operation

The benchmark has two operating modes:

### `1. Client Mode (--beegfs)` 
Run this from a **Client Node**. It writes to the standard BeeGFS mountpoints. It writes output to the `results/` directory.

```bash
./run_fio.sh --beegfs

# Optional Flags:
# --hdd-dir /custom/hdd/path
# --ssd-dir /custom/ssd/path
```

### `2. Storage Mode (--ost)`
Run this while logged directly into a **Storage Node (OST)**. It writes explicitly to the backend local drives, bypassing BeeGFS network and metadata entirely to gather raw baseline comparisons. It creates and saves files to the `ost_results/` directory.

```bash
./run_fio.sh --ost

# Optional Flags:
# --hdd-ost-dir /local/hdd/mount  (Will automatically append 1, 2, 3, 4)
# --ssd-ost-dir /local/nvme/mount (Will automatically append 1, 2, 3) 
```

## Parsing the Results
After running benchmarks, you can use the `parse_results.py` script to automatically parse, format, and generate side-by-side Speedup / Overhead comparison tables between arrays:

```bash
# Summarize the client node (BeeGFS) results
python3 parse_results.py --beegfs

# Summarize the storage node (OST) results (will auto-compare to BeeGFS if available)
python3 parse_results.py --ost
```

This will automatically create a new `summary_comparison_YYYYMMDD_HHMMSS.txt` file alongside your raw logs in the targeted output directory.

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
