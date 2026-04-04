# FIO Benchmark Matrix Suite

A comprehensive framework for benchmarking multiple workloads across different BeeGFS storage pools and raw physical targets (OSTs). 

This suite replaces the old static `.fio` jobs with an automated matrix runner (`matrix_benchmark.py`). It dynamically constructs configurations based on `fio_config.json`, running multiple repetitions (for measuring variance), across multiple dataset sizes, generating an aggregated JSON dump, and tracking the explicit BeeGFS chunk targets (OST hits).

---

## 1. Configure the Matrix

The file `fio_config.json` controls exactly what workloads run during the benchmark. Instead of writing custom `.fio` files, simply toggle options here:

```json
{
    "runs_per_test": 5,
    "file_sizes": ["1g", "10g"],
    "modes": {
        "seq_read": true,
        "seq_write": true,
        "rand_read": true,
        "rand_write": true,
        "seq_rw": true,
        "rand_rw": true
    },
    "block_size_seq": "1m",
    "block_size_rand": "4k",
    "io_depth": 32,
    "num_jobs": 4
}
```

## 2. Execute the Suite

Use `matrix_benchmark.py` to trigger the tests. It supports benchmarking the BeeGFS client mounts (`--beegfs`) or the raw local block devices (`--ost`).

### BeeGFS Mode (Client)
Benchmarking standard mount points. It also attempts to log which OST chunks were hit using `beegfs-ctl`.

```bash
# Run against ALL default pools (HDD, SSD)
python3 matrix_benchmark.py --beegfs

# Run only against the HDD pool
python3 matrix_benchmark.py --beegfs --pool hdd

# Run against a specific custom named pool
python3 matrix_benchmark.py --beegfs --pool nvme-fast --custom-dir /mnt/beegfs/advay/nvme-fast
```

### Storage Mode (OST Backend)
Run this when logged directly into a Storage Node. It explicitly bypasses BeeGFS network overhead to ascertain backend baseline speed.

```bash
python3 matrix_benchmark.py --ost --pool hdd
```

## 3. Analyze the Results

Because `matrix_benchmark.py` generates large `.json` collections detailing all repetitions of every workload combination, we use **`analyze_matrix.py`** to distill this into clean averages.

By default, running the analyzer without arguments automatically groups and analyzes the most recent benchmark run:

```bash
python3 analyze_matrix.py
```
*(Or point to a specific file: `python3 analyze_matrix.py results/matrix_results_20260404_153000.json`)*
