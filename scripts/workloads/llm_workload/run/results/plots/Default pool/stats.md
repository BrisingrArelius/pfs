# I/O access-pattern report — eval_llama31_8b_adam_per-rank_pool1_TP2_load_183817

## Request size

* 5,558 read operations, 3.24 MB mean
* p50 524.29 KB, p95 28.84 MB, p99 28.84 MB, max 262.14 MB
* operations at or below 524.29 KB: **83.7% of operations** but **13.5% of bytes** — the count/byte split that decides whether seek cost or bandwidth dominates

## Spatial pattern

* contiguous: 906 (16.3%)
* strided: 1 (0.0%)
* forward-jump: 1,496 (27.0%)
* backward-seek: 3,139 (56.6%)
* 3,139 backward transitions covering 59.57 GB of revisited range

## Sharing

* workload label **SSF** over 2 ranks
* files by class: {'full': 7, 'private': 2}

## Re-reads

* 15.10 GB read exactly once; 966.79 MB in blocks read more than once (max 8x) — an *upper bound* on cache need, since it assumes every such block must be resident at once
* 9,262 of 39,912 block accesses were repeats; reuse distance p50 1 blocks, p90 114 blocks
* an LRU cache of **0.00 GB** turns 50% of the duplicate reads into hits
* an LRU cache of **0.06 GB** turns 90% of the duplicate reads into hits
* an LRU cache of **3.48 GB** turns 99% of the duplicate reads into hits

## Latency

* p50 4,800 us, p95 168,200 us, p99 199,329 us
* 0.0% of reads under 10 us (cache looks cold)
* effective bandwidth on >=1 MB operations: 162 MB/s

## Striping

* stripe slots used: 4, max/min imbalance 1.24x
* **slots, not disks** — pass --target-map

## Caveats

* cross-rank clock alignment: yes (start_abs)
* latency is blocking-syscall duration; it does not decompose into client cache / network / server queue / media
* reads served from Python's BufferedReader never enter the kernel and are invisible here, correctly so
