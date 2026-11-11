# CosmoFlow vs 3D U-Net: I/O findings

## Experimental setup

DLIO 2.0.0 emulates I/O, not full model training: eight MPI ranks, five epochs,
zero simulated computation, and all-rank checkpoints after epochs 2 and 4.
Caches are dropped before each invocation, not between epochs.

Files use **four BeeGFS targets with 512 KiB chunks**. SSD read/write caps are
45 MiB/s (fast) and 15 MiB/s (slow). Both workloads' typical files span all
four targets; their application request size is not the stripe chunk size.

## Workload and trace characteristics

| | CosmoFlow | 3D U-Net |
|---|---|---|
| Scientific task | Predict cosmological parameters from 3D matter distributions | Segment structures in 3D images |
| DLIO input | 8,192 TFRecord files; one record/file | 168 NPZ files; one sample/file |
| Configured mean sample size | **2,828,486 B** (≈2.83 MB) | **146,600,628 B** (≈146.6 MB) |
| Reader | TensorFlow; batch 1/rank; four configured parallel file reads | PyTorch; batch 4/rank; four reader workers |
| Dominant recorded POSIX read request | **256 KiB**: 828,810 requests | **512 KiB**: 230,455 requests |
| Dataset opens / POSIX reads | **81,920 / 992,655** | **800 / 234,455** |
| Checkpoint payload per rank | **100,000,000 B** | **499,153,191 B** |

The request and open counts are from **one instrumented pool-14 invocation**;
they are aggregate API calls over five epochs, not per-epoch physical-device
traffic. Configured sample sizes are means; actual file sizes vary.

### Reader behavior from DLIO 2.0.0 source

- **CosmoFlow:** the [generator](https://github.com/argonne-lcf/dlio_benchmark/blob/v2.0.0/dlio_benchmark/data_generator/tf_generator.py)
  writes a serialized example per configured sample. The [reader](https://github.com/argonne-lcf/dlio_benchmark/blob/v2.0.0/dlio_benchmark/reader/tf_reader.py)
  shards filenames, uses `TFRecordDataset` with four parallel reads, parses
  records, and prefetches. With one record/file, consuming a sample requires
  the serialized record—not a chosen slice of a multi-record file. It returns
  a preconstructed resized image rather than using the parsed image for training.
- **U-Net:** the [NPZ reader](https://github.com/argonne-lcf/dlio_benchmark/blob/v2.0.0/dlio_benchmark/reader/npz_reader.py)
  obtains the `x` array with `np.load(filename, allow_pickle=True)['x']`, then
  selects a sample. The [PyTorch loader](https://github.com/argonne-lcf/dlio_benchmark/blob/v2.0.0/dlio_benchmark/data_loader/torch_data_loader.py)
  uses worker processes and prefetching; this is not a deliberate chunk-only
  reader. Source code alone does not prove every byte was fetched from physical
  storage on every access for either workload.
- **Original application ≠ our emulation:** the [CosmoFlow paper](https://arxiv.org/html/1808.04728v2)
  reports 512 MB TFRecord files containing 64 samples, not our one-record files.
  [3D U-Net](https://arxiv.org/abs/1606.06650) is a volumetric segmentation
  network; our generated NPZ sizes and reader settings describe the DLIO profile.

## Performance observation

Earlier **uninstrumented runs, three repetitions per condition**; throughput
is total processed samples divided by full DLIO run time, including checkpoints:

| Fast / slow targets | Pool | CosmoFlow samples/s | U-Net samples/s |
|---|---:|---:|---:|
| 4 / 0 | 14 | 45.21 | 1.639 |
| 3 / 1 | 15 | — | 1.078 |
| 2 / 2 | 16 | 37.10 | 1.074 |
| 1 / 3 | 17 | — | 1.073 |
| 0 / 4 | 18 | 32.24 | 1.074 |

**Observed:** U-Net drops ≈34% with the first slow target and then plateaus;
CosmoFlow declines progressively across its three measured mixes (≈29% from
4/0 to 0/4). Three CosmoFlow points do not prove a linear relationship.

U-Net's first epoch averaged ≈129 s at 4/0 versus ≈384–387 s with slow targets.
Single instrumented CosmoFlow runs show ≈262, 460 and 627 s for 4/0, 2/2 and
0/4 respectively. Most later epochs are similar; the instrumented CosmoFlow
0/4 epoch 5 (≈202 s) is an exception.

## Interpretation to test

**Hypothesis:** CosmoFlow's small files and concurrent readers may overlap
per-file setup and transfers, making overall progress respond gradually to the
fast/slow mix. U-Net's much larger files and batch requirements may expose a
slow required piece on the critical path, producing a step and plateau.

File size and four-target striping alone **do not prove** either mechanism:
both workloads use all targets. Test with per-batch/file completion and
time-aligned per-target traffic, or vary file size and concurrency independently.
The nominal 4/0, 2/2 and 0/4 cap sums (180, 120, 60 MiB/s) are not measured
application throughput.

Later-epoch similarity and whole-run device traffic below API read volume are
**consistent with cache reuse**, not proof of the cache layer or of every read
being cached. Compare the cache microbenchmark first, then local FIO for
request-size/device context, mdtest for open-intensive behavior, and iperf for
network capacity. None alone establishes the bottleneck.

## Reading the figures

**A:** native epochs, nested rank-0 checkpoint timers and POSIX activity.
**B:** rank-owned activity, including recorded readers. **C:** actual DXT
request-size categories over time. Cross-process timing is coarse, and POSIX
API activity is not physical disk service.

U-Net's small **448 B and 693 B** writes occur with the large checkpoints in
epochs 2 and 4; their exact internal purpose is not identified. CosmoFlow
checkpoint writes appear as aggregate STDIO counters (~1.49 GiB/run), not
POSIX DXT write events; an empty POSIX write timeline does not mean no writes.
Neither checkpoint path adds fsync. Opens are not necessarily metadata RPCs.

## Figures

- U-Net: [4 fast / 0 slow](../results/workloads/plots/dlio/throttled_ssd/unet/darshan/4fast-0slow-pool14.png) · [2 fast / 2 slow](../results/workloads/plots/dlio/throttled_ssd/unet/darshan/2fast-2slow-pool16.png)
- CosmoFlow: [4 fast / 0 slow](../results/workloads/plots/dlio/throttled_ssd/cosmoflow/darshan/4fast-0slow-pool14.png) · [2 fast / 2 slow](../results/workloads/plots/dlio/throttled_ssd/cosmoflow/darshan/2fast-2slow-pool16.png) · [0 fast / 4 slow](../results/workloads/plots/dlio/throttled_ssd/cosmoflow/darshan/0fast-4slow-pool18.png)
- Repeated results: [U-Net](../results/workloads/plots/dlio/throttled_ssd/unet/comparison.csv) · [CosmoFlow](../results/workloads/plots/dlio/throttled_ssd/cosmoflow/comparison.csv)
