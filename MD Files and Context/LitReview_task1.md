# Literature Review — Task 1: I/O Profile Classification & Thresholds

Supports [Task 1 - Profiles Setup.md](Task%201%20-%20%20Profiles%20Setup.md): finding literature-backed, Darshan-counter-derivable thresholds for size / access-pattern / type / frequency, used to redefine the profiles in [`profiles_backup.json`](../scripts/workloads/profiles_backup.json).

**Note:** "Location in Paper" fields from the original handout (`Lit_Review_Manoj.pdf`) are known-wrong and omitted. Only dimensions each paper actually covers are listed. Paper 10's stated year (2008) looks inconsistent with its use of the IO500 suite (~2017+) — flagged, not corrected.

---

## Part A — Original 10 Papers

**1. Characterization of I/O Behaviors in Cloud Storage Workloads** (2023, IEEE TC)
- R/W/Mixed: majority of request count decides read- vs. write-dominated. `POSIX_BYTES_READS > POSIX_BYTES_WRITES`
- Frequency: burstiness = req/sec fluctuating across subintervals. `(POSIX_READS+POSIX_WRITES)/Total_Seconds`
- Time: self-similar if Hurst H∈(0.5,1), via inter-arrival deltas.
- *Not covered: size, sequential/random, strided.*

**2. An In-Depth Analysis of Cloud Block Storage Workloads in Large-Scale Production** (2020, IEEE IISWC)
- R/W: write-to-read ratio; write-dominant if >1, very high if >100. `POSIX_WRITES/POSIX_READS`
- Size: small <100 KiB (75% of ops ≤32 KiB read / ≤16 KiB write). `POSIX_SIZE_READ_0_100 + POSIX_SIZE_WRITE_0_100`
- Frequency: burstiness ratio (peak/avg req rate) >100 = high burst.
- Time: burstiness if inter-arrival percentile medians <1.3 ms.
- Random: min distance to previous 32 offsets >128 KiB. `1 − (POSIX_SEQ_READS/POSIX_READS)`
- *Not covered: strided.*

**3. Storage Workload Characterization and Consolidation in Virtualized Environments** (2009, VPACT)
- R/W: read:write ratio, e.g. TPC-C = 2:1. `POSIX_READS/POSIX_WRITES`
- Size: app page sizes 2 KB/8 KB; streams coalesce to >256K.
- Frequency: arrivals/sec (1s, 30s windows) + queue depth bins 1–32+.
- Time: latency 0.5 ms (logs) – 15 ms (data reads).
- Sequential/Random: seek-distance based; small = local, large = random. `POSIX_SEQ_READS/POSIX_READS`
- Strided: MS Exchange — 20% of reads at a consistent 32 KB offset (repeated seek-delta histogram).

**4. Extracting Flexible, Replayable Models from Large Block Traces** (2012, FAST)
- R/W: P(read) from op counts in a trace chunk. `POSIX_READS/(POSIX_READS+POSIX_WRITES)`
- Size: histogram bins <4/8/12/16 KB. `POSIX_SIZE_READ_0_100` vs `POSIX_SIZE_READ_1M_4M`
- Frequency: absolute request count per size/offset bin.
- Time: inter-arrival times, target error <10–15% vs. real throughput/latency.
- Sequential/Random: offset delta between consecutive requests. `POSIX_SEQ_READS/POSIX_READS`
- *Not covered: strided.*

**5. SSD-based Workload Characteristics and Their Performance Implications** (2021, ACM TOS)
- R/W: no global cutoff; e.g. RocksDB analyzed at 50/50 read/update.
- Size: request size vs. device page size; 4 KB = standard fs page size; median writes 1–1024 KB.
- Frequency: "virtual time" counter; study requires ≥10,000 total ops, ≥5,000 writes.
- Time: sensing overhead (t_R) 25–91 µs across flash types.
- Sequential/Random: logical locality = P(page within distance D written within window T). `POSIX_SEQ_READS/POSIX_READS`
- *Not covered: strided.*

**6. Extracting and Characterizing I/O Behavior of HPC Workloads** (2022, IEEE CLUSTER) — **HPC-specific**
- R/W/Metadata-heavy: metadata-heavy when metadata ops dominate (CosmoFlow: 98% of I/O time). `(POSIX_OPENS+POSIX_F_CLOSE_START_TIMESTAMP)/(POSIX_READS+POSIX_WRITES+POSIX_OPENS+POSIX_F_CLOSE_START_TIMESTAMP)`
- Size: **small <4 KB or 64 KB; large >16 MB.**
- Frequency: ops distribution, e.g. Montage MPI: 4M reads vs 1M writes moving 21 GB.
- Time: % of runtime as I/O (11% CM1, 75% HACC).
- Sequential: continuous-stream access, no significant seeks.
- Strided: 3D array / library "chunking"; needs POSIX histograms of repeating gaps or MPI-IO counters.

**7. Workload Characterization for Enterprise Disk Drives** (Kashyap, 2018, ACM TOS 14(2), Art. 19 — [full text](https://dl.acm.org/doi/10.1145/3151847))
- R/W: no formal `>=` cutoff is stated in the paper — verified quotes are: abstract, "reads are the dominant workload accounting for **80%** of the accesses to the drive"; §2.3(3), "Reads are the dominant workload accounting for **60–80%** of the total bytes transferred"; §5 conclusion, "~80% of the overall throughput." (`POSIX_READS/(POSIX_READS+POSIX_WRITES)` is our own Darshan-counter mapping for "accesses," `POSIX_BYTES_READ/(POSIX_BYTES_READ+POSIX_BYTES_WRITTEN)` for "bytes transferred" — not the paper's own formula.)
- Size: small <100 KB; large >100 KB.
- Frequency: 90% of ops <100 KB by count, but 75–95% of bytes in the >100 KB ops.
- Time: throughput/IOPS over fixed 2-hr windows.
- Sequential/Random: random=seek required; sequential=0-block gap; near-sequential=few-block gap.
- *Not covered: strided (mentions "near-sequential" only).*

**8. Optimization of Reading Data via Classified Block Access Patterns in File Systems** (2016, IEEE Access)
- R/W: read ratio 64–99% for "read-intensive" study set.
- Size: avg read size 8.2 KB (TPC-C) – 59.3 KB (src1). `POSIX_BYTES_READ/POSIX_READS`
- Frequency: block-access events within a "prediction window."
- Time: avg response-time reduction of 14.6–17.9% vs. sequential prefetch.
- Sequential/Random: consecutive-block access after a cache miss.
- Strided: qualitative — main reason sequential prefetch fails; needs recurring seek-distance histograms.

**9. Characterization of Storage Workload Traces from Production Windows Servers** (2008, IEEE IISWC)
- R/W/Metadata-heavy: metadata (NTFS logs) skews ratio e.g. 1:1→1:2.
- Size: common modes 4 KB (NTFS), 8 KB (Exchange), 64 KB (large-transfer).
- Frequency: Total IOs vs. Total GB transferred.
- Time: Hurst H≫0.5 = strong self-similarity/burstiness.
- Sequential: offset delta == 0. Verified quote (Section V.A, WBS trace specifically): "4% of write requests and 13% of read requests... are exactly sequential." Table I's overall "% Seq IOs Initiated" spans a much wider range across the paper's 12 traces — **0.77% to 95.25%** (e.g. DAP-DS: 0.77%, LM-TBE: 70.43%). No single number here — the original "4–32%" note in this review was an imprecise paraphrase; corrected.
- Strided: qualitative "jumps," often in 4 KB multiples.

**10. Understanding and Predicting Cross-Application I/O Interference in HPC Storage Systems** (stated 2008 IEEE IISWC — year looks off given IO500 usage, unverified) — **HPC-specific**
- R/W/Metadata: IO500 tasks "ior-easy-write" (data) vs. "mdt-easy" (metadata). `POSIX_OPENS+POSIX_STATS+POSIX_F_CLOSE_START_TIMESTAMP` vs. `POSIX_READS+POSIX_WRITES`
- Size: "ior-easy" = large sequential; "ior-hard" = small/unaligned.
- Frequency: total ops/time window; slowdown bands mild(<2x)/moderate(2–5x)/severe(≥5x).
- Time: degradation = avg(iotime_interference/iotime_baseline).
- Sequential/Random: "ior-easy" vs. "ior-hard" task sets.
- Strided: IO500 "hard" tasks, non-contiguous, slowdowns up to 40.9x — **requires Darshan DXT logs**, not standard POSIX counters.

---

## Part B — Additional Candidate Papers (not in original handout)

Papers 1–10 are mostly general-purpose (cloud/enterprise/Windows/SSD) with Darshan formulas retrofitted after the fact; only #6 and #10 are HPC-native, and none originate from Darshan itself. These fill that gap:

1. **Carns et al., "24/7 Characterization of Petascale I/O Workloads"** (IASDS 2009) — [PDF](https://www.mcs.anl.gov/uploads/cels/papers/P1660.pdf). The paper that introduced Darshan; real size/pattern stats from DOE apps, directly counter-compatible.
2. **Luu et al., "A Multiplatform Study of I/O Behavior on Petascale Supercomputers"** (HPDC 2015) — [PDF](https://sdm.lbl.gov/~sbyna/research/papers/201506-HPDC-iologs.pdf). Mines 1M+ real Darshan logs across DOE/ALCF machines over 6 years — same methodology as this project's plan. **Best methodological match.**
3. **Snyder et al., "Techniques for Modeling Large-Scale HPC I/O Workloads"** (PMBS 2015) — [PDF](https://sdm.lbl.gov/~sbyna/research/papers/201511-PMBS2015-IOWorkload.pdf). Synthesizes synthetic I/O workloads from Darshan characterizations — same goal as `profiles_backup.json`/`posix_synthetic_workloads.c`.
4. **Patel et al., "Uncovering Access, Reuse, and Sharing Characteristics of I/O-Intensive Files"** (USENIX FAST 2020) — [link](https://www.usenix.org/conference/fast20/presentation/patel-hpc-systems). Covers SSF/FPP/partial-shared file access from Theta/Summit — fills the sharing-pattern gap, ties to the Summer Plan's worked example.
5. **Lockwood et al., "A Year in the Life of a Parallel File System"** (SC 2018) — [PDF](https://sdm.lbl.gov/~sbyna/research/papers/201811-SC18-YearLifePFS.pdf). Full year of production PFS data (Cori/Edison/Mira) — matches this project's "extend to 1 year of logs" plan; good source for long-timescale frequency framing.
6. **Drishti** (Bez, Ather, Byna, "Drishti: Guiding End-Users in the I/O Optimization Journey," PDSW 2022) — [PDF](https://pdsw.org/pdsw22/papers/bez-pdsw2022.pdf) / [docs](https://drishti-io.readthedocs.io/) / [source](https://github.com/hpc-io/drishti-io). Small-request check: **<1 MB, flagged if >10% of read or write ops** (`drishti/includes/config.py`: `small_bytes=1048576`, `small_requests=0.1`). Verified basis (not stated in the docs, confirmed by reading the paper + source directly):
   - **1 MB cutoff** is not independently derived — it falls out of Darshan's own `POSIX_SIZE_READ_*`/`POSIX_SIZE_WRITE_*` histogram bucket boundaries (`...,100K_1M | 1M_4M,...`); Drishti just sums everything below the `1M_4M` bucket.
   - **"Small is bad" claim** is attributed by the paper to two external root-cause studies, not derived in-paper: Wang et al., "A Zoom-in Analysis of I/O Logs to Detect Root Causes of I/O Performance Bottlenecks" (CCGRID 2019), and Wang et al., "IOMiner" (CLUSTER 2018) — see #7 below.
   - **10% trigger threshold** is an admin-tunable default ("system administrators can tune based on each platform's characteristics"), not statistically fitted — no derivation given.
   - **Empirical validation (post-hoc, not the basis for the number):** applied to 112,612 real Cori Darshan logs (Mar 1–5, 2022), this combination flagged 57.59% of jobs for small reads and 57.32% for small writes — evidence the threshold catches something common in production HPC, but not evidence for why 1 MB/10% specifically.
7. **Wang et al., "A Zoom-in Analysis of I/O Logs to Detect Root Causes of I/O Performance Bottlenecks"** (CCGRID 2019). The actual root-cause study Drishti cites for "small I/O harms performance" — worth pulling directly if a more rigorously-derived size threshold is needed than Drishti's convenience cutoff.

**Open gap:** no strong Darshan-native source found for strided/nd_strided thresholds specifically — check HDF5/parallel-NetCDF I/O literature separately.

---

## Part C — Strided / nd_strided Access Pattern (HDF5, PnetCDF, MPI-IO Literature)

Follow-up on Part B's open gap. None of these sources originate from Darshan either, but they're the literature that actually defines strided/nd_strided access structurally (as opposed to Papers 3/6/8/9/10's qualitative mentions).

1. **Bez, Byna, Ibrahim, "I/O Access Patterns in HPC Applications: A 360-Degree Survey"** (2023, ACM Computing Surveys) — [PDF](https://sbyna.github.io/research/papers/2023/2023-IOpatterns-360-degree.pdf). The most rigorous formal definition found in this whole review:
   - Contiguous/sequential: `off_{p,i+1} = off_{p,i} + size_{p,i}` (each request starts where the previous one ended).
   - **Simple-strided**: `off_{p,i+1} = off_{p,i} + stride_i`, where `stride_i` is constant across requests — i.e. same request size *and* same offset increment every time.
   - Multi-dimensional patterns: described via **HDF5 hyperslabs** — four arrays (offset, stride, count, block) per dimension — and MPI derived datatypes/PnetCDF selectors, but the paper does **not** formalize a general nd_strided classification criterion or numeric threshold; it stops at describing the mechanism.
   - **No quantitative thresholds anywhere** for classifying contiguous vs. strided vs. random.
   - Directly confirms the Darshan gap: "Darshan Extended Tracing [DXT] does not yet capture fine-grained information about high-level libraries, such as HDF5," requiring manual instrumentation/timestamps instead — and states no existing tool observes access patterns consistently across all I/O-stack layers.
   - *Best available formal citation for "strided," but confirms rather than closes the gap.*

2. **Thakur, Gropp, Lusk, "Optimizing Noncontiguous Accesses in MPI-IO"** (ANL/MCS-TM-234, 1999/2002) — [PDF](https://web.cels.anl.gov/~thakur/papers/mpi-io-noncontig.pdf). The foundational ROMIO paper.
   - Defines noncontiguity via MPI derived datatypes (`vector`/`hvector`, `indexed`/`hindexed`, `subarray`, `darray`) and MPI file views (`MPI_File_set_view`).
   - No general threshold for choosing data sieving vs. two-phase collective I/O based on a noncontiguity metric — one qualitative data point: holes larger than **5×** the data-segment size degrade data-sieving performance (and ROMIO doesn't auto-detect/adapt to this).
   - Confirms noncontiguous-pattern detection needs per-request structure (stride sizes, segment/hole locations) — not derivable from aggregate byte/op counts.
   - *Not covered: nd_strided/HDF5-specific detail (this predates HDF5's modern hyperslab-heavy usage).*

3. **Kang, Breitenfeld, Hou, Liao, Ross, Byna, "Optimizing Performance of Parallel I/O Accesses to Non-contiguous Blocks in Multiple Array Variables"** (2021, IEEE BigData) — [PDF](https://sdm.lbl.gov/~sbyna/research/papers/2021/2021-IEEE-BigData-multi-dataset.pdf). Closest thing to an actual nd_strided characterization study.
   - Characterizes nd_strided-style access as per-process lists of n-dimensional sub-arrays across many datasets — e.g. E3SM climate model "F case": 402 datasets, 21,632 processes, **1.37 billion** total I/O blocks.
   - Ties directly to HDF5 hyperslabs: `H5Sselect_hyperslab` + `H5S_SELECT_OR` to coalesce multiple sub-array selections into one dataspace.
   - No general block-count/size/stride thresholds defined — workloads are characterized descriptively per-application, not via a cutoff.
   - Confirms trace/iteration-level profiling is what's actually needed (two-phase I/O iteration counts, per-phase timing breakdown) — exceeds anything in Darshan's aggregate summary.
   - *Not covered: PnetCDF (paper only discusses HDF5/ADIOS).*

4. **Li, Liao, Choudhary, Ross, Thakur, Gropp, Latham, Siegel, Gallagher, Zingale, "Parallel netCDF: A High-Performance Scientific I/O Interface"** (SC 2003) — [PDF](https://parallel-netcdf.github.io/wiki/pnetcdf-sc2003.pdf). The paper that defines PnetCDF's strided-access API.
   - Formally defines strided subarray access via the **"vars" family**: `start[]`/`count[]`/`stride[]` (plus `imap[]` for mapped/strided-in-memory access) — the canonical PnetCDF strided-access primitive.
   - No regularity/gap threshold given for classification.
   - Confirms per-call, file-view-level detail is required — not aggregate.

**Verdict on the gap:** confirmed genuinely open, not just unexplored. Every source here that actually *defines* strided/nd_strided access does so structurally (offset/stride equations, MPI derived datatypes, HDF5 hyperslab arrays, PnetCDF start/count/stride) and explicitly says classification needs per-request or trace-level (DXT) granularity — none reduce to a validated Darshan POSIX/MPI-IO summary-counter threshold. Even **Snyder et al. 2015** (Part B #3), despite being specifically about synthesizing synthetic workloads from Darshan data, only uses Darshan's access-*size* histograms and does no stride-aware reconstruction at all — reinforcing that this isn't a citation-search gap, it's a real methodological gap in the field.

The closest Darshan-native proxy remains the raw counters Darshan does expose — `POSIX_STRIDE1_STRIDE4` (the four most frequent strides) and `POSIX_STRIDE1_STRIDE4_COUNT` (their counts) — which could support a heuristic like "strided if the top stride's count / total accesses exceeds some fraction X%." That heuristic would be **this project's own construction**, not one validated by any paper reviewed here, and should be documented as such if adopted. **nd_strided has no Darshan-counter proxy at all** — Darshan's stride counters are flat (one stride value per record), not dimensional, so there's no native way to distinguish "strided" from "multi-dimensionally strided" without DXT trace data.

---

## Part D — Empirical Verification: Contiguous-Ratio Distribution on Real ALCF Polaris Logs

Since no paper reviewed (Parts A–C) gives a validated file-level threshold for "% of ops
contiguous ⇒ classify file as sequential" (see Cross-Paper Synthesis below), this was
derived directly from real Darshan logs instead of literature. Method, code, and full
results: [`CONTIG_TESTING_CLAUDE/`](../CONTIG_TESTING_CLAUDE/README.md).

**Data:** real ALCF Polaris Darshan logs (verified genuine via mount points `/lus/eagle`,
`/lus/grand`, `/home`, `/local/scratch`, Slingshot `cxi0`/`cxi1` interfaces, and the
`[jobid]-[random_val].darshan` anonymized naming scheme matching the ALCF collection's own
documentation). 12,301 logs / 89,389 file records, covering 9 days (2024-04-24–30,
2024-05-11/12) — a partial slice, not the full year-long collection.

**Method:** for every file record, `contig_ratio = (POSIX_CONSEC_READS + POSIX_CONSEC_WRITES) / (POSIX_READS + POSIX_WRITES)`, aggregated across MPI ranks per file, binned into 5%-wide buckets across all files.

**Finding — the raw distribution is dominated by a quantization artifact, not real signal:**
unfiltered, the shape looks trimodal (46.6% of files at 0–5%, 14.4% at 50–55%, 36.1% at
95–100%). This is mostly a counting artifact: `POSIX_CONSEC_*` can never flag a file's
*first* op, so a file with 1 total op is **structurally forced to 0%** regardless of what
that op actually did, and a file with 2 total ops can only land at 0% or 50% — nothing
else is possible. Verified directly: the 0–5% bucket has a median of **1** total op
(99.5% have ≤4); the 50–55% bucket has a median of **2** (82.4% have ≤4). Only the
95–100% bucket has real weight behind it (median **864** total ops).

**Filtering to `total_ops ≥ 20`** (32,852 files) collapses this almost entirely: **98.11%
of files land at 95–100% contiguous**, mean 98.70%, median 99.88%.

**Implication for thresholds:** a file-level contiguous-ratio threshold is only meaningful
above a minimum-op-count floor (this data suggests ≥20 as a reasonable floor — below it
the ratio is quantized, not descriptive). Above that floor, in this real HPC sample,
"sequential" isn't a fuzzy percentage call at all — the overwhelming majority of files
that do meaningful I/O are essentially fully contiguous (≥95%), which is a much cleaner
empirical basis than any of Papers 1–10 or Paper 9's 0.77–95.25% enterprise-Windows
range. This is HPC-native, Darshan-native, and Polaris-specific (single-platform,
9-day sample) — worth re-running against the fuller collection once available to check
it holds beyond this slice.

---

## Cross-Paper Synthesis

- **Size:** ~4 KB "small" is consistent across papers 4/5/6/9 — matches current profile threshold. "Large" varies widely (100 KB–4 MB) except the HPC-specific Paper 6, which uses **>16 MB** — best basis to revise the current 4 MB+ cutoff. Drishti (Part B #6) offers a third, HPC-production-validated anchor at **<1 MB**, though its cutoff traces back to Darshan's own histogram bucket boundaries rather than an independent derivation — useful as a middle tier, not a replacement for Paper 6's >16 MB "large" bound.
- **Type:** Read-ratio formula is universal; "dominant" cutoffs range 50–80%+. **Metadata-heavy** recurs as a distinct 5th category (Papers 6, 10) — worth adding.
- **Frequency:** Weakest-supported dimension — no paper uses a flat op-count cutoff like the current "<5K seldom/>20K frequent." All use rate/burstiness (ops/sec, peak:avg ratio, Hurst parameter). Should be reframed as a rate, not a raw count. Part D's empirical check adds one concrete data point regardless: a minimum-op-count **floor** (≥20) is needed before *any* per-file ratio counter is meaningful — below it, low op counts quantize ratios like contig_ratio into a few discrete values that look like signal but aren't.
- **Access pattern:** Sequential/random via `POSIX_SEQ_READS/POSIX_READS` is solid and Darshan-native, and now has real-data backing: Part D found real ALCF Polaris files (`total_ops ≥ 20`) are 98.11% contiguous (95–100% bucket) — a much cleaner empirical signal than any literature threshold reviewed. **Strided/nd_strided is a confirmed, genuine gap, not just an unexplored one** — Papers 3, 6, 9, 10 flagged it qualitatively, and Part C's HPC-I/O-library-native literature (Bez et al. 2023's formal `off_{p,i+1}=off_{p,i}+stride_i` definition, Thakur & Gropp's MPI derived-datatype/file-view model, Kang et al. 2021's HDF5-hyperslab nd_strided study, Li et al. 2003's PnetCDF `start/count/stride` API) all independently confirm that classifying strided/nd_strided access requires per-request or DXT trace-level detail — none reduce to a validated Darshan summary-counter threshold. Best available Darshan-native proxy: `POSIX_STRIDE1_STRIDE4`/`_COUNT` (top-4 strides + counts) for a **strided** heuristic (unvalidated by any paper, our own construction); **nd_strided has no Darshan-counter proxy at all** since Darshan's stride counters are flat, not dimensional.
- **Missing dimension:** file **sharing pattern** (SSF/FPP/partial-shared) recurs in HPC literature (Patel et al., Summer Plan) but isn't one of the current four parameters.
