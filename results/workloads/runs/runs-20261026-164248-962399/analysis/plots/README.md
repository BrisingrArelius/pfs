# unet3d_v100 figures

## Pool Comparison

![Pool ranking by input, checkpoint, and whole-run metrics; read the phase-specific caveats.](pool_comparison.png)

Pool ranking by input, checkpoint, and whole-run metrics; read the phase-specific caveats.

## Training Epochs

![Per-epoch training measurements; absent U-Net native rates are not plotted as zeros.](training_epochs.png)

Per-epoch training measurements; absent U-Net native rates are not plotted as zeros.

## Checkpoint Rank0

![Rank-0 checkpoint save completion without fsync.](checkpoint_rank0.png)

Rank-0 checkpoint save completion without fsync.

## Training Repetitions

![Run-to-run comparison across the fixed pool order.](training_repetitions.png)

Run-to-run comparison across the fixed pool order.

## Rank Load P95

![Application-level per-rank loading-step p95, summarized by run.](rank_load_p95.png)

Application-level per-rank loading-step p95, summarized by run.

## Dataset Ost Assignments

![Input file assignments per OST; not bytes or traffic.](dataset_ost_assignments.png)

Input file assignments per OST; not bytes or traffic.
