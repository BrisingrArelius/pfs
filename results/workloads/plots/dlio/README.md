# DLIO BeeGFS pool comparisons

Choose a workload to see its pool comparison first, then its epoch,
checkpoint, rank-loading and OST-placement figures:

- [ResNet-50](resnet/README.md)
- [U-Net 3D](unet/README.md)

Both sweeps use eight ranks on `anjuna2`, three repetitions per pool, and
the existing Default (`colva4`), HDD (`colva1–3`), and SSD (`colva1–3`)
target sets. Higher training/checkpoint rates or lower whole-run time
describe **these configurations**; pool placement and server fan-out are
confounded. Checkpoint rates time rank 0 only and do not include `fsync`.
These figures do not compare the different workloads' absolute rates to
each other.
