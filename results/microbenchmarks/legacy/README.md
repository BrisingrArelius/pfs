# Historical microbenchmark evidence

| Directory | Source and interpretation |
|---|---|
| `fio-local/20260404/` | Two local-OST JSON datasets, original plots and analyzer transcript |
| `fio-beegfs/20260402/` | Three original text summaries restored from Git commit `edc3d69` |
| `fio-beegfs/20260416/` | Structured end-to-end BeeGFS HDD/SSD FIO matrix |
| `network/` | Original iperf outputs, interface/switch notes, client config and capacity snapshot |
| `placement/from-logs/` | Original OST space/usage logs and `du` evidence from `logs_and_checkpoints/` |
| `placement/from-scripts/` | Separate original script-local OST log |
| `placement/plots/` | Original script-local heatmaps, including the `23rd april/` group |

## Local FIO (April 4)

`matrix_results_20260404_220033.json` is the main 840-row dataset behind the
preserved OST plots. `matrix_results_20260404_210221.json` is preliminary.
Labels `HDD_OST1..4` and `SSD_OST1..3` are runner labels, not verified BeeGFS target
IDs. The runner constructs local mount paths; historical host/device mapping is
unresolved. These tests bypass BeeGFS client-to-storage-server transport.

The retained analyzer transcript originally lived at `scripts/fio/r.txt` and
identifies the main dataset explicitly. Existing plot names and labels are kept
as historical output. `fsize` is passed to FIO as `size` alongside `nrfiles`; do
not assume the label describes each individual file's size. No labels or images
were regenerated during the refactor.

## BeeGFS-pool FIO (April 2 and April 16)

The April 2 source summaries were recovered from
`edc3d69:scripts/fio/results/`. They contain summary values rather than a retained
per-repetition dataset. The legacy duplicate random-read WRITE parser artifact
must not be interpreted as a separate measured write workload. A previously
derived normalized JSON and plots were lost in undo and were not recreated.
One summary value cannot support a variability/box-plot claim.

The April 16 dataset has 240 rows with HDD/SSD labels. Its filename records
`20260416_023543`; that is not its Git commit date. Historical file counts and
`fsize` labels are retained literally with the same FIO geometry caveat above.

## Network and placement limits

Approximately 2.35 Gbit/s TCP agrees numerically with approximately 280–282 MiB/s
sequential pool throughput. That is suggestive, not proof of the active BeeGFS
transport: `connUseRDMA = true` allows RDMA and TCP fallback. Record the actual
connection evidence in future runs. These snapshots do not establish current
topology or pool membership.

Some placement logs contain offline-node errors. Keep those errors as experiment
context. Heatmap-to-log attribution, exact commands and run manifests are not
fully recovered; directory grouping is provenance preservation, not a claim that
every adjacent figure was generated from every adjacent log.
