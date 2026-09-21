# Historical trace-characterization outputs

| Directory | Original location | Contents |
|---|---|---|
| `contiguity/` | `CONTIG_TESTING_CLAUDE/output/` | Per-file CSV, distribution tables and plots |
| `frequency/` | `FREQ_TESTING_CLAUDE/output/` | Rate split tables and distribution/split plots |

Files are preserved byte-for-byte. An identical contiguity CSV left under
`archive/trace-analysis/` by the interrupted refactor was compared and merged.
No analysis was rerun and no missing outputs were generated.

The source READMEs describe a nine-day partial Polaris corpus (14,900 Darshan
logs, 162,722 file records), not a full year or BeeGFS measurements. The raw logs
are external. Historical counts/claims remain those of the original reports;
this relocation does not independently revalidate them.

The frequency per-file intermediate CSV is not present in this retained output
set. See the [frequency guide](../../../scripts/trace_characterization/frequency/README.md)
and [contiguity guide](../../../scripts/trace_characterization/contiguity/README.md) for
the original processing and provenance. Current trace-analysis scripts place new
outputs under `results/trace_analysis/runs/<run-id>/`.
