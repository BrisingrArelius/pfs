# Historical storage-pool helpers

This folder retains historical BeeGFS pool-management scripts for provenance.
They are **not** a reviewed live inventory, a reversible state transaction, or
a benchmark runner. D/S/H comparisons belong to the separate
[DLIO application experiment](../../../docs/specs/DL_Experiment.md); the
[microbenchmark suite](../../../docs/specs/MicroBenchmarks.md) measures layer
effects only.

## Files

- `configure_pools.sh` — create or update HDD/SSD storage pools and move hard-coded targets
- `reset_pools.sh` — move hard-coded targets to Default and remove named HDD/SSD pools

## Historical usage

Do not run these scripts as DLIO setup or restoration.
They encode old target memberships and cannot restore the exact live baseline.
DLIO needs a reviewed inventory and reversible pool transactions with verified
restoration instead.

## Notes

- These scripts are environment-specific and assume BeeGFS administrative privileges.
- The scripts move target membership, not existing file contents or directory patterns.
- Their 4xx targets and target 104 media assignment conflict with historical
  context notes. Resolve live inventory separately before treating them as current setup.
