# Pooling Scripts

Helper scripts for configuring BeeGFS storage pools used by the workload pipeline.

Relocated unchanged from `scripts/pooling_scripts/`. These scripts contain
historical target inventories and are not verified against the current cluster.
See [methodology](../../../docs/methodology.md) for D/S/H and
[deferred path repairs](../../../TODOS_SCRIPT_CHANGES.md). The revised experiment
is documentation, not newly implemented pool-management behavior.

## Files

- `configure_pools.sh` — create or update HDD/SSD storage pools and move hard-coded targets
- `reset_pools.sh` — move hard-coded targets to Default and remove named HDD/SSD pools

## Historical usage (inventory review required)

Run the pool configuration script with sudo:

```bash
cd scripts/microbenchmarks/placement
sudo ./configure_pools.sh
```

The historical reset moves targets back to Default:

```bash
sudo ./reset_pools.sh
```

## Notes

- These scripts are environment-specific and assume BeeGFS administrative privileges.
- The scripts move target membership, not existing file contents or directory patterns.
- Their 4xx targets and target 104 media assignment conflict with historical
  context notes. Resolve live inventory separately before treating them as current setup.
