# Pooling Scripts

Helper scripts for configuring BeeGFS storage pools used by the workload pipeline.

## Files

- `configure_pools.sh` — create or update HDD/SSD storage pools and mountpoint directories
- `reset_pools.sh` — return the configured directories to the default storage pool

## Usage

Run the pool configuration script with sudo:

```bash
cd scripts/pooling_scripts
sudo ./configure_pools.sh
```

Reset the pool assignment if you want to return the directories to the default pool:

```bash
sudo ./reset_pools.sh
```

## Notes

- These scripts are environment-specific and assume BeeGFS administrative privileges.
- Use them before running the workload pipeline to ensure the workload directories are assigned to the correct storage pools.
