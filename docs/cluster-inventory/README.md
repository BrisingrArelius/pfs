# Storage-node inventory

The following read-only command block was supplied for `colva2`, `colva3`,
and `colva4` on 2026-09-21. `colva1` used earlier separate inventory and
device-model queries; its numeric `targetNumID` values were collected separately
at 2026-09-21T10:07:57+00:00.

```bash
date --iso-8601=seconds
hostname --fqdn
systemctl is-active beegfs-storage

sudo grep -E '^[[:space:]]*(storeStorageDirectory|connInterfacesFile|connUseRDMA|connTCPFallbackEnabled)[[:space:]]*=' \
  /etc/beegfs/beegfs-storage.conf

mapfile -t dirs < <(
  sudo awk -F= '/^[[:space:]]*storeStorageDirectory[[:space:]]*=/{print $2}' \
    /etc/beegfs/beegfs-storage.conf |
  tr ',' '\n' |
  xargs -n1
)

for d in "${dirs[@]}"; do
  echo "===== $d ====="
  sudo grep -H . "$d"/targetID "$d"/targetNumID 2>/dev/null
  findmnt -T "$d" -o TARGET,SOURCE,FSTYPE,OPTIONS
  df -hT "$d"
  df -i "$d"
done

lsblk -e 7 -o NAME,KNAME,PKNAME,TYPE,SIZE,ROTA,VENDOR,MODEL,TRAN,FSTYPE,MOUNTPOINTS
cat /proc/mdstat
sudo pvs
sudo vgs
sudo lvs -a -o lv_name,vg_name,lv_size,devices

sudo cat /etc/beegfs/beegfs-network-interfaces.txt
ip -brief address
ip route
rdma link show
ethtool enp7s0 | grep -E 'Speed:|Duplex:|Link detected:'

ls -l /dev/disk/by-path/
lspci -nn | grep -Ei 'storage|sata|sas|scsi|raid|non-volatile|nvme'
```

## Captured output

- [`colva1`](../../results/cluster-inventory/20260921/colva1.txt)
- [`colva2`](../../results/cluster-inventory/20260921/colva2.txt)
- [`colva3`](../../results/cluster-inventory/20260921/colva3.txt)
- [`colva4`](../../results/cluster-inventory/20260921/colva4.txt)

The text files are summaries transcribed from user-supplied terminal output,
not verbatim raw transcripts. They retain selected configuration, target, mount,
block-device, volume, network, and controller observations.

## Target and device summary

| Host | HDD targets | NVMe targets | Operating-system disk |
|---|---|---|---|
| `colva1` | 101-103 map in order to `/mnt/hdd2..4` and `/dev/sdb1..sdd1` | 104-107 map in order to `/mnt/nvme0..3`; block devices are listed in the host transcript | `/dev/sda` |
| `colva2` | 201-204 map in order to `/mnt/hdd1..4` and `/dev/sda1..sdd1` | 205-207 map in order to `/mnt/nvme1..3` and `/dev/nvme1n1p1..nvme3n1p1` | `/dev/nvme0n1` |
| `colva3` | 301-304 map in order to `/mnt/hdd1..4` and `/dev/sda1..sdd1` | 305-307 map in order to `/mnt/nvme1..3` and `/dev/nvme1n1p1..nvme3n1p1` | `/dev/nvme0n1` |
| `colva4` | 401, 402, 403 map to `/mnt/hdd1`, `/mnt/hdd3`, `/mnt/hdd4` on `/dev/sda1`, `/dev/sdc1`, `/dev/sdd1` | 404 maps to `/dev/nvme0n1p1`; 405-407 map to whole devices `/dev/nvme1n1..nvme3n1` | `/dev/sdb` |

All target filesystems are XFS. No active MD arrays were reported. LVM is used
only by operating-system volumes. SATA disks on each host share the Intel
controller at PCI `00:17.0`; each NVMe disk has a separate PCI controller
endpoint. All storage hosts reported a 2,500-Mb/s full-duplex `enp7s0` link and
no RDMA links.

`colva1` uses HGST HUS722T2TAL HDDs and Samsung SSD 980 PRO 1TB NVMe devices.
The exact drive-model strings for the other storage hosts were not captured.
