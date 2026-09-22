#!/usr/bin/env python3
"""Capture the one-time BeeGFS and network path source of truth."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CLUSTER_HOSTS = ("anjuna2", "anjuna3", "colva1", "colva2", "colva3", "colva4")

HOST_SCRIPT = r'''set -u

section() {
    printf '\n===== %s =====\n' "$1"
}

capture() {
    label="$1"
    shift
    section "$label"
    "$@" 2>&1
    rc=$?
    printf '[exit_status=%s]\n' "$rc"
    return 0
}

capture_shell() {
    label="$1"
    command="$2"
    section "$label"
    bash -o pipefail -c "$command" 2>&1
    rc=$?
    printf '[exit_status=%s]\n' "$rc"
    return 0
}

section "identity"
date --iso-8601=seconds 2>&1
hostname --fqdn 2>&1
uname -a 2>&1
printf '[exit_status=0]\n'

capture_shell "tool versions" '
for tool in beegfs-ctl beegfs-net ip ss rdma ethtool iperf3; do
    printf '%s\n' "-- $tool --"
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
        case "$tool" in
            beegfs-ctl) timeout 10s beegfs-ctl --version || true ;;
            beegfs-net) timeout 10s beegfs-net --help 2>&1 | sed -n "1,8p" || true ;;
            iperf3) iperf3 --version || true ;;
            *) "$tool" -V 2>&1 | sed -n "1,3p" || true ;;
        esac
    else
        printf "not installed\n"
    fi
done'

capture_shell "BeeGFS mounts and procfs" '
findmnt -t beegfs,beegfs_nodev -o TARGET,SOURCE,FSTYPE,OPTIONS || true
if [ -d /proc/fs/beegfs ]; then
    find /proc/fs/beegfs -maxdepth 2 -type f -o -type d | sort
else
    printf "/proc/fs/beegfs is absent\n"
fi'

capture_shell "BeeGFS network configuration" '
for config in \
    /etc/beegfs/beegfs-client.conf \
    /etc/beegfs/beegfs-storage.conf \
    /etc/beegfs/beegfs-meta.conf \
    /etc/beegfs/beegfs-mgmtd.conf; do
    [ -e "$config" ] || continue
    printf '%s\n' "-- $config --"
    if [ -r "$config" ]; then
        grep -E "^[[:space:]]*(sysMgmtdHost|connInterfacesFile|connInterfacesList|connRDMAInterfacesFile|connNetFilterFile|connTcpOnlyFilterFile|connUseRDMA|connTCPFallbackEnabled|connPortShift|connStoragePortTCP|connMetaPortTCP|connMgmtdPortTCP)[[:space:]]*=" "$config" || true
        interface_file=$(awk -F= "/^[[:space:]]*connInterfacesFile[[:space:]]*=/{sub(/^[[:space:]]*/, \"\", \$2); sub(/[[:space:]]*$/, \"\", \$2); print \$2; exit}" "$config")
        rdma_file=$(awk -F= "/^[[:space:]]*connRDMAInterfacesFile[[:space:]]*=/{sub(/^[[:space:]]*/, \"\", \$2); sub(/[[:space:]]*$/, \"\", \$2); print \$2; exit}" "$config")
        for listed_file in "$interface_file" "$rdma_file"; do
            [ -n "$listed_file" ] || continue
            printf '%s\n' "-- referenced file: $listed_file --"
            if [ -r "$listed_file" ]; then
                cat "$listed_file"
            else
                printf "missing or unreadable\n"
            fi
        done
    else
        printf "unreadable without elevated privileges\n"
    fi
done'

capture "BeeGFS storage nodes and advertised NICs" timeout 30s \
    beegfs-ctl --listnodes --nodetype=storage --nicdetails
capture "BeeGFS metadata nodes and advertised NICs" timeout 30s \
    beegfs-ctl --listnodes --nodetype=meta --nicdetails
capture "BeeGFS management nodes and advertised NICs" timeout 30s \
    beegfs-ctl --listnodes --nodetype=mgmt --nicdetails
capture "BeeGFS client nodes and advertised NICs" timeout 30s \
    beegfs-ctl --listnodes --nodetype=client --nicdetails
capture "beegfs-net active connection report" timeout 30s beegfs-net

capture "IP addresses" ip -details -json address show
capture "IP routes" ip -details -json route show table all
capture "Interface counters" ip -statistics -statistics -json link show
capture "RDMA links" rdma -j -d link show

capture_shell "Clock synchronization" '
if command -v timedatectl >/dev/null 2>&1; then
    timedatectl show --property=NTPSynchronized --property=NTP --property=Timezone
else
    printf "timedatectl is not installed\n"
fi
if command -v chronyc >/dev/null 2>&1; then
    chronyc tracking || true
fi'

capture_shell "Interface speed and duplex" '
for path in /sys/class/net/*; do
    dev=${path##*/}
    [ "$dev" = lo ] && continue
    printf '%s\n' "-- $dev --"
    for property in operstate speed duplex mtu; do
        printf "%s=" "$property"
        cat "$path/$property" 2>/dev/null || printf "unavailable\n"
    done
    if command -v ethtool >/dev/null 2>&1; then
        ethtool "$dev" 2>/dev/null | grep -E "^[[:space:]]*(Speed|Duplex|Link detected):" || true
    fi
done'

capture_shell "Routes to known cluster addresses" '
for destination in \
    10.1.19.73 10.1.19.74 10.1.19.76 10.1.19.77 \
    192.168.0.2 192.168.0.3 192.168.0.4 192.168.0.5 192.168.0.7; do
    printf '%s\n' "-- $destination --"
    ip -details -json route get "$destination" 2>&1 || true
done'

capture_shell "Established BeeGFS TCP sockets" '
if command -v ss >/dev/null 2>&1; then
    ss -H -t -n -o 2>&1 | grep -E "(:8003|:8005|:8008)([[:space:]]|$)" || printf "no matching established sockets visible\n"
else
    printf "ss is not installed\n"
fi'

capture_shell "TCP settings" '
for key in \
    net.ipv4.tcp_congestion_control \
    net.ipv4.tcp_available_congestion_control \
    net.core.rmem_default net.core.rmem_max \
    net.core.wmem_default net.core.wmem_max \
    net.ipv4.tcp_rmem net.ipv4.tcp_wmem; do
    sysctl "$key" 2>&1 || true
done'
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture read-only BeeGFS configuration, advertised-interface, "
            "active-connection, route, and link evidence for the fixed "
            "network benchmark inventory."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for collected evidence",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        choices=CLUSTER_HOSTS,
        default=list(CLUSTER_HOSTS),
        help="Hosts to collect; defaults to all six cluster hosts",
    )
    parser.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=10,
        help="SSH connection timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--host-timeout",
        type=int,
        default=180,
        help="Maximum collection time per host in seconds (default: 180)",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def local_host_aliases() -> set[str]:
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    return {hostname, hostname.split(".")[0], fqdn, fqdn.split(".")[0]}


def collect_host(
    host: str, ssh_connect_timeout: int, host_timeout: int
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    if host in local_host_aliases():
        command = ["bash", "-s"]
    else:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={ssh_connect_timeout}",
            host,
            "bash",
            "-s",
        ]

    completed = subprocess.run(
        command,
        input=HOST_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=host_timeout,
        check=False,
    )
    return command, completed


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"error: output directory is not empty: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    report: dict[str, object] = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "coordinator": socket.getfqdn(),
        "hosts": {},
    }
    failed = False

    for host in args.hosts:
        print(f"collecting: {host}", flush=True)
        host_started = datetime.now(timezone.utc)
        try:
            command, completed = collect_host(
                host, args.ssh_connect_timeout, args.host_timeout
            )
            transcript = (
                f"Collection requested for: {host}\n"
                f"Coordinator: {socket.getfqdn()}\n"
                f"Started: {host_started.isoformat()}\n"
                f"Command: {' '.join(command)}\n"
                f"\n######## STDOUT ########\n{completed.stdout}"
                f"\n######## STDERR ########\n{completed.stderr}"
            )
            atomic_write_text(output_dir / f"{host}.txt", transcript)
            host_report = {
                "started_at": host_started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_status": completed.returncode,
                "artifact": f"{host}.txt",
            }
            if completed.returncode != 0:
                failed = True
        except subprocess.TimeoutExpired as error:
            failed = True
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            transcript = (
                f"Collection requested for: {host}\n"
                f"Coordinator: {socket.getfqdn()}\n"
                f"Started: {host_started.isoformat()}\n"
                f"Timed out after: {args.host_timeout} seconds\n"
                f"\n######## PARTIAL STDOUT ########\n{stdout}"
                f"\n######## PARTIAL STDERR ########\n{stderr}"
            )
            atomic_write_text(output_dir / f"{host}.txt", transcript)
            host_report = {
                "started_at": host_started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": "timeout",
                "artifact": f"{host}.txt",
            }

        report["hosts"][host] = host_report  # type: ignore[index]
        atomic_write_text(
            output_dir / "collection_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "incomplete" if failed else "completed"
    atomic_write_text(
        output_dir / "collection_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"{report['status']}: {output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
