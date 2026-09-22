#!/usr/bin/env python3
"""Coordinate durable, resumable iperf3 TCP measurements from anjuna3."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from run_support import (
    BudgetExpired,
    atomic_json,
    atomic_text,
    duration,
    make_directory,
    now,
    read_json,
    remaining_seconds,
    safe_artifact,
    set_deadline,
    sync_directory,
)


HERE = Path(__file__).resolve().parent
CLIENTS = ("anjuna2", "anjuna3")
SERVERS = ("colva1", "colva2", "colva3", "colva4")
PROCESS_FILES = ("output.json", "stderr", "identity", "exit_status", "argv")

REMOTE_PROCESS_SCRIPT = r'''set -u
directory=$1
watchdog=$2
release_epoch=$3
shift 3
umask 077
parent=${directory%/*}
mkdir -p "$parent"
if ! mkdir "$directory"; then
    printf 'remote artifact directory already exists: %s\n' "$directory" >&2
    exit 73
fi
printf '%q ' "$@" >"$directory/argv"
printf '\n' >>"$directory/argv"
supervisor_pid=$$
supervisor_start=$(awk '{print $22}' "/proc/$supervisor_pid/stat")
boot=$(cat /proc/sys/kernel/random/boot_id)
uid=$(id -u)
write_identity() {
    temporary="$directory/.identity-$$"
    {
        printf 'supervisor_pid=%s\n' "$supervisor_pid"
        printf 'supervisor_start=%s\n' "$supervisor_start"
        printf 'barrier_pid=%s\n' "${barrier_pid:-0}"
        printf 'barrier_start=%s\n' "${barrier_start:-0}"
        printf 'pid=%s\n' "${pid:-0}"
        printf 'pgid=%s\n' "${pgid:-0}"
        printf 'start=%s\n' "${start:-0}"
        printf 'boot=%s\n' "$boot"
        printf 'uid=%s\n' "$uid"
        printf 'release_epoch=%s\n' "$release_epoch"
        printf 'launched_epoch_ns=%s\n' "${launched_epoch_ns:-0}"
    } >"$temporary"
    mv "$temporary" "$directory/identity"
}
stop_child() {
    if [ "${barrier_pid:-0}" != "0" ]; then
        kill -TERM -- "$barrier_pid" 2>/dev/null || true
    fi
    if [ "${pgid:-0}" != "0" ]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
}
trap 'stop_child; exit 143' TERM INT HUP
barrier_pid=0
barrier_start=0
if [ "$release_epoch" != "0" ]; then
    delay=$(python3 -c 'import sys,time; print(max(0.0, float(sys.argv[1])-time.time()))' "$release_epoch")
    sleep "$delay" &
    barrier_pid=$!
    barrier_start=$(awk '{print $22}' "/proc/$barrier_pid/stat")
fi
write_identity
if [ "$barrier_pid" != "0" ]; then
    wait "$barrier_pid" || exit $?
    barrier_pid=0
    barrier_start=0
fi
launched_epoch_ns=$(date +%s%N)
setsid timeout --signal=TERM --kill-after=5 "$watchdog" "$@" \
    >"$directory/output.json" 2>"$directory/stderr" &
pid=$!
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
start=$(awk '{print $22}' "/proc/$pid/stat")
boot=$(cat /proc/sys/kernel/random/boot_id)
uid=$(id -u)
write_identity
wait "$pid"
rc=$?
printf '%s\n' "$rc" >"$directory/exit_status"
exit "$rc"
'''

READINESS_SCRIPT = r'''set -u
directory=$1
address=$2
port=$3
[ -r "$directory/identity" ] || exit 1
pid=$(awk -F= '$1=="pid"{print $2}' "$directory/identity")
start=$(awk -F= '$1=="start"{print $2}' "$directory/identity")
boot=$(awk -F= '$1=="boot"{print $2}' "$directory/identity")
[ "$(cat /proc/sys/kernel/random/boot_id)" = "$boot" ] || exit 3
[ -r "/proc/$pid/stat" ] || exit 2
[ "$(awk '{print $22}' "/proc/$pid/stat")" = "$start" ] || exit 3
ss -H -ltn "sport = :$port" | grep -F -- "$address:$port" >/dev/null
'''

CLEANUP_SCRIPT = r'''set -u
directory=$1
if [ ! -d "$directory" ]; then
    exit 0
fi
count=0
while [ ! -r "$directory/identity" ] && [ "$count" -lt 20 ]; do
    sleep 0.1
    count=$((count + 1))
done
[ -r "$directory/identity" ] || exit 4
supervisor_pid=$(awk -F= '$1=="supervisor_pid"{print $2}' "$directory/identity")
supervisor_start=$(awk -F= '$1=="supervisor_start"{print $2}' "$directory/identity")
barrier_pid=$(awk -F= '$1=="barrier_pid"{print $2}' "$directory/identity")
barrier_start=$(awk -F= '$1=="barrier_start"{print $2}' "$directory/identity")
pid=$(awk -F= '$1=="pid"{print $2}' "$directory/identity")
pgid=$(awk -F= '$1=="pgid"{print $2}' "$directory/identity")
start=$(awk -F= '$1=="start"{print $2}' "$directory/identity")
boot=$(awk -F= '$1=="boot"{print $2}' "$directory/identity")
uid=$(awk -F= '$1=="uid"{print $2}' "$directory/identity")
[ "$(cat /proc/sys/kernel/random/boot_id)" = "$boot" ] || exit 3
[ "$uid" = "$(id -u)" ] || exit 3
if [ -r "/proc/$supervisor_pid/stat" ]; then
    [ "$(awk '{print $22}' "/proc/$supervisor_pid/stat")" = "$supervisor_start" ] || exit 3
fi
if [ "$pid" = "0" ]; then
    if [ "$barrier_pid" != "0" ] && [ -r "/proc/$barrier_pid/stat" ]; then
        [ "$(awk '{print $22}' "/proc/$barrier_pid/stat")" = "$barrier_start" ] || exit 3
        kill -TERM -- "$barrier_pid" 2>/dev/null || true
    fi
    if [ -r "/proc/$supervisor_pid/stat" ]; then
        kill -TERM -- "$supervisor_pid" 2>/dev/null || true
    fi
    exit 0
fi
if [ ! -r "/proc/$pid/stat" ]; then
    exit 0
fi
[ "$(awk '{print $22}' "/proc/$pid/stat")" = "$start" ] || exit 3
[ "$pid" = "$pgid" ] || exit 3
kill -TERM -- "-$pgid" 2>/dev/null || true
count=0
while [ -r "/proc/$pid/stat" ] && [ "$count" -lt 50 ]; do
    sleep 0.1
    count=$((count + 1))
done
if [ -r "/proc/$pid/stat" ]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
fi
exit 0
'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "network_config.json")
    parser.add_argument("--inventory", type=Path, default=HERE / "network_inventory.json")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--time-limit", type=duration)
    timing.add_argument("--deadline", help="ISO-8601 timestamp with timezone")
    timing.add_argument("--extend-deadline", type=duration)
    parser.add_argument("--cleanup-buffer", type=duration, default=300)
    args = parser.parse_args(argv)
    try:
        if args.extend_deadline is not None and (args.resume or args.pilot):
            raise ValueError("extension mode does not accept --resume or --pilot")
        if args.deadline:
            parsed = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("deadline needs a timezone")
            args.deadline = parsed.timestamp()
        if args.time_limit is not None and args.time_limit <= args.cleanup_buffer:
            raise ValueError("time limit must exceed cleanup buffer")
    except ValueError as error:
        parser.error(str(error))
    args.results_dir = args.results_dir.absolute()
    return args


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_config(config):
    required = {
        "protocol_version", "coordinator", "repetitions", "order_seed", "transport",
        "directions", "isolated_streams", "concurrent_streams", "omit_seconds",
        "duration_seconds", "interval_seconds", "barrier_lead_seconds", "ports", "validation", "timeouts",
        "planning", "remote_root",
    }
    if set(config) != required:
        raise ValueError("network configuration keys differ from the supported schema")
    if config["protocol_version"] != 1 or config["coordinator"] != "anjuna3":
        raise ValueError("require network protocol 1 coordinated from anjuna3")
    if config["transport"] != "tcp" or config["directions"] != ["client_to_oss", "oss_to_client"]:
        raise ValueError("require the fixed two-direction TCP protocol")
    if config["repetitions"] != 5 or config["isolated_streams"] != [1, 4] or config["concurrent_streams"] != 4:
        raise ValueError("require five repetitions with one/four isolated streams and four concurrent streams")
    if set(config["ports"]) != set(CLIENTS) or len(set(config["ports"].values())) != 2:
        raise ValueError("require one distinct port per client")
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in config["ports"].values()):
        raise ValueError("iperf3 ports must be unprivileged integers")
    positive = [config["order_seed"], config["omit_seconds"], config["duration_seconds"],
                config["interval_seconds"], config["barrier_lead_seconds"], *config["ports"].values(),
                *config["validation"].values(), *config["timeouts"].values(),
                *config["planning"].values()]
    if any(not is_number(value) or value <= 0 for value in positive):
        raise ValueError("network timing, port, validation and planning values must be positive")
    traffic_floor = (config["barrier_lead_seconds"] + config["omit_seconds"]
                     + config["duration_seconds"])
    if config["timeouts"]["client_seconds"] <= traffic_floor:
        raise ValueError("client timeout must exceed barrier, omit and measured duration")
    if config["planning"]["isolated_seconds"] <= traffic_floor:
        raise ValueError("isolated planning estimate cannot cover barrier and traffic")
    if config["planning"]["concurrent_epoch_seconds"] <= traffic_floor:
        raise ValueError("concurrent planning estimate cannot cover barrier and traffic")
    minimum_watchdog = (config["timeouts"]["readiness_seconds"] + config["barrier_lead_seconds"]
                        + config["omit_seconds"] + config["duration_seconds"]
                        + config["timeouts"]["server_exit_seconds"])
    if config["timeouts"]["remote_watchdog_seconds"] <= minimum_watchdog:
        raise ValueError("remote watchdog cannot cover readiness, barrier and measurement")
    root = Path(config["remote_root"])
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise ValueError("remote_root must be a confined absolute directory")


def validate_inventory(inventory):
    if inventory.get("schema_version") != 1:
        raise ValueError("unsupported network inventory schema")
    if inventory.get("confirmed") is not True:
        raise ValueError("network inventory is not confirmed; capture and review the path source of truth")
    if inventory.get("transport") != "tcp" or inventory.get("rdma", {}).get("available") is not False:
        raise ValueError("this runner requires a confirmed TCP-only inventory")
    hosts = inventory.get("hosts", {})
    if set(hosts) != set(CLIENTS + SERVERS):
        raise ValueError("inventory must contain exactly two clients and four storage servers")
    for host, item in hosts.items():
        required = ("ssh_host", "clock_synchronized")
        if any(item.get(key) in (None, "") for key in required):
            raise ValueError(f"inventory host {host} has unconfirmed fields")
        if not re.fullmatch(r"[a-zA-Z0-9.-]+", item["ssh_host"]):
            raise ValueError(f"unsafe SSH host for {host}")
        if item["clock_synchronized"] is not True:
            raise ValueError(f"clock synchronization is not confirmed for {host}")
    paths = inventory.get("paths", {})
    if set(paths) != set(CLIENTS):
        raise ValueError("inventory paths must cover both clients")
    for client in CLIENTS:
        if set(paths[client]) != set(SERVERS):
            raise ValueError(f"inventory paths for {client} must cover every OSS")
        for server in SERVERS:
            path = paths[client][server]
            endpoint_fields = ("source_address", "source_interface", "source_mtu",
                               "source_link_mbps", "destination_address",
                               "destination_interface", "destination_mtu",
                               "destination_link_mbps")
            if (any(path.get(key) in (None, "") for key in endpoint_fields)
                    or path.get("evidence_level") not in {"confirmed", "active_confirmed"}):
                raise ValueError(f"path {client}->{server} is not confirmed")
            ipaddress.ip_address(path["source_address"])
            ipaddress.ip_address(path["destination_address"])
            for side in ("source", "destination"):
                if not re.fullmatch(r"[a-zA-Z0-9_.:-]+", path[f"{side}_interface"]):
                    raise ValueError(f"unsafe {side} interface for {client}->{server}")
                if type(path[f"{side}_mtu"]) is not int or path[f"{side}_mtu"] <= 0:
                    raise ValueError(f"invalid {side} MTU for {client}->{server}")
                if not is_number(path[f"{side}_link_mbps"]) or path[f"{side}_link_mbps"] <= 0:
                    raise ValueError(f"invalid {side} link speed for {client}->{server}")


def plan_units(config, inventory, pilot=False):
    repetitions = 1 if pilot else config["repetitions"]
    units = []

    def member(client, server, direction, streams):
        path = inventory["paths"][client][server]
        return {
            "id": f"{client}__{server}__{direction}__p{streams}",
            "client": client,
            "server": server,
            "direction": direction,
            "streams": streams,
            "port": config["ports"][client],
            "source_address": path["source_address"],
            "source_interface": path["source_interface"],
            "destination_address": path["destination_address"],
            "destination_interface": path["destination_interface"],
            "path_link_mbps": min(path["source_link_mbps"], path["destination_link_mbps"]),
        }

    if pilot:
        for direction in config["directions"]:
            for streams in config["isolated_streams"]:
                item = member("anjuna2", "colva2", direction, streams)
                units.append({"id": f"isolated__{item['id']}__r1", "mode": "isolated",
                              "repetition": 1, "members": [item], "attempts": []})
        return units

    isolated = [(client, server, direction, streams)
                for client in CLIENTS for server in SERVERS
                for direction in config["directions"] for streams in config["isolated_streams"]]
    for repetition in range(1, repetitions + 1):
        ordered = list(isolated)
        random.Random(config["order_seed"] + repetition).shuffle(ordered)
        for client, server, direction, streams in ordered:
            item = member(client, server, direction, streams)
            units.append({"id": f"isolated__{item['id']}__r{repetition}", "mode": "isolated",
                          "repetition": repetition, "members": [item], "attempts": []})
    for repetition in range(1, repetitions + 1):
        concurrent = [(client, direction) for client in CLIENTS for direction in config["directions"]]
        random.Random(config["order_seed"] + 100 + repetition).shuffle(concurrent)
        for client, direction in concurrent:
            members = [member(client, server, direction, config["concurrent_streams"])
                       for server in SERVERS]
            units.append({"id": f"fanout1__{client}__{direction}__r{repetition}",
                          "mode": "one_client_four_oss", "repetition": repetition,
                          "members": members, "attempts": []})
    for repetition in range(1, repetitions + 1):
        directions = list(config["directions"])
        random.Random(config["order_seed"] + 200 + repetition).shuffle(directions)
        for direction in directions:
            members = [member(client, server, direction, config["concurrent_streams"])
                       for server in SERVERS for client in CLIENTS]
            units.append({"id": f"fanout2__{direction}__r{repetition}",
                          "mode": "two_clients_four_oss", "repetition": repetition,
                          "members": members, "attempts": []})
    return units


def local_aliases():
    names = {socket.gethostname(), socket.getfqdn()}
    return names | {name.split(".")[0] for name in names}


def host_prefix(host, inventory, config):
    if host in local_aliases():
        return []
    return ["ssh", "-o", "BatchMode=yes", "-o",
            f"ConnectTimeout={config['timeouts']['ssh_connect_seconds']}",
            inventory["hosts"][host]["ssh_host"]]


def command_for_host(host, argv, inventory, config):
    prefix = host_prefix(host, inventory, config)
    return prefix + ([shlex.join(argv)] if prefix else list(argv))


def host_command(host, argv, inventory, config, timeout=None, input_text=None):
    command = command_for_host(host, argv, inventory, config)
    return subprocess.run(command, input=input_text, text=True, capture_output=True,
                          timeout=timeout or config["timeouts"]["command_seconds"], check=False)


def tool_version(output):
    match = re.search(r"iperf\s+3(?:\.\d+)+[^\n]*", output, re.IGNORECASE)
    if not match:
        raise ValueError(f"not an iperf3 version: {output!r}")
    return match.group(0).strip()


def route_matches(route, interface, source):
    selected_source = route.get("prefsrc") or route.get("src") or route.get("from")
    return route.get("dev") == interface and selected_source == source


def preflight(config, inventory):
    coordinator = socket.gethostname().split(".")[0]
    if coordinator != config["coordinator"]:
        raise ValueError(f"run on {config['coordinator']}, not {coordinator}")
    observations = {}
    required_options = ("--server", "--one-off", "--json", "--bind", "--port",
                        "--client", "--parallel", "--omit", "--time", "--reverse")
    for host in CLIENTS + SERVERS:
        script = ("set -u; hostname -s; for tool in iperf3 ip ss timeout python3 setsid ps awk "
                  "date tr grep id timedatectl cat; do command -v \"$tool\"; done; "
                  "iperf3 --version; iperf3 --help")
        result = host_command(host, ["bash", "-c", script], inventory, config)
        if result.returncode:
            raise ValueError(f"host preflight failed on {host}: {result.stderr.strip()}")
        if result.stdout.splitlines()[0].strip() != host:
            raise ValueError(f"SSH destination for {host} reported another hostname")
        if any(option not in result.stdout for option in required_options):
            raise ValueError(f"iperf3 on {host} lacks required options")
        version = tool_version(result.stdout)
        clock = host_command(host, ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
                             inventory, config)
        if clock.returncode or clock.stdout.strip().lower() != "yes":
            raise ValueError(f"live clock synchronization failed on {host}")
        observations[host] = {"iperf3_version": version, "clock_synchronized": True,
                              "interfaces": {}}
    endpoints = {}
    for client in CLIENTS:
        for server in SERVERS:
            path = inventory["paths"][client][server]
            for host, side in ((client, "source"), (server, "destination")):
                key = (host, path[f"{side}_interface"])
                properties = {"mtu": path[f"{side}_mtu"],
                              "link_mbps": path[f"{side}_link_mbps"]}
                if key in endpoints and any(endpoints[key][field] != value
                                            for field, value in properties.items()):
                    raise ValueError(f"inconsistent endpoint properties for {host}:{key[1]}")
                endpoint = endpoints.setdefault(key, {**properties, "addresses": set()})
                endpoint["addresses"].add(path[f"{side}_address"])
    for (host, interface), expected in sorted(endpoints.items()):
        result = host_command(host, ["ip", "-json", "address", "show", "dev", interface],
                              inventory, config)
        links = json.loads(result.stdout) if result.returncode == 0 else []
        addresses = {entry.get("local") for link in links for entry in link.get("addr_info", [])}
        if (len(links) != 1 or links[0].get("ifname") != interface
                or links[0].get("operstate") != "UP" or not expected["addresses"].issubset(addresses)
                or links[0].get("mtu") != expected["mtu"]):
            raise ValueError(f"live endpoint differs on {host}:{interface}")
        speed = host_command(host, ["cat", f"/sys/class/net/{interface}/speed"], inventory, config)
        if speed.returncode or float(speed.stdout.strip()) != expected["link_mbps"]:
            raise ValueError(f"live link speed differs on {host}:{interface}")
        observations[host]["interfaces"][interface] = {
            "addresses": sorted(expected["addresses"]), "mtu": expected["mtu"],
            "link_mbps": expected["link_mbps"], "operstate": "UP",
        }
    for client in CLIENTS:
        for server in SERVERS:
            path = inventory["paths"][client][server]
            checks = ((client, path["destination_address"], path["source_address"]),
                      (server, path["source_address"], path["destination_address"]))
            for host, destination, source in checks:
                result = host_command(host, ["ip", "-json", "route", "get", destination,
                                             "from", source], inventory, config)
                routes = json.loads(result.stdout) if result.returncode == 0 else []
                interface = path["source_interface" if host == client else "destination_interface"]
                if not routes or not route_matches(routes[0], interface, source):
                    raise ValueError(f"live route differs for {host}: {source}->{destination}")
    for server in SERVERS:
        for port in config["ports"].values():
            result = host_command(server, ["ss", "-H", "-ltn", f"sport = :{port}"], inventory, config)
            if result.returncode or result.stdout.strip():
                raise ValueError(f"configured port {port} is not free on {server}")
    return observations


def measure_clock(host, inventory, config):
    started = time.time_ns()
    result = host_command(host, ["date", "+%s%N"], inventory, config)
    ended = time.time_ns()
    if result.returncode or not result.stdout.strip().isdigit():
        raise ValueError(f"cannot measure clock offset on {host}")
    remote = int(result.stdout.strip())
    midpoint = (started + ended) / 2
    return {"offset_seconds": (remote - midpoint) / 1e9,
            "uncertainty_seconds": (ended - started) / 2e9}


def build_commands(member, config):
    server = ["iperf3", "--server", "--one-off", "--json",
              "--bind", member["destination_address"], "--port", str(member["port"])]
    client = ["iperf3", "--client", member["destination_address"],
              "--bind", member["source_address"], "--port", str(member["port"]),
              "--parallel", str(member["streams"]), "--omit", str(config["omit_seconds"]),
              "--time", str(config["duration_seconds"]), "--interval",
              str(config["interval_seconds"]), "--json"]
    if member["direction"] == "oss_to_client":
        client.append("--reverse")
    return server, client


def parse_identity(path):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    required = {"pid", "pgid", "start", "boot", "uid", "launched_epoch_ns"}
    required |= {"supervisor_pid", "supervisor_start", "barrier_pid", "barrier_start",
                 "release_epoch"}
    if not required.issubset(values) or not all(values[key] for key in required):
        raise ValueError(f"invalid process identity: {path}")
    return values


def start_process(host, role, argv, remote_dir, artifact, inventory, config,
                  release_epoch=None):
    make_directory(artifact)
    wrapper_stdout = (artifact / f"{role}_wrapper_stdout").open("xb")
    wrapper_stderr = (artifact / f"{role}_wrapper_stderr").open("xb")
    command = command_for_host(host, [
        "bash", "-s", "--", remote_dir,
        str(config["timeouts"]["remote_watchdog_seconds"]),
        str(release_epoch or 0), *argv,
    ], inventory, config)
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=wrapper_stdout,
                                   stderr=wrapper_stderr, text=True, start_new_session=True)
        process.stdin.write(REMOTE_PROCESS_SCRIPT)
        process.stdin.close()
    except BaseException as original:
        cleanup_error = None
        if "process" in locals():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            saved = {"host": host, "role": role, "argv": argv,
                     "remote_dir": remote_dir}
            try:
                remote_cleanup(saved, inventory, config)
                remove_remote_artifacts(saved, config, inventory)
            except Exception as error:
                cleanup_error = error
        wrapper_stdout.close()
        wrapper_stderr.close()
        if cleanup_error is not None:
            raise RuntimeError(f"process launch failed and cleanup was unresolved: {cleanup_error}") from original
        raise
    return {"host": host, "role": role, "argv": argv, "remote_dir": remote_dir,
            "artifact": artifact, "process": process, "stdout": wrapper_stdout,
            "stderr": wrapper_stderr, "launched_monotonic": time.monotonic()}


def readiness(process, member, inventory, config):
    deadline = time.monotonic() + config["timeouts"]["readiness_seconds"]
    while time.monotonic() < deadline:
        if process["process"].poll() is not None:
            raise ValueError(f"server exited before readiness on {process['host']}")
        result = host_command(process["host"], ["bash", "-s", "--", process["remote_dir"],
            member["destination_address"], str(member["port"])], inventory, config,
            input_text=READINESS_SCRIPT)
        if result.returncode == 0:
            return
        if result.returncode not in (1, 2):
            raise ValueError(f"server identity/readiness failed on {process['host']}: {result.stderr.strip()}")
        time.sleep(0.2)
    raise TimeoutError(f"server readiness exceeded {config['timeouts']['readiness_seconds']}s")


def stop_local_process(process):
    child = process["process"]
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def remote_cleanup(process, inventory, config):
    result = host_command(process["host"], ["bash", "-s", "--", process["remote_dir"]],
                          inventory, config, input_text=CLEANUP_SCRIPT)
    if result.returncode:
        raise ValueError(f"owned process cleanup failed on {process['host']}: {result.stderr.strip()}")
    if process["role"] == "server":
        port = process["argv"][process["argv"].index("--port") + 1]
        result = host_command(process["host"], ["ss", "-H", "-ltn", f"sport = :{port}"],
                              inventory, config)
        if result.returncode or result.stdout.strip():
            raise ValueError(f"server port {port} remains occupied on {process['host']}")


def terminate_owned_process(process, inventory, config):
    errors = []
    try:
        remote_cleanup(process, inventory, config)
    except Exception as error:
        errors.append(str(error))
    try:
        stop_local_process(process)
    except Exception as error:
        errors.append(str(error))
    return errors


def wait_processes(processes, timeout_seconds, run):
    started = time.monotonic()
    while True:
        running = [item for item in processes if item["process"].poll() is None]
        if not running:
            return
        if remaining_seconds(run["deadline"], run["cleanup_seconds"]) <= 0:
            raise BudgetExpired("reservation cleanup window reached")
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"iperf processes exceeded {timeout_seconds}s")
        failed = [item for item in processes if item["process"].poll() not in (None, 0)]
        if failed:
            raise subprocess.CalledProcessError(failed[0]["process"].returncode, failed[0]["argv"])
        time.sleep(0.2)


def finish_process_handles(processes):
    for item in processes:
        for handle in (item["stdout"], item["stderr"]):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()


def copy_process_artifacts(process, destination, inventory, config):
    make_directory(destination)
    host = process["host"]
    if host in local_aliases():
        source = Path(process["remote_dir"])
        for name in PROCESS_FILES:
            path = source / name
            if path.is_file() and not path.is_symlink():
                shutil.copy2(path, destination / name)
    else:
        remote = inventory["hosts"][host]["ssh_host"]
        sources = [f"{remote}:{process['remote_dir']}/{name}" for name in PROCESS_FILES]
        result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o",
                                 f"ConnectTimeout={config['timeouts']['ssh_connect_seconds']}",
                                 *sources, str(destination)],
                                capture_output=True, text=True,
                                timeout=config["timeouts"]["command_seconds"], check=False)
        if result.returncode:
            raise ValueError(f"artifact transfer failed from {host}: {result.stderr.strip()}")
    sync_directory(destination)


def remove_remote_artifacts(process, config, inventory):
    root = Path(config["remote_root"])
    remote = Path(process["remote_dir"])
    if not remote.is_relative_to(root) or len(remote.parts) < len(root.parts) + 5:
        raise ValueError(f"refusing unsafe remote cleanup: {remote}")
    result = host_command(process["host"], ["rm", "-rf", "--", str(remote)], inventory, config)
    if result.returncode:
        raise ValueError(f"remote artifact cleanup failed on {process['host']}: {result.stderr.strip()}")


def numeric(value, label):
    if not is_number(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def validate_iperf_payload(payload, member, config, perspective="client"):
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError(f"iperf3 {perspective} JSON is an error or not an object")
    test = payload.get("start", {}).get("test_start", {})
    if str(test.get("protocol", "")).upper() != "TCP":
        raise ValueError("iperf3 protocol is not TCP")
    if test.get("num_streams") != member["streams"]:
        raise ValueError("iperf3 stream count differs")
    if not math.isclose(float(test.get("omit", -1)), config["omit_seconds"], abs_tol=0.01):
        raise ValueError("iperf3 omit interval differs")
    if not math.isclose(float(test.get("duration", -1)), config["duration_seconds"], abs_tol=0.01):
        raise ValueError("iperf3 requested duration differs")
    expected_reverse = member["direction"] == "oss_to_client"
    if bool(test.get("reverse", False)) != expected_reverse:
        raise ValueError("iperf3 reverse mode differs")
    if perspective == "client":
        connected = payload.get("start", {}).get("connected", [])
        if len(connected) != member["streams"]:
            raise ValueError("iperf3 connected stream count differs")
        for stream in connected:
            if (stream.get("local_host") != member["source_address"]
                    or stream.get("remote_host") != member["destination_address"]
                    or stream.get("remote_port") != member["port"]):
                raise ValueError("iperf3 connected endpoints differ")
    else:
        connected = payload.get("start", {}).get("connected", [])
        if len(connected) != member["streams"]:
            raise ValueError("iperf3 server connected stream count differs")
        for stream in connected:
            if (stream.get("local_host") != member["destination_address"]
                    or stream.get("remote_host") != member["source_address"]
                    or stream.get("local_port") != member["port"]):
                raise ValueError("iperf3 server endpoints differ")
    end = payload.get("end", {})
    sent, received = end.get("sum_sent", {}), end.get("sum_received", {})
    sender_bps = numeric(sent.get("bits_per_second"), "sender throughput")
    receiver_bps = numeric(received.get("bits_per_second"), "receiver throughput")
    sender_bytes = numeric(sent.get("bytes"), "sender bytes")
    receiver_bytes = numeric(received.get("bytes"), "receiver bytes")
    sender_seconds = numeric(sent.get("seconds"), "sender duration")
    receiver_seconds = numeric(received.get("seconds"), "receiver duration")
    tolerance = config["validation"]["duration_tolerance_seconds"]
    if perspective == "client":
        active = (("sender", sender_bps, sender_bytes, sender_seconds),
                  ("receiver", receiver_bps, receiver_bytes, receiver_seconds))
    elif expected_reverse:
        active = (("server sender", sender_bps, sender_bytes, sender_seconds),)
    else:
        active = (("server receiver", receiver_bps, receiver_bytes, receiver_seconds),)
    for label, bits_per_second, byte_count, seconds in active:
        if (bits_per_second <= 0 or byte_count <= 0
                or abs(seconds - config["duration_seconds"]) > tolerance):
            raise ValueError(f"iperf3 {label} throughput, bytes or duration is invalid")
    streams = end.get("streams", [])
    if len(streams) != member["streams"]:
        raise ValueError("iperf3 final stream count differs")
    for stream in streams:
        for direction in ("sender", "receiver"):
            if not isinstance(stream.get(direction), dict):
                raise ValueError("iperf3 per-stream evidence is missing")
            numeric(stream[direction].get("bits_per_second"), "per-stream throughput")
    intervals = payload.get("intervals", [])
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("iperf3 interval evidence is missing")
    omitted_seconds = 0.0
    measured_seconds = 0.0
    for interval in intervals:
        total = interval.get("sum", {})
        seconds = numeric(total.get("seconds"), "interval duration")
        if seconds <= 0:
            raise ValueError("iperf3 interval duration is invalid")
        numeric(total.get("bits_per_second"), "interval throughput")
        if total.get("omitted") is True:
            omitted_seconds += seconds
        else:
            measured_seconds += seconds
    tolerance = config["validation"]["duration_tolerance_seconds"]
    if (abs(omitted_seconds - config["omit_seconds"]) > tolerance
            or abs(measured_seconds - config["duration_seconds"]) > tolerance):
        raise ValueError("iperf3 interval duration coverage differs")
    cpu = end.get("cpu_utilization_percent", {})
    if not isinstance(cpu, dict):
        raise ValueError("iperf3 CPU utilization is missing")
    return {
        "sender_bits_per_second": sender_bps,
        "receiver_bits_per_second": receiver_bps,
        "sender_bytes": sender_bytes,
        "receiver_bytes": receiver_bytes,
        "sender_seconds": sender_seconds,
        "receiver_seconds": receiver_seconds,
        "retransmits": sent.get("retransmits"),
        "client_cpu_percent": cpu.get("host_total") if perspective == "client" else cpu.get("remote_total"),
        "server_cpu_percent": cpu.get("remote_total") if perspective == "client" else cpu.get("host_total"),
    }


def validate_member_artifacts(root, relative, member, config):
    folder = safe_artifact(root, relative)
    if json.loads((folder / "member.json").read_text()) != member:
        raise ValueError("saved member provenance differs")
    expected_commands = dict(zip(("server", "client"), build_commands(member, config)))
    if json.loads((folder / "commands.json").read_text()) != expected_commands:
        raise ValueError("saved command provenance differs")
    summaries = {}
    for role in ("client", "server"):
        path = folder / role
        for name in PROCESS_FILES:
            if not (path / name).is_file() or (path / name).is_symlink():
                raise ValueError(f"missing {role}/{name}")
        if int((path / "exit_status").read_text().strip()) != 0:
            raise ValueError(f"{role} process did not exit successfully")
        parse_identity(path / "identity")
        if shlex.split((path / "argv").read_text()) != expected_commands[role]:
            raise ValueError(f"{role} argv differs from the fixed command")
        payload = json.loads((path / "output.json").read_text())
        summaries[role] = validate_iperf_payload(payload, member, config, role)
    client_bytes = summaries["client"]["receiver_bytes"]
    server_field = "sender_bytes" if member["direction"] == "oss_to_client" else "receiver_bytes"
    server_bytes = summaries["server"][server_field]
    if abs(client_bytes - server_bytes) > max(client_bytes, server_bytes) * 0.01:
        raise ValueError("client/server delivered-byte evidence disagrees")
    return summaries["client"]


def capture_telemetry(host, interface, phase, folder, inventory, config):
    result = host_command(host, ["ip", "-statistics", "-statistics", "-json",
                                 "link", "show", "dev", interface], inventory, config)
    prefix = f"telemetry-{phase}-{host}-{interface}"
    atomic_text(folder / f"{prefix}.json", result.stdout)
    atomic_text(folder / f"{prefix}.stderr", result.stderr)
    if result.returncode:
        raise ValueError(f"telemetry failed on {host}")
    json.loads(result.stdout)


def run_markdown(manifest):
    config = manifest["config"]
    counts = {mode: sum(unit["mode"] == mode for unit in manifest["units"])
              for mode in ("isolated", "one_client_four_oss", "two_clients_four_oss")}
    return "\n".join([
        "# iperf3 network transport run", "",
        "> Generated from `manifest.json`; native endpoint JSON remains authoritative.", "",
        "## Identity", "", f"- Run ID: `{manifest['run_id']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Protocol version: `{config['protocol_version']}`",
        f"- Inventory fingerprint: `{manifest['inventory_fingerprint']}`", "",
        "## Protocol", "", f"- Transport: `{config['transport']}`",
        f"- Timing: {config['omit_seconds']} s omitted + {config['duration_seconds']} s measured",
        f"- Isolated streams: `{config['isolated_streams']}`",
        f"- Concurrent streams per path: `{config['concurrent_streams']}`",
        f"- Repetitions: **{1 if manifest['mode'] == 'pilot' else config['repetitions']}**", "",
        "## Matrix", "", f"- Isolated units: **{counts['isolated']}**",
        f"- One-client fan-out epochs: **{counts['one_client_four_oss']}**",
        f"- Two-client fan-out epochs: **{counts['two_clients_four_oss']}**",
        f"- Path sessions: **{sum(len(unit['members']) for unit in manifest['units'])}**", "",
    ])


def save(run):
    if run["deadline"].exists():
        try:
            run["manifest"]["sessions"][-1]["deadline"] = read_json(run["deadline"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    atomic_json(run["root"] / "manifest.json", run["manifest"])
    atomic_text(run["root"] / "RUN.md", run_markdown(run["manifest"]))


def fingerprints(config, inventory, observations):
    inventory_fingerprint = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    scientific = {key: value for key, value in config.items()
                  if key not in {"planning", "timeouts"}}
    scientific.update(inventory=inventory,
                      tool_versions={host: item["iperf3_version"]
                                     for host, item in observations.items()})
    return (hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest(),
            inventory_fingerprint)


def revalidate_attempt_evidence(root, unit, attempt, config):
    if len(attempt.get("members", [])) != len(unit["members"]):
        raise ValueError("saved attempt member count differs")
    folder = safe_artifact(root, attempt["artifacts"])
    endpoints = {(member["client"], member["source_interface"])
                 for member in unit["members"]}
    endpoints |= {(member["server"], member["destination_interface"])
                  for member in unit["members"]}
    for host, interface in endpoints:
        for phase in ("before", "after"):
            telemetry = folder / f"telemetry-{phase}-{host}-{interface}.json"
            if not telemetry.is_file() or telemetry.is_symlink():
                raise ValueError(f"missing telemetry for {host}:{interface}")
            payload = json.loads(telemetry.read_text())
            if (not isinstance(payload, list) or len(payload) != 1
                    or payload[0].get("ifname") != interface):
                raise ValueError(f"invalid telemetry for {host}:{interface}")
    starts, summaries = [], []
    clocks = attempt["clock_observations"]
    for saved, expected in zip(attempt["members"], unit["members"]):
        if saved["member"] != expected:
            raise ValueError("saved attempt member differs")
        summary = validate_member_artifacts(root, saved["artifacts"], expected, config)
        identity_path = safe_artifact(root, saved["artifacts"]) / "client" / "identity"
        identity = parse_identity(identity_path)
        raw_start = int(identity["launched_epoch_ns"]) / 1e9
        starts.append(raw_start - clocks[expected["client"]]["offset_seconds"])
        summaries.append(summary)
    launch_skew = max(starts) - min(starts)
    release_lateness = max(starts) - attempt["release_epoch"]
    if launch_skew > config["validation"]["maximum_launch_skew_seconds"]:
        raise ValueError("saved attempt launch skew exceeds limit")
    if release_lateness > config["validation"]["maximum_release_lateness_seconds"]:
        raise ValueError("saved attempt release lateness exceeds limit")
    return summaries, launch_skew, release_lateness


def load_run(args, config, inventory, observations):
    manifest_path = args.results_dir / "manifest.json"
    if manifest_path.exists() != args.resume:
        raise ValueError("use --resume for an existing manifest; a new run needs a new directory")
    old = read_json(manifest_path) if args.resume else None
    mode = old.get("mode", "full") if old else ("pilot" if args.pilot else "full")
    if old and args.pilot and mode != "pilot":
        raise ValueError("--pilot cannot change an existing full run")
    fingerprint, inventory_fingerprint = fingerprints(config, inventory, observations)
    if old and (old["fingerprint"] != fingerprint
                or old["inventory_fingerprint"] != inventory_fingerprint):
        raise ValueError("resume rejected: configuration, inventory or tool versions changed")
    manifest = old or {
        "run_id": uuid.uuid4().hex,
        "mode": mode,
        "fingerprint": fingerprint,
        "inventory_fingerprint": inventory_fingerprint,
        "config": config,
        "inventory": inventory,
        "tool_observations": observations,
        "coordinator": config["coordinator"],
        "units": plan_units(config, inventory, mode == "pilot"),
        "sessions": [],
    }
    for unit in manifest["units"]:
        for attempt in unit["attempts"]:
            if attempt["state"] == "running":
                attempt.update(state="interrupted", error="previous session ended without checkpoint")
            if attempt["state"] == "completed":
                try:
                    for item in attempt["members"]:
                        validate_member_artifacts(args.results_dir, item["artifacts"], item["member"], config)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    attempt.update(state="failed", error=f"invalid saved evidence: {error}")
        if (unit["attempts"] and unit["attempts"][-1]["state"] == "failed"
                and unit["attempts"][-1].get("cleanup") == "completed"):
            attempt = unit["attempts"][-1]
            try:
                summaries, skew, lateness = revalidate_attempt_evidence(
                    args.results_dir, unit, attempt, config)
                previous_error = attempt.pop("error", None)
                for saved, summary in zip(attempt["members"], summaries):
                    saved["summary"] = summary
                attempt.update(state="completed", launch_skew_seconds=skew,
                               maximum_release_lateness_seconds=lateness,
                               revalidated_at=now(), revalidated_from_error=previous_error)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
    for session in manifest["sessions"]:
        if session["outcome"] == "running":
            session["outcome"] = "interrupted"
    manifest["sessions"].append({
        "id": len(manifest["sessions"]) + 1,
        "started_at": now(),
        "outcome": "running",
        "kernel": platform.release(),
        "allocation_id": os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID"),
    })
    return {"root": args.results_dir, "deadline": args.results_dir / "deadline.json",
            "cleanup_seconds": args.cleanup_buffer, "config": config,
            "inventory": inventory, "manifest": manifest}


def completed(unit):
    return bool(unit["attempts"] and unit["attempts"][-1]["state"] == "completed"
                and unit["attempts"][-1].get("cleanup") == "completed")


def recover_attempt_cleanup(run, attempt):
    """Clean one durably journaled attempt without repeating network traffic."""
    if attempt.get("cleanup") == "completed":
        return
    errors = []
    for saved in attempt.get("members", []):
        member = saved["member"]
        for role, host, key in (
                ("server", member["server"], "server_remote"),
                ("client", member["client"], "client_remote")):
            command = build_commands(member, run["config"])[0 if role == "server" else 1]
            process = {"host": host, "role": role, "argv": command,
                       "remote_dir": saved[key]}
            try:
                remote_cleanup(process, run["inventory"], run["config"])
                remove_remote_artifacts(process, run["config"], run["inventory"])
            except Exception as error:
                errors.append(str(error))
    if errors:
        attempt.update(cleanup="failed", cleanup_errors=errors)
        save(run)
        raise ValueError("saved attempt cleanup failed: " + "; ".join(errors))
    attempt["cleanup"] = "completed"
    attempt.pop("cleanup_errors", None)
    save(run)


def recover_existing_attempts(args, config, inventory):
    """Reclaim abandoned owned processes before port-availability preflight."""
    manifest = read_json(args.results_dir / "manifest.json")
    old_config, old_inventory = manifest["config"], manifest["inventory"]
    validate_config(old_config)
    validate_inventory(old_inventory)
    scientific = lambda item: {key: value for key, value in item.items()
                               if key not in {"planning", "timeouts"}}
    if scientific(old_config) != scientific(config) or old_inventory != inventory:
        raise ValueError("resume rejected before cleanup: scientific configuration or inventory changed")
    run = {"root": args.results_dir, "deadline": args.results_dir / "deadline.json",
           "cleanup_seconds": args.cleanup_buffer, "config": old_config,
           "inventory": old_inventory, "manifest": manifest}
    for unit in manifest["units"]:
        for attempt in unit.get("attempts", []):
            if attempt.get("state") == "running":
                attempt.update(state="interrupted", error="previous session ended without checkpoint")
            recover_attempt_cleanup(run, attempt)
    for session in manifest.get("sessions", []):
        if session.get("outcome") == "running":
            session["outcome"] = "interrupted"
    save(run)


def execute_unit(run, unit):
    config, inventory, manifest = run["config"], run["inventory"], run["manifest"]
    estimate = config["planning"]["isolated_seconds" if unit["mode"] == "isolated" else "concurrent_epoch_seconds"]
    admission = estimate + config["planning"]["overhead_seconds"]
    if remaining_seconds(run["deadline"], run["cleanup_seconds"]) < admission:
        raise BudgetExpired(f"not enough time for {unit['id']}")
    number = len(unit["attempts"]) + 1
    relative = f"raw/{unit['id']}/attempt-{number}"
    folder = safe_artifact(run["root"], relative)
    if folder.exists():
        raise ValueError(f"refusing to overwrite attempt artifacts: {folder}")
    make_directory(folder)
    attempt = {"id": number, "state": "running", "started_at": now(),
               "session": manifest["sessions"][-1]["id"], "artifacts": relative,
               "estimated_wall_seconds": estimate, "cleanup": "pending", "members": []}
    unit["attempts"].append(attempt)
    save(run)
    processes = []
    copied = set()
    started = time.monotonic()
    try:
        endpoints = sorted({(member["client"], member["source_interface"])
                            for member in unit["members"]}
                           | {(member["server"], member["destination_interface"])
                              for member in unit["members"]})
        for host, interface in endpoints:
            capture_telemetry(host, interface, "before", folder, inventory, config)
        for index, member in enumerate(unit["members"]):
            member_folder = folder / member["id"]
            if member_folder.exists():
                raise ValueError(f"duplicate member artifact folder: {member['id']}")
            make_directory(member_folder)
            atomic_json(member_folder / "member.json", member)
            server_command, client_command = build_commands(member, config)
            atomic_json(member_folder / "commands.json", {"server": server_command, "client": client_command})
            remote_base = (f"{config['remote_root']}/{manifest['run_id']}/{unit['id']}/"
                           f"attempt-{number}/{index}-{member['id']}")
            attempt["members"].append({"member": member,
                                       "artifacts": f"{relative}/{member['id']}",
                                       "server_remote": f"{remote_base}/server",
                                       "client_remote": f"{remote_base}/client"})
            save(run)
            server_process = start_process(member["server"], "server", server_command,
                                           f"{remote_base}/server", member_folder,
                                           inventory, config)
            processes.append(server_process)
        save(run)
        with ThreadPoolExecutor(max_workers=len(unit["members"])) as executor:
            checks = [executor.submit(readiness, process, member, inventory, config)
                      for member, process in zip(unit["members"], processes)]
            for check in checks:
                check.result()
        client_processes = []
        clock_observations = {
            host: measure_clock(host, inventory, config)
            for host in sorted({member["client"] for member in unit["members"]})
        }
        for host, observation in clock_observations.items():
            if abs(observation["offset_seconds"]) > config["validation"]["maximum_clock_offset_seconds"]:
                raise ValueError(f"clock offset exceeds limit on {host}")
            if observation["uncertainty_seconds"] > config["validation"]["maximum_clock_uncertainty_seconds"]:
                raise ValueError(f"clock uncertainty exceeds limit on {host}")
        release_epoch = time.time() + config["barrier_lead_seconds"]
        attempt["release_epoch"] = release_epoch
        attempt["clock_observations"] = clock_observations
        for index, member in enumerate(unit["members"]):
            _, client_command = build_commands(member, config)
            member_folder = folder / member["id"]
            remote_dir = attempt["members"][index]["client_remote"]
            client_process = start_process(member["client"], "client", client_command,
                                           remote_dir, member_folder, inventory, config,
                                           release_epoch=release_epoch)
            client_processes.append(client_process)
            processes.append(client_process)
        save(run)
        wait_processes(client_processes, config["timeouts"]["client_seconds"], run)
        wait_processes(processes[:len(unit["members"])],
                       config["timeouts"]["server_exit_seconds"], run)
        finish_process_handles(processes)
        for host, interface in endpoints:
            capture_telemetry(host, interface, "after", folder, inventory, config)
        client_starts = []
        for index, member in enumerate(unit["members"]):
            member_folder = folder / member["id"]
            server_process = processes[index]
            client_process = client_processes[index]
            copy_process_artifacts(server_process, member_folder / "server", inventory, config)
            copied.add(id(server_process))
            copy_process_artifacts(client_process, member_folder / "client", inventory, config)
            copied.add(id(client_process))
            identity = parse_identity(member_folder / "client" / "identity")
            raw_start = int(identity["launched_epoch_ns"]) / 1e9
            adjusted_start = raw_start - clock_observations[member["client"]]["offset_seconds"]
            client_starts.append(adjusted_start)
            summary = validate_member_artifacts(run["root"], f"{relative}/{member['id']}", member, config)
            attempt["members"][index]["summary"] = summary
        launch_skew = max(client_starts) - min(client_starts)
        attempt["launch_skew_seconds"] = launch_skew
        if launch_skew > config["validation"]["maximum_launch_skew_seconds"]:
            raise ValueError(f"client launch skew {launch_skew:.3f}s exceeds limit")
        release_lateness = max(client_starts) - release_epoch
        attempt["maximum_release_lateness_seconds"] = release_lateness
        if release_lateness > config["validation"]["maximum_release_lateness_seconds"]:
            raise ValueError(f"client release lateness {release_lateness:.3f}s exceeds limit")
        attempt["state"] = "completed"
    except BaseException as error:
        attempt.update(state="interrupted" if isinstance(error, (BudgetExpired, KeyboardInterrupt)) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        with ThreadPoolExecutor(max_workers=max(1, len(processes))) as executor:
            cleanups = [executor.submit(terminate_owned_process, process, inventory, config)
                        for process in reversed(processes)]
            for cleanup in cleanups:
                cleanup_errors.extend(cleanup.result())
        for process in processes:
            if not process["stdout"].closed:
                try:
                    process["stdout"].close()
                    process["stderr"].close()
                except OSError:
                    pass
            if id(process) not in copied:
                try:
                    destination = process["artifact"] / process["role"]
                    copy_process_artifacts(process, destination, inventory, config)
                    copied.add(id(process))
                except Exception as error:
                    cleanup_errors.append(str(error))
        for process in processes:
            if id(process) in copied:
                try:
                    remove_remote_artifacts(process, config, inventory)
                except Exception as error:
                    cleanup_errors.append(str(error))
        if cleanup_errors:
            attempt["cleanup_errors"] = cleanup_errors
            attempt["cleanup"] = "failed"
        else:
            attempt["cleanup"] = "completed"
            attempt.pop("cleanup_errors", None)
        attempt.update(ended_at=now(), wall_seconds=time.monotonic() - started)
        save(run)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise ValueError("attempt cleanup failed: " + "; ".join(cleanup_errors))


def main(argv=None):
    args = parse_args(argv)
    run = None
    session_started = time.monotonic()
    try:
        if args.extend_deadline is not None:
            set_deadline(args.results_dir / "deadline.json", extend=args.extend_deadline)
            print("Reservation deadline extended.")
            return 0
        config = read_json(args.config)
        inventory = read_json(args.inventory)
        validate_config(config)
        validate_inventory(inventory)
        if not args.resume and args.results_dir.exists() and any(
                item.name not in {".lock", "deadline.json"}
                for item in args.results_dir.iterdir()):
            raise ValueError("a new run needs an empty results directory")
        if args.resume and not args.results_dir.is_dir():
            raise ValueError("resume directory does not exist")
        make_directory(args.results_dir)
        with (args.results_dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            set_deadline(args.results_dir / "deadline.json",
                         time_limit=args.time_limit, deadline=args.deadline)
            if args.resume:
                recover_existing_attempts(args, config, inventory)
            observations = preflight(config, inventory)
            run = load_run(args, config, inventory, observations)
            save(run)
            try:
                for unit in run["manifest"]["units"]:
                    for attempt in unit["attempts"]:
                        recover_attempt_cleanup(run, attempt)
                    if not completed(unit):
                        print(f"{unit['id']}: {len(unit['members'])} path session(s)", flush=True)
                        execute_unit(run, unit)
                outcome, code = "completed", 0
            except BudgetExpired as error:
                outcome, code = "budget_stop", 0
                run["manifest"]["sessions"][-1]["message"] = str(error)
            except KeyboardInterrupt:
                outcome, code = "interrupted", 130
            except Exception as error:
                outcome, code = "failed", 1
                run["manifest"]["sessions"][-1]["error"] = f"{type(error).__name__}: {error}"
                print(f"Stopped: {error}", file=sys.stderr)
            run["manifest"]["sessions"][-1].update(
                outcome=outcome, ended_at=now(), wall_seconds=time.monotonic() - session_started)
            save(run)
            print(f"{outcome}: {args.results_dir / 'manifest.json'}")
            return code
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError) as error:
        print(f"Cannot run: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
