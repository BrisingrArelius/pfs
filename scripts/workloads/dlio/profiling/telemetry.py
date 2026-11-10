"""Passive per-host samplers; no per-sample SSH and no generated I/O load."""

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


INVENTORY = r'''
import json, pathlib, subprocess
text = pathlib.Path('/etc/beegfs/beegfs-storage.conf').read_text()
paths = next(line.split('=',1)[1] for line in text.splitlines()
             if line.strip().startswith('storeStorageDirectory') and '=' in line)
result = {}
for raw in paths.split(','):
    if not raw.strip(): continue
    path = pathlib.Path(raw.strip())
    target = int((path / 'targetNumID').read_text())
    source = subprocess.check_output(['findmnt','-n','-o','SOURCE','--target',str(path)],text=True).strip()
    device = pathlib.Path(source).resolve().name
    block = pathlib.Path('/sys/class/block') / device
    if not block.exists(): raise RuntimeError('Not a block device: ' + source)
    if (block / 'partition').exists(): device = block.resolve().parent.name
    result[target] = {'directory':str(path),'source':source,'device':device}
print(json.dumps(result))
'''

SAMPLER = r'''
import json, os, pathlib, select, sys, time
cfg = json.loads(sys.argv[1]); interval = float(sys.argv[2]); case = cfg.get('case')
def sample():
    row = {'wall_s':time.time(),'monotonic_s':time.monotonic(),'devices':{},'interfaces':{},'memory':{},'processes':[]}
    for device in cfg['devices']:
        fields = list(map(int,(pathlib.Path('/sys/class/block')/device/'stat').read_text().split()))
        row['devices'][device] = {'read_bytes':fields[2]*512,'write_bytes':fields[6]*512}
    for interface in cfg['interfaces']:
        root = pathlib.Path('/sys/class/net')/interface/'statistics'
        row['interfaces'][interface] = {key:int((root/(key+'_bytes')).read_text()) for key in ('rx','tx')}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':',1)
        if key in ('MemAvailable','Cached','Dirty','Writeback'): row['memory'][key] = int(value.split()[0])*1024
    if case:
        known = {}
        for entry in (pathlib.Path(case)/'anchors').glob('pid-*.json'):
            try:
                data=json.loads(entry.read_text()); known[data['pid']]=data
            except (OSError,ValueError): pass
        process_map = {}
        for entry in pathlib.Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try:
                text=(entry/'stat').read_text(); fields=text[text.rfind(')')+2:].split()
                process_map[int(entry.name)]={'ppid':int(fields[1]),'cpu_ticks':int(fields[11])+int(fields[12]),'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
            except (OSError,ValueError,IndexError): continue
        selected=set(known)
        while True:
            children={pid for pid,data in process_map.items() if data['ppid'] in selected}
            if children <= selected: break
            selected.update(children)
        for pid in selected & process_map.keys():
            row['processes'].append({'pid':pid,'instrumented':pid in known,**process_map[pid]})
        row['clock_ticks_per_second']=os.sysconf('SC_CLK_TCK')
    return row
while True:
    print(json.dumps(sample()),flush=True)
    ready,_,_=select.select([sys.stdin],[],[],interval)
    if ready: break
print(json.dumps(sample()),flush=True)
'''


def remote(execute, host, arguments):
    return execute(["ssh", host, shlex.join(arguments)], check=True,
                   capture_output=True, text=True).stdout


def route(execute, destination):
    return json.loads(execute(["ip", "-j", "route", "get", socket.gethostbyname(destination)],
                              check=True, capture_output=True, text=True).stdout)[0]


def discover(execute):
    """Live mappings, before the final cache preparation."""
    hosts = {}
    interfaces = set()
    for host in ("colva1", "colva2", "colva3", "colva4"):
        client_route = route(execute, host)
        interfaces.add(client_route["dev"])
        source = client_route.get("prefsrc", client_route.get("src"))
        if not source:
            raise RuntimeError(f"No client source address in route to {host}")
        server_route = json.loads(remote(execute, host, ["ip", "-j", "route", "get", source]))[0]
        targets = json.loads(remote(execute, host, ["sudo", "-n", "python3", "-c", INVENTORY]))
        hosts[host] = {"targets": targets, "interfaces": [server_route["dev"]],
                       "route": server_route, "client_route": client_route}
    hosts["client"] = {"interfaces": sorted(interfaces), "targets": {},
                       "hostname": socket.gethostname()}
    return hosts


class Telemetry:
    def __init__(self, case, hosts, targets, execute, interval=1):
        self.case, self.hosts, self.targets = Path(case), hosts, set(targets)
        self.execute, self.interval, self.jobs = execute, interval, []
        self.clocks = {}

    def clock(self, host):
        if host == "client":
            return {"offset_s": 0, "uncertainty_s": 0}
        before = time.time()
        value = float(remote(self.execute, host, ["python3", "-c", "import time; print(time.time())"]))
        after = time.time()
        return {"offset_s": value - (before+after)/2, "uncertainty_s": (after-before)/2}

    def start(self):
        folder = self.case / "telemetry"
        folder.mkdir()
        seen = set()
        try:
            for host, info in self.hosts.items():
                selected = {target: mapping for target, mapping in info["targets"].items()
                            if int(target) in self.targets}
                seen.update(map(int, selected))
                devices = sorted({mapping["device"] for mapping in selected.values()})
                if host != "client" and not devices:
                    continue
                config = {"devices": devices, "interfaces": info["interfaces"],
                          "case": str(self.case) if host == "client" else None}
                self.clocks[host] = {"before": self.clock(host), "config": config,
                                     "target_mapping": selected}
                arguments = [sys.executable if host == "client" else "python3", "-u", "-c", SAMPLER,
                             json.dumps(config), str(self.interval)]
                command = arguments if host == "client" else ["ssh", host, shlex.join(arguments)]
                log = (folder / f"{host}.jsonl").open("w")
                errors = (folder / f"{host}.stderr").open("w")
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=errors,
                                           text=True, start_new_session=True)
                self.jobs.append((host, process, log, errors))
            if seen != self.targets:
                raise RuntimeError(f"Missing telemetry target mappings: {self.targets-seen}")
            time.sleep(min(self.interval, 1))
            self.check()
        except BaseException:
            self.stop()
            raise

    def check(self):
        for host, process, log, _ in self.jobs:
            if process.poll() is not None:
                raise RuntimeError(f"{host} telemetry exited early; inspect {log.name}")

    def stop(self):
        errors = []
        for host, process, log, stderr in self.jobs:
            try:
                process.communicate(input="stop\n", timeout=10)
                if process.returncode:
                    errors.append(f"{host}: sampler exit {process.returncode}")
            except (subprocess.TimeoutExpired, BrokenPipeError):
                import signal
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                errors.append(f"{host}: sampler did not stop cleanly")
            finally:
                log.close()
                stderr.close()
            try:
                self.clocks[host]["after"] = self.clock(host)
            except Exception as error:
                errors.append(f"{host}: closing clock anchor unavailable: {error}")
        if self.jobs:
            (self.case / "telemetry/clocks.json").write_text(json.dumps(self.clocks, indent=2)+"\n")
            (self.case / "telemetry/status.json").write_text(json.dumps({"errors": errors}, indent=2)+"\n")
        self.jobs = []
        return errors
