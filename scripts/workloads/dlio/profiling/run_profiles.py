#!/usr/bin/env python3
"""Profile U-Net/CosmoFlow once on normal 1/3/6 and throttled 14/16/18."""

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DLIO = HERE.parent
sys.path.insert(0, str(DLIO / "throttled_ssd"))
import pool_common as shared
from configure_pool import apply_limits, clear_limits, normal_members, restore, transition, verify_normal
from analyze_profiles import analyze_case, dump, table
from telemetry import Telemetry, discover


def duration(value):
    if re.fullmatch(r"\d+(\.\d+)?", value):
        seconds = float(value)
    else:
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", value)
        if not match or not any(match.groups()):
            raise argparse.ArgumentTypeError("Use seconds, 6h, 90m or 1h30m")
        seconds = sum(float(x or 0)*scale for x,scale in zip(match.groups(), (3600,60,1)))
    if seconds <= 0:
        raise argparse.ArgumentTypeError("Duration must be positive")
    return seconds


def matrix(workload):
    names = ("cosmoflow", "unet") if workload == "both" else (workload,)
    return [(name, suite, pool) for name in names for suite, pools in
            (("normal", (1,3,6)), ("throttled", (14,16,18))) for pool in pools]


def settings(profile):
    return {"dataset": {"num_files_train": profile.files, "num_samples_per_file": 1},
            "reader": {"batch_size": profile.batch_size, "read_threads": profile.reader_threads,
                       "file_shuffle": "seed", "sample_shuffle": "seed", "prefetch_size": 2},
            "train": {"epochs": 5, "computation_time": 0},
            "workflow": {"generate_data": False, "train": True, "checkpoint": True},
            "checkpoint": {"type": "all_ranks", "model_size": profile.model_bytes,
                           "checkpoint_after_epoch": 2, "epochs_between_checkpoints": 2,
                           "num_layers": 0, "layer_parameters": []}}


def build_command(name, dataset, checkpoints, raw):
    profile = shared.WORKLOADS[name]
    command = ["mpirun", "-np", "8", "dlio_benchmark", f"workload={profile.profile}"]
    values = settings(profile)
    # num_samples_per_file is checked against native configuration, not changed.
    for section, fields in values.items():
        for key, value in fields.items():
            if key == "num_samples_per_file":
                continue
            command.append(f"++workload.{section}.{key}={value}")
    command.extend((f"++workload.dataset.data_folder={dataset}",
                    f"++workload.checkpoint.checkpoint_folder={checkpoints}",
                    f"++workload.checkpoint.checkpoint_mechanism_classname={shared.CLASSES[name]}",
                    f"hydra.run.dir={raw}"))
    return command


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_runtime(explicit=None):
    """Reuse the same installed paths/environment as the older Darshan jobs."""
    selected = explicit or os.environ.get("PFS_DARSHAN_LIBRARY") or os.environ.get("DARSHAN_RUNTIME_LIB")
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Selected existing Darshan runtime not found: {path}")
        return path
    candidates = [Path.home()/"darshan-nompi/lib/libdarshan.so",
                  Path("/opt/darshan-nompi/lib/libdarshan.so"),
                  Path.home()/".local/darshan-nonmpi/lib/libdarshan.so",
                  Path("/usr/local/lib/libdarshan.so"),
                  Path("/usr/local/lib/libdarshan.so.0.0.0"),
                  Path("/usr/lib/libdarshan.so"),Path("/usr/lib64/libdarshan.so")]
    candidates.extend(sorted(Path.home().glob("darshan*/lib/libdarshan.so")))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("Existing Darshan runtime not located; use --darshan-library with the path from your previous Darshan job")


def mpi_compatibility(linked, execute):
    """Permit an installed MPI build only with the active MPI library."""
    code = ("import json,re; from pathlib import Path; from mpi4py import MPI; "
            "paths={str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines() "
            "if re.search(r'/lib(?:mpi|mpich)[.]so',line)}; "
            "print(json.dumps({'version':MPI.Get_library_version(),'libraries':sorted(paths)}))")
    active = json.loads(execute([sys.executable,"-c",code],env=clean_environment(),
                                check=True,capture_output=True,text=True).stdout)
    required = {str(Path(path).resolve()) for path in re.findall(
        r"^\s*lib(?:mpi|mpich)[.]so\S*\s+=>\s+(/\S+)",linked,re.M)}
    if not required <= set(active["libraries"]):
        raise RuntimeError("This existing Darshan library links a different MPI than mpi4py; "
                           "select the compatible installed runtime used by your previous jobs "
                           "with DARSHAN_RUNTIME_LIB or --darshan-library. No software changes were made.")
    return {**active,"darshan_mpi_libraries":sorted(required)}


def source_identity(library):
    version = importlib.metadata.version("dlio-benchmark")
    if version != "2.0.0":
        raise RuntimeError("DLIO 2.0.0 is required")
    paths = [DLIO/"normal/dlio_pool_profiles.py", DLIO/"normal/run_unet3d_pools.py",
             DLIO/"normal/run_cosmoflow_pools.py", HERE/"hooks/sitecustomize.py",
             HERE/"run_profiles.py", DLIO/"throttled_ssd/pool_common.py"]
    return {"sources": {str(path.relative_to(DLIO)): sha(path) for path in paths},
            "library": str(library), "library_sha256": sha(library),
            "mpirun": str(Path(shutil.which("mpirun") or "mpirun").resolve()),
            "dlio_version": version}


def instrumentation(case, dataset, checkpoints, library, dxt_mib, module_mib):
    for directory in ("anchors", "darshan", "raw"):
        (case/directory).mkdir()
    # Paths in Darshan configuration are whitespace-delimited EREs.
    if any(char.isspace() for path in (case, dataset, checkpoints) for char in str(path)):
        raise ValueError("Profiling paths must not contain whitespace")
    config = case / "darshan.conf"
    config.write_text(f"MOD_ENABLE DXT_POSIX\nMODMEM {module_mib}\nNAMEMEM 8\n"
        f"MAX_RECORDS 16384 POSIX,STDIO\nMAX_RECORDS {dxt_mib*1024} DXT_POSIX\n"
        "DISABLE_SHARED_REDUCTION\nNAME_EXCLUDE .* *\n"
        f"NAME_INCLUDE ^{re.escape(str(dataset))}/ POSIX,STDIO,DXT_POSIX\n"
        f"NAME_INCLUDE ^{re.escape(str(checkpoints))}/ POSIX,STDIO,DXT_POSIX\n"
        f"NAME_INCLUDE ^{re.escape(str(case/'anchors'))}/.*[.]anchor-(start|end)$ POSIX,DXT_POSIX\n")
    return {"LD_PRELOAD": str(library), "DARSHAN_ENABLE_NONMPI": "1",
            "DARSHAN_LOG_DIR_PATH":str(case/"darshan"), "DARSHAN_LOGPATH":str(case/"darshan"),
            "DXT_ENABLE_IO_TRACE": "1", "DARSHAN_CONFIG_PATH": str(config),
            "DARSHAN_DISABLE_SHARED_REDUCTION": "1", "DARSHAN_DUMP_CONFIG": "1",
            "PFS_DXT_CASE": str(case),
            "PYTHONPATH": os.pathsep.join((str(HERE/"hooks"), str(DLIO), str(DLIO/"normal"),
                                           os.environ.get("PYTHONPATH", "")))}


def launcher(base_command, environment):
    # Instrument only rank executables, never mpirun/ssh/controller copies.
    return [*base_command[:3], "env", *(f"{key}={value}" for key,value in environment.items()), *base_command[3:]]


def clean_environment():
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Unset LD_PRELOAD on the controller; the runner injects it only into ranks")
    return {key:value for key,value in os.environ.items()
            if not key.startswith(("DARSHAN_", "DXT_", "PFS_DXT_"))}


def owned_processes(root):
    """Identify this study's ranks/readers, even if Hydra orphaned them."""
    found = []
    root = str(root)
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            environment = (path/"environ").read_bytes().split(b"\0")
            value = next((item.split(b"=",1)[1].decode() for item in environment if item.startswith(b"PFS_DXT_CASE=")), "")
            stat = (path/"stat").read_text()
            if (value == root or value.startswith(root+"/")) and stat[stat.rfind(")")+2:].split()[0] != "Z":
                found.append(int(path.name))
        except (OSError,ValueError,UnicodeError):
            continue
    return found


def terminate_owned(root):
    """Never move targets while this study still has live reader processes."""
    for sig in (signal.SIGTERM,signal.SIGKILL):
        pids = owned_processes(root)
        if not pids:
            return
        for pid in pids:
            try:
                # Bind the signal to this process instance where supported.
                descriptor = os.pidfd_open(pid)
                try:
                    signal.pidfd_send_signal(descriptor,sig)
                finally:
                    os.close(descriptor)
            except ProcessLookupError:
                pass
        until = time.monotonic()+2
        while time.monotonic()<until and owned_processes(root):
            time.sleep(.05)
    if owned_processes(root):
        raise RuntimeError(f"Owned readers still alive; pool moves withheld: {owned_processes(root)}")


def preflight(output, args):
    for tool in ("mpirun", "darshan-parser", "darshan-dxt-parser"):
        if not shutil.which(tool):
            raise RuntimeError(f"Missing executable: {tool}")
    if not args.darshan_library.is_file():
        raise RuntimeError(f"No Darshan runtime at {args.darshan_library}")
    linked = shared.execute(["ldd", str(args.darshan_library)], check=True, capture_output=True, text=True).stdout
    if "not found" in linked:
        raise RuntimeError(f"Unresolved Darshan dependencies:\n{linked}")
    mpi = mpi_compatibility(linked,shared.execute)
    output_fs = shared.execute(["findmnt","-n","-o","FSTYPE","--target",str(output)],
                               check=True,capture_output=True,text=True).stdout.strip()
    if output_fs in ("beegfs","nfs","nfs4","lustre","cifs","ceph","gpfs","fuse.sshfs"):
        raise RuntimeError("Use --output on client-local storage; network trace writes would contaminate traffic evidence")
    probe = output / ("preflight-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    probe.mkdir()
    data = probe / "smoke-data"
    data.mkdir()
    env = instrumentation(probe, data, data, args.darshan_library, args.dxt_mib, args.module_mib)
    dump(probe/"case.json", {"smoke": True, "dataset": str(data), "checkpoints": str(data)})
    command = launcher(["mpirun", "-np", "8", sys.executable, str(HERE/"smoke.py")], env)
    old_deadline = shared.DEADLINE
    shared.DEADLINE = min(old_deadline or float("inf"), time.monotonic()+120)
    try:
        with (probe/"console.log").open("w") as file:
            shared.execute(command, env=clean_environment(), stdout=file, stderr=subprocess.STDOUT, check=True)
        analyze_case(probe, execute=shared.execute)
    except shared.DeadlineReached as error:
        if old_deadline is None or time.monotonic() < old_deadline:
            raise RuntimeError(f"Existing Darshan/MPI preflight timed out; no full workload or pool move started. Inspect {probe/'console.log'} and select a compatible already-installed runtime.") from error
        raise
    finally:
        terminate_owned(probe)
        shared.DEADLINE = old_deadline
    versions = {}
    for tool in ("mpirun","darshan-parser","darshan-dxt-parser"):
        # Some installed Darshan utilities reject --version. Actual log parsing
        # above is the compatibility check; provenance probing is informational.
        result = shared.execute([tool,"--version"],check=False,capture_output=True,text=True)
        binary = Path(shutil.which(tool)).resolve()
        versions[tool] = {"path":str(binary),"sha256":sha(binary),
                          "stdout":result.stdout,"stderr":result.stderr,"returncode":result.returncode}
    versions.update(darshan_dependencies=linked, mpi4py=mpi,
                    darshan_runtime=str(args.darshan_library),python=sys.version, output_filesystem=output_fs,
                    git_revision=shared.execute(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(),
                    git_status=shared.execute(["git","status","--short"],check=True,capture_output=True,text=True).stdout)
    dump(probe/"provenance.json", versions)
    print("Preflight passed: eight ranks and eight forked readers captured with aligned clocks.", flush=True)


def locate():
    listing = shared.beegfs("--liststoragepools")
    matches = [pool for pool in (6,*shared.FAST) if set(shared.TARGETS) <= normal_members(listing,pool)]
    if len(matches) != 1:
        raise RuntimeError("Targets split/missing: inspect pool membership before continuing")
    return matches[0]


def normalize():
    active = locate()
    if active != 6:
        apply_limits(active)
        restore(active)
    else:
        clear_limits()
        verify_normal()


def validate_layout(raw, pool, members, directory=False):
    if not re.search(r"Chunksize:\s*512K",raw):
        raise RuntimeError("Expected 512-KiB BeeGFS chunks")
    if directory:
        # Directories store a desired pattern, not a file's actual target list.
        if (not re.search(rf"Storage Pool:\s*{pool}\b",raw,re.I)
                or not re.search(r"Number of storage targets: desired:\s*4\b",raw)):
            raise RuntimeError("Unexpected checkpoint-directory pool/stripe pattern")
        return []
    targets = [int(x) for x in re.findall(r"\+ (\d+) @ ",raw)]
    if len(targets) != 4 or len(set(targets)) != 4 or not set(targets) <= members:
        raise RuntimeError("File must use four distinct targets in the selected pool")
    return targets


def prepare(name, suite, pool, attempt):
    profile = shared.WORKLOADS[name]
    root = shared.prepare(profile, pool) if suite == "throttled" else shared.DATA_ROOT/profile.folder/f"pool{pool}"
    files = sorted((root/"dataset/train").glob(f"*.{profile.extension}"))
    if len(files) != profile.files or not (root/"dataset/valid").is_dir():
        raise RuntimeError(f"Incomplete existing dataset: {root}")
    checkpoints = root / "checkpoints" / (attempt.parent.parent.name+"-"+attempt.parent.name+"-"+attempt.name)
    checkpoints.mkdir()
    listing = shared.beegfs("--liststoragepools")
    members = normal_members(listing,pool)
    from configure_pool import remote
    limits = {host:remote(host,"systemctl","show","beegfs-storage.service",
                         "-p","IOReadBandwidthMax","-p","IOWriteBandwidthMax")
              for host in ("colva1","colva2","colva3","colva4")}
    kernel_limits = {host:remote(host,"sudo","-n","sh","-c",
        'cg=$(systemctl show -P ControlGroup beegfs-storage.service); cat "/sys/fs/cgroup${cg}/io.max"')
        for host in ("colva1","colva2","colva3","colva4")}
    if suite == "normal" and any(line.split("=",1)[1].strip() for text in limits.values()
                                  for line in text.splitlines() if "=" in line):
        raise RuntimeError("Normal profiling requires uncapped storage services on all four hosts")
    if suite == "throttled":
        shared.verify(pool)
    dump(attempt/"pool-state.json",{"listing":listing,"service_limits":limits,"kernel_io_max":kernel_limits})
    sampled = list(dict.fromkeys(files[::max(1,len(files)//64)][:64]+files[-1:]))
    layouts = []
    for path in [*sampled, checkpoints]:
        raw = shared.beegfs("--getentryinfo", "--verbose", path)
        targets = validate_layout(raw,pool,members,directory=path==checkpoints)
        layouts.append({"path": str(path), "ordered_targets": targets, "raw": raw})
    dump(attempt/"layout-sample.json",layouts)
    table(attempt/"input-inventory.csv", [{"name": path.name,"size_bytes": path.stat().st_size} for path in files])
    return root, checkpoints, members


def cache_prepare():
    shared.execute(["sync"],check=True)
    for host in ("colva1","colva2","colva3","colva4"):
        shared.execute(["ssh",host,"sudo -n sh -c 'sync && printf 3 > /proc/sys/vm/drop_caches'"],check=True)
    shared.execute(["sudo","-n","tee","/proc/sys/vm/drop_caches"],input="3\n",text=True,
                   stdout=subprocess.DEVNULL,check=True)
    if shared.remaining() is not None and shared.remaining() < 30:
        raise shared.DeadlineReached("Insufficient budget for cache preparation")
    time.sleep(30)


def case_run(attempt, name, suite, pool, args, hosts):
    root, checkpoints, targets = prepare(name,suite,pool,attempt)
    environment = instrumentation(attempt,root/"dataset",checkpoints,args.darshan_library,args.dxt_mib,args.module_mib)
    command = build_command(name,root/"dataset",checkpoints,attempt/"raw")
    manifest = {"workload":name,"suite":suite,"pool":pool,"dataset":str(root/"dataset"),
        "checkpoints":str(checkpoints),"settings":settings(shared.WORKLOADS[name]),
        "timezone_offset_s":datetime.now().astimezone().utcoffset().total_seconds(),
        "base_command":command,"command":launcher(command,environment),"instrumentation":environment,
        "trace_mode":"non-MPI per-process POSIX/DXT, PID mapped to MPI owner rank",
        "storage_targets":sorted(targets),
        "telemetry_interval_s":args.sample_seconds,
        "cache_protocol":"client sync; colva1-4 sync/drop; client drop; wait 30s; no between-epoch drops"}
    dump(attempt/"case.json",manifest)
    telemetry = Telemetry(attempt,hosts,targets,shared.execute,args.sample_seconds)
    telemetry.start()
    try:
        cache_prepare()
        telemetry.check()
        print(f"{name}: {suite} pool {pool}, profiled invocation 1/1 starting",flush=True)
        with (attempt/"raw/console.log").open("w") as file:
            shared.execute(manifest["command"],env=clean_environment(),stdout=file,stderr=subprocess.STDOUT,check=True)
        telemetry.check()
        if shared.remaining() is None or shared.remaining() > 3:
            time.sleep(3)  # Observe delayed writeback outside native DLIO timing.
    finally:
        try:
            terminate_owned(attempt)
        finally:
            telemetry.stop()
    sizes = shared.checkpoint_sizes([path for path in checkpoints.rglob("*") if path.is_file()],
                                   shared.WORKLOADS[name],(2,4),shared.WORKLOADS[name].model_bytes)
    table(attempt/"checkpoint-sizes.csv", [{"epoch":epoch,"rank":rank,"bytes":size} for (epoch,rank),size in sizes.items()])
    # Preserve placement evidence before deleting this invocation's checkpoints.
    checkpoint_layouts = []
    for path in sorted(p for p in checkpoints.rglob("*") if p.is_file()):
        checkpoint_layouts.append({"path":str(path),"raw":shared.beegfs("--getentryinfo","--verbose",path)})
    dump(attempt/"checkpoint-layouts.json",checkpoint_layouts)
    analyze_case(attempt,execute=shared.execute)
    shutil.rmtree(checkpoints)
    print(f"{name}: {suite} pool {pool}, invocation 1/1 complete; trace coverage passed",flush=True)


def sweep(output,args):
    end = time.monotonic()+args.deadline if args.deadline is not None else None
    reserve = min(60,args.deadline*.1) if end else 0
    shared.DEADLINE, shared.MANAGED = end-reserve if end else None, True
    state_path = output/"state.json"
    pool_phase = False
    restored = False
    try:
        identity = source_identity(args.darshan_library)
        if args.resume:
            state = json.loads(state_path.read_text())
            if state["identity"] != identity or state["trace_memory"] != [args.dxt_mib,args.module_mib]:
                raise RuntimeError("Workload/instrumentation source, runtime or memory settings changed; do not mix this with an existing trace study")
        else:
            if not args.preflight:
                verify_normal()
            state = {"identity":identity,"trace_memory":[args.dxt_mib,args.module_mib],
                     "matrix":matrix(args.workload),"cases":{},"status":"running"}
            dump(state_path,state)
        if args.resume:
            terminate_owned(output)
        # Preflight never changes targets or invokes DLIO.
        preflight(output,args)
        if args.preflight:
            state["status"] = "preflight_passed"
            dump(state_path,state)
            return
        pool_phase = True
        active = 6
        if args.resume:
            normalize()
        else:
            verify_normal()
        hosts = discover(shared.execute)
        dump(output/"live-telemetry-mapping.json",hosts)
        for name,suite,pool in state["matrix"]:
            key = f"{name}-{suite}-pool{pool}"
            progress = state["cases"].get(key)
            if progress and progress["status"] == "complete":
                print(f"{key}: trace already complete; skipping",flush=True)
                continue
            if suite == "normal" and active != 6:
                restore(active)
                active = 6
            if suite == "throttled" and active != pool:
                transition(active,pool)
                active = pool
            folder = output/key
            folder.mkdir(exist_ok=True)
            attempt = folder/("attempt-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            attempt.mkdir()
            state["cases"][key] = {"status":"running","attempt":str(attempt.relative_to(output))}
            dump(state_path,state)
            case_run(attempt,name,suite,pool,args,hosts)
            state["cases"][key]["status"] = "complete"
            dump(state_path,state)
        shared.DEADLINE = end
        normalize()
        restored = True
        state["status"] = "complete"
        dump(state_path,state)
        print(f"Profile study complete: {output}",flush=True)
    except BaseException as error:
        if "state" in locals():
            state["status"] = "interrupted" if isinstance(error,(shared.DeadlineReached,KeyboardInterrupt)) else "failed"
            state["error"] = str(error)
            for progress in state["cases"].values():
                if progress["status"] == "running":
                    progress["status"] = state["status"]
            dump(state_path,state)
        raise
    finally:
        # Even a parser/coverage error stops further launches and restores pools.
        # Never touch pools for --preflight or a failure before the run phase.
        if pool_phase and not restored:
            shared.DEADLINE = end if end else time.monotonic()+60
            try:
                terminate_owned(output)
                normalize()
                dump(output/"restoration.json",{"restored":True})
            except Exception as error:
                dump(output/"restoration.json",{"restored":False,"error":str(error)})
                print(f"Restoration incomplete: {error}; inspect pool membership/caps",flush=True)
        shared.DEADLINE, shared.MANAGED = None, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload",choices=("both","unet","cosmoflow"),default="both")
    parser.add_argument("--darshan-library",type=Path,help="Existing runtime; auto-detects earlier Darshan job paths/env")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--continue",dest="resume",type=Path,metavar="RUN_DIR")
    parser.add_argument("--deadline",type=duration)
    parser.add_argument("--dxt-mib",type=int,default=64,help="DXT POSIX memory budget per process")
    parser.add_argument("--module-mib",type=int,default=32,help="Aggregate module memory per process")
    parser.add_argument("--sample-seconds",type=float,default=1)
    parser.add_argument("--preflight",action="store_true",help="Only the tiny 8-rank/fork instrumentation check; no pool changes or DLIO")
    parser.add_argument("--plan",action="store_true",help="Print case matrix without executing or writing anything")
    args = parser.parse_args()
    if args.plan:
        for name,suite,pool in matrix(args.workload):
            print(f"{name:10} {suite:10} pool {pool:2} | 8 ranks | 5 epochs | 1 invocation")
        return
    try:
        args.darshan_library = existing_runtime(args.darshan_library)
    except RuntimeError as error:
        parser.error(str(error))
    print(f"Using existing Darshan runtime: {args.darshan_library}",flush=True)
    if min(args.dxt_mib,args.module_mib,args.sample_seconds) <= 0:
        parser.error("Memory budgets and sample interval must be positive")
    if args.resume and args.output:
        parser.error("Use either --continue RUN_DIR or --output")
    root = shared.RESULT_ROOT/"profiling"
    root.mkdir(parents=True,exist_ok=True)
    with (root/".profile.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another profile study is running")
        output = (args.resume or args.output or root/("profile-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).expanduser().resolve()
        if args.resume:
            if not (output/"state.json").exists():
                parser.error("No state.json in continuation directory")
        else:
            output.mkdir(parents=True,exist_ok=False)
        old = {signum:signal.signal(signum,lambda signum,frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {signum}")))
               for signum in (signal.SIGTERM,signal.SIGHUP)}
        print(f"Profiling output: {output}",flush=True)
        try:
            sweep(output,args)
        except (shared.DeadlineReached,KeyboardInterrupt) as error:
            print(f"Stopped: {error}. Partial attempts retained; use --continue {output}",flush=True)
        finally:
            for signum,handler in old.items():
                signal.signal(signum,handler)


if __name__ == "__main__":
    main()
