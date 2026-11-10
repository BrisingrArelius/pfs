"""Opt-in launcher for DLIO 2.0.0's built-in Darshan profiler."""

from datetime import datetime, timedelta
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def add_argument(parser):
    parser.add_argument("--darshan", nargs="?", type=Path,
                        const=Path(os.environ.get("DARSHAN_RUNTIME_LIB",
                                   str(Path.home()/"darshan-nompi/lib/libdarshan.so.0.0.0"))),
                        metavar="LIBRARY", help="Enable DLIO's built-in Darshan profiler; optionally specify the installed runtime")


def validate(library):
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Unset controller LD_PRELOAD; --darshan loads the runtime into ranks only")
    library = Path(library).expanduser().resolve(strict=True)
    if not library.is_file():
        raise ValueError(f"Darshan runtime is not a file: {library}")
    for tool in ("darshan-parser", "darshan-dxt-parser"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} must be available to validate saved logs")
    return library


def clean_environment(environment):
    return {key:value for key,value in environment.items()
            if not key.startswith(("DARSHAN_", "DXT_"))}


def instrument(command, directory, dataset, checkpoints, library):
    """Keep workload arguments; inject the runtime only after mpirun -np N."""
    logs = directory/"darshan"
    logs.mkdir()
    for days in range(2):
        date = datetime.now()+timedelta(days=days)
        (logs/str(date.year)/str(date.month)/str(date.day)).mkdir(parents=True, exist_ok=True)
    paths = [Path(path).resolve() for path in (dataset, checkpoints)]
    if any(char.isspace() for path in paths+[logs.resolve()] for char in str(path)):
        raise ValueError("Darshan capture paths must not contain whitespace")
    config = logs/"darshan.conf"
    config.write_text("MOD_ENABLE POSIX,STDIO,DXT_POSIX\nMODMEM 96\nNAMEMEM 8\n"
                      "MAX_RECORDS 16384 POSIX,STDIO\nMAX_RECORDS 65536 DXT_POSIX\n"
                      "NAME_EXCLUDE .* *\n"+"".join(
                          f"NAME_INCLUDE ^{re.escape(str(path))}/ POSIX,STDIO,DXT_POSIX\n"
                          for path in paths))
    variables = {"LD_PRELOAD":str(library), "DARSHAN_ENABLE_NONMPI":"1",
                 "DARSHAN_MOD_ENABLE":"POSIX,STDIO,DXT_POSIX",
                 "DARSHAN_LOG_DIR_PATH":str(logs.resolve()), "DARSHAN_LOGPATH":str(logs.resolve()),
                 "DARSHAN_CONFIG_PATH":str(config.resolve())}
    return [*command[:3], "env", "-u", "DARSHAN_LOGFILE", "-u", "DARSHAN_DISABLE",
            *(f"{key}={value}" for key,value in variables.items()),
            sys.executable, str(Path(__file__).resolve()), *command[4:],
            "++workload.workflow.profiling=true", "++workload.profiling.profiler=darshan"]


def check_logs(directory, ranks=8):
    logs = directory/"darshan"
    files = sorted(logs.rglob("*.darshan"))
    workers = sorted(logs.glob("worker-*.json"))
    identities = [logs/f"rank-{rank}.json" for rank in range(ranks)]+workers
    for path in identities:
        identity = json.loads(path.read_text())
        matches = [path for path in files if re.search(rf"_id\d+-{identity['pid']}_", path.name)]
        if len(matches) != 1:
            raise RuntimeError(f"Rank {identity['rank']}, PID {identity['pid']}: expected one native Darshan log, found {len(matches)} in {logs}")
    for path in files:
        for tool in ("darshan-parser", "darshan-dxt-parser"):
            subprocess.run([tool, str(path)], stdout=subprocess.DEVNULL, check=True)
    print(f"Darshan: {ranks} rank logs present; {len(workers)} forked-process logs checked; {len(files)} total logs parsed without incomplete-module errors: {logs}", flush=True)


def main():
    from dlio_benchmark.profiler.darshan_profiler import DarshanProfiler

    original = DarshanProfiler.__init__
    original_start = DarshanProfiler.start

    def native_filename(self):
        original(self)
        # DLIO forces one filename for every rank. Let the existing non-MPI
        # runtime generate PID-specific names; keep DLIO's start/stop methods.
        os.environ.pop("DARSHAN_LOGFILE", None)

    def start(self):
        original_start(self)
        # Darshan 3.4/3.5 tests presence, not the value: DLIO's "0" still
        # disables initialization in new reader/utility processes.
        os.environ.pop("DARSHAN_DISABLE", None)

    DarshanProfiler.__init__ = native_filename
    DarshanProfiler.start = start
    rank = os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK"))
    if rank is None:
        raise RuntimeError("MPI rank identity is unavailable")
    logs = Path(os.environ["DARSHAN_LOG_DIR_PATH"])
    owner = os.getpid()

    def identity():
        if os.getpid() != owner and "DARSHAN_DISABLE" in os.environ:
            return  # No capture expected outside DLIO's active profiling window.
        name = f"rank-{int(rank)}" if os.getpid() == owner else f"worker-{os.getpid()}"
        (logs/f"{name}.json").write_text(json.dumps(
            {"rank":int(rank), "pid":os.getpid(), "parent_pid":os.getppid()})+"\n")

    identity()
    os.register_at_fork(after_in_child=identity)
    sys.argv[0] = "dlio_benchmark"
    next(iter(entry_points(group="console_scripts", name="dlio_benchmark"))).load()()


if __name__ == "__main__":
    main()
