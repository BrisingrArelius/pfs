"""Small instrumentation preflight, not a DLIO workload or performance pilot."""

import os
from pathlib import Path
import subprocess

from mpi4py import MPI


rank = MPI.COMM_WORLD.Get_rank()
assert int(os.environ["PFS_DXT_RANK"]) == rank
case = Path(os.environ["PFS_DXT_CASE"])
# DLIO gathers system information using utility subprocesses. They inherit
# the tracing environment and must never collide with the rank's log.
subprocess.run(["/bin/true"],check=True)
path = case / "smoke-data" / f"rank-{rank}.bin"
path.write_bytes(b"x" * 4096)
assert path.read_bytes() == b"x" * 4096
child = os.fork()
if child == 0:
    assert path.read_bytes() == b"x" * 4096
    os._exit(0)
_, status = os.waitpid(child, 0)
assert status == 0
MPI.COMM_WORLD.Barrier()
if rank == 0:
    print(MPI.Get_library_version(), flush=True)
