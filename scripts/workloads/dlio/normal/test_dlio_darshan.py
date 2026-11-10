"""Local checks only: no MPI launch, cluster administration or DLIO training."""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dlio_darshan as darshan


class LauncherTests(unittest.TestCase):
    def test_inherited_capture_settings_do_not_override_job_configuration(self):
        environment={"PATH":"/bin", "PYTHONPATH":"/project", "DARSHAN_MOD_DISABLE":"POSIX",
                     "DARSHAN_DISABLE":"1", "DXT_ENABLE_IO_TRACE":"0"}
        self.assertEqual(darshan.clean_environment(environment),{"PATH":"/bin", "PYTHONPATH":"/project"})
        self.assertIn("DARSHAN_MOD_DISABLE",environment)

    def test_original_workload_commands_are_unchanged(self):
        root=Path(__file__).resolve().parents[4]
        for suite in ("normal", "throttled_ssd"):
            path=f"scripts/workloads/dlio/{suite}/pool_common.py"
            original=subprocess.check_output(["git","show",f"8844ed9:{path}"],cwd=root,text=True)
            current=(root/path).read_text()
            def workload_commands(source):
                return [ast.dump(node.value) for node in ast.walk(ast.parse(source))
                        if isinstance(node,ast.Assign) and isinstance(node.value,ast.List)
                        and any(isinstance(target,ast.Name) and target.id=="command" for target in node.targets)]
            self.assertTrue(workload_commands(original))
            self.assertEqual(workload_commands(current),workload_commands(original))

    def test_all_four_cli_entry_points(self):
        root=Path(__file__).resolve().parents[1]
        for suite in ("normal", "throttled_ssd"):
            for name in ("cosmoflow", "unet3d"):
                result=subprocess.run([sys.executable,str(root/suite/f"run_{name}_pools.py"),"--help"],
                                      capture_output=True,text=True,check=True)
                self.assertIn("--darshan",result.stdout)
                self.assertIn("--pilot",result.stdout)

    def test_opt_in_and_original_workload_arguments(self):
        parser = argparse.ArgumentParser()
        darshan.add_argument(parser)
        self.assertIsNone(parser.parse_args([]).darshan)
        self.assertIsInstance(parser.parse_args(["--darshan"]).darshan, Path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = ["mpirun", "-np", "8", "dlio_benchmark", "workload=cosmoflow_v100",
                        "++workload.train.epochs=5", "++workload.reader.read_threads=4"]
            command = darshan.instrument(original, root, root/"data", root/"checkpoints", Path("/existing/libdarshan.so"))
            start = command.index(str(Path(darshan.__file__).resolve()))+1
            self.assertEqual(command[start:-2], original[4:])
            self.assertEqual(command[-2:], ["++workload.workflow.profiling=true", "++workload.profiling.profiler=darshan"])
            self.assertEqual(original[3], "dlio_benchmark")

    def test_missing_rank_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"darshan").mkdir()
            (root/"darshan/rank-0.json").write_text(json.dumps({"rank":0,"pid":123}))
            with self.assertRaisesRegex(RuntimeError, "Rank 0"):
                darshan.check_logs(root, ranks=1)

    def test_incomplete_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"darshan").mkdir()
            (root/"darshan/rank-0.json").write_text(json.dumps({"rank":0,"pid":123}))
            (root/"darshan/test_id123-123_1.darshan").touch()
            error=subprocess.CalledProcessError(1,["darshan-parser"])
            with patch.object(darshan.subprocess,"run",side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    darshan.check_logs(root, ranks=1)


class RuntimeTests(unittest.TestCase):
    def test_native_names_helpers_fork_and_stdio(self):
        library = Path(os.environ.get("PFS_TEST_DARSHAN_LIBRARY", "/usr/local/lib/libdarshan.so.0.0.0"))
        if not library.exists() or not shutil.which("darshan-parser") or not shutil.which("darshan-dxt-parser"):
            self.skipTest("Installed local Darshan runtime/utilities unavailable")
        # Substitute only DLIO's entry point/profiler, unavailable locally.
        # The actual Darshan library, file I/O, fork and utility are real.
        code = """
import ctypes, os, subprocess, sys, types
from pathlib import Path
import dlio_darshan as d
data, checkpoints = map(Path, sys.argv[1:3])
class Profiler:
    def __init__(self):
        os.environ['DARSHAN_LOGFILE'] = str(data/'colliding.darshan')
    def start(self):
        os.environ['DARSHAN_DISABLE'] = '0'
    def stop(self):
        os.environ['DARSHAN_DISABLE'] = '1'
module = types.ModuleType('dlio_benchmark.profiler.darshan_profiler')
module.DarshanProfiler = Profiler
sys.modules[module.__name__] = module
def run():
    profiler = Profiler()
    assert 'DARSHAN_LOGFILE' not in os.environ
    profiler.start()
    assert 'DARSHAN_DISABLE' not in os.environ
    subprocess.run(['/bin/true'], check=True)
    path = data/('data-'+os.environ['PMI_RANK'])
    path.write_bytes(b'x'*4096)
    assert path.read_bytes() == b'x'*4096
    child = os.fork()
    if child == 0:
        assert path.read_bytes() == b'x'*4096
        os._exit(0)
    assert os.waitpid(child, 0)[1] == 0
    if os.environ['PMI_RANK'] == '0':
        paths = [data/('record-'+str(i)) for i in range(8192)]
        for item in paths:
            item.write_bytes(b'record-capacity-probe')
        for epoch in range(5):
            for item in paths:
                assert item.read_bytes() == b'record-capacity-probe'
    libc = ctypes.CDLL(None)
    libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.fopen.restype = ctypes.c_void_p
    libc.fputs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    libc.fclose.argtypes = [ctypes.c_void_p]
    stream = libc.fopen(os.fsencode(checkpoints/('index-'+os.environ['PMI_RANK'])), b'wb')
    assert stream
    assert libc.fputs(b'checkpoint-index', stream) >= 0
    assert libc.fclose(stream) == 0
    profiler.stop()
    assert os.environ['DARSHAN_DISABLE'] == '1'
class Entry:
    def load(self):
        return run
d.entry_points = lambda **kwargs: [Entry()]
d.main()
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, checkpoints = root/"data", root/"checkpoints"
            data.mkdir()
            checkpoints.mkdir()
            command = darshan.instrument(["mpirun","-np","8","dlio_benchmark"], root, data, checkpoints, library)
            variables = dict(value.split("=",1) for value in command[8:command.index(sys.executable)])
            environment = {key:value for key,value in os.environ.items()
                           if not key.startswith(("DARSHAN_", "DXT_"))}
            environment.update(variables, PYTHONPATH=str(Path(darshan.__file__).parent))
            for rank in range(8):
                subprocess.run([sys.executable,"-c",code,str(data),str(checkpoints)],
                               env={**environment,"PMI_RANK":str(rank)},
                               capture_output=True,text=True,check=True,timeout=30)
            darshan.check_logs(root)
            self.assertEqual(len(list((root/"darshan").glob("worker-*.json"))),8)
            files = list((root/"darshan").rglob("*.darshan"))
            self.assertEqual(len(files),24)  # Eight roots, readers and utilities.
            traced_readers = 0
            for path in files:
                parsed = subprocess.check_output(["darshan-parser","--total",str(path)],text=True)
                if "# exe: /bin/true" in parsed:
                    continue
                trace = subprocess.check_output(["darshan-dxt-parser",str(path)],text=True)
                self.assertIn(str(data), trace)
                self.assertIn("read", trace)
                traced_readers += 1
            self.assertEqual(traced_readers,16)
            rank = json.loads((root/"darshan/rank-0.json").read_text())
            parent = next(path for path in files if f"-{rank['pid']}_" in path.name)
            parsed = subprocess.check_output(["darshan-parser","--total",str(parent)],text=True)
            self.assertIn("STDIO_BYTES_WRITTEN",parsed)


if __name__ == "__main__":
    unittest.main()
