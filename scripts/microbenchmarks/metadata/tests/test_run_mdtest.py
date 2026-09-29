"""Focused local checks for the fixed mdtest plan, ownership and raw resume."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_mdtest as md


class MetadataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=md.RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_fixed_plans_and_command(self):
        self.assertEqual(len(md.units(True)), 4)
        full = md.units(False)
        self.assertEqual(len(full), 90)
        self.assertEqual(len({row["id"] for row in full}), 90)
        self.assertEqual({row["ranks"] for row in full}, {1, 4, 16})
        argv = md.command(md.units(True)[-1], "/work", 1000)
        self.assertEqual(argv[1:7], ["-wdir", "/work", "-np", "8", "-hosts",
                                     "anjuna2:4,anjuna3:4"])
        self.assertEqual(argv[argv.index("-n") + 1], "1000")
        self.assertIn("-u", md.command(md.units(True)[2], "/work", 1000))

    def test_cleanup_rejects_links_and_preserves_siblings(self):
        shared = self.root / "metadata-pilot-test"
        shared.mkdir()
        unit = md.units(True)[0]
        owner = {"run_id": shared.name}
        work = shared / unit["id"]
        work.mkdir()
        md.save(work / "owner.json", owner)
        (work / "work").mkdir()
        sibling = shared / "keep"
        sibling.write_text("untouched")
        (work / "work" / "link").symlink_to(self.root)
        with self.assertRaises(ValueError):
            md.cleanup(work, shared, unit, owner)
        (work / "work" / "link").unlink()
        md.cleanup(work, shared, unit, owner)
        self.assertEqual(sibling.read_text(), "untouched")

    def test_pilot_raw_checkpoint_and_resume(self):
        runs = self.root / "results" / "microbenchmarks" / "runs"
        runs.mkdir(parents=True)
        namespace = self.root / "beegfs" / "pfs" / ".metadata-mdtest"
        namespace.parent.mkdir(parents=True)
        state = {"mount": "beegfs beegfs_nodev /mnt/beegfs rw", "mount_device": 1,
                 "client_config_sha256": "local"}
        baseline = {**state, "tools": {name: {"path": f"/mock/{name}", "sha256": name}
                                      for name in ("mpirun", "mdtest")}}
        remote = {"client_config_sha256": {"anjuna2": "a2", "anjuna3": "a3"}}
        launched = []

        def fake_launch(argv, output, _timeout):
            launched.append(argv)
            (output / "stdout.txt").write_text("native mdtest output\n")
            (output / "stderr.txt").write_text("")
            return {"returncode": 0, "interrupted": False}

        with (patch.object(md, "RUNS", runs), patch.object(md, "NAMESPACE", namespace),
              patch.object(md, "MPIRUN", Path("/mock/mpirun")),
              patch.object(md, "MDTEST", Path("/mock/mdtest")),
              patch.object(md, "preflight", return_value=baseline),
              patch.object(md, "mount_state", return_value=state),
              patch.object(md, "remote_probe", return_value=remote),
              patch.object(md, "launch", side_effect=fake_launch)):
            md.benchmark("metadata-pilot-local", True)
            md.benchmark("metadata-pilot-local", True)
            self.assertEqual(len(launched), 4)
            case = runs / "metadata-pilot-local" / "cases" / "pilot-01"
            self.assertEqual(json.loads((case / "result.json").read_text())["cleanup"], "completed")
            (case / "stdout.txt").write_text("changed")
            with self.assertRaises(ValueError):
                md.benchmark("metadata-pilot-local", True)

    def test_remote_probe_shell_syntax(self):
        class Result:
            returncode = 0
            stdout = "anjuna2 hash2\nanjuna3 hash3\n"
            stderr = ""

        with patch.object(md.subprocess, "run", return_value=Result()) as called:
            md.remote_probe(self.root / "owner.json", "mdtest-hash")
        argv = called.call_args.args[0]
        self.assertEqual(argv[1:3], ["-wdir", str(self.root)])
        script = argv[9]
        self.assertEqual(subprocess.run(["/bin/sh", "-n", "-c", script]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
