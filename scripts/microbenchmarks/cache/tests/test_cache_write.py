"""Offline contracts for the separate write runner and shared preparation."""

import json
import os
from pathlib import Path
import pwd
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cache_common as common
import run_cache_read as read
import run_cache_write as write


class CacheWriteContract(unittest.TestCase):
    def test_case_counts_and_states(self):
        counts = {("buffered", "false"): (2, 10),
                  ("buffered", "true"): (2, 10),
                  ("native", "false"): (2, 10),
                  ("native", "true"): (4, 20)}
        for (mode, fsync), (pilot, full) in counts.items():
            self.assertEqual(len(write.cases(True, mode, fsync)), pilot)
            self.assertEqual(len(write.cases(False, mode, fsync)), full)
            self.assertEqual(sum(state == "client_ram" for _, state in
                                 write.cases(True, mode, fsync)),
                             2 if (mode, fsync) == ("native", "true") else 0)

    def test_read_and_write_share_preparation(self):
        self.assertIs(read.drop, common.drop)
        self.assertIs(read.target_file, common.target_file)
        self.assertIs(read.prepared_run, common.prepared_run)
        self.assertIs(write.drop, common.drop)
        self.assertIs(write.target_file, common.target_file)
        self.assertIs(write.prepared_run, common.prepared_run)

    def test_shared_preparation_saves_live_configuration_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            shared = root / "shared"
            runs.mkdir()
            shared.mkdir()
            results = runs / "cache-unit"
            results.mkdir()
            owner = {"run_id": "cache-unit", "user": pwd.getpwuid(os.getuid()).pw_name,
                     "mode": "buffered", "pilot": True}
            common.record(results / "owner.json", owner)
            config = root / "config"
            config.write_text("tuneFileCacheType = buffered\n"
                              "tuneFileCacheBufSize = 524288\n"
                              "tuneRemoteFSync = true\n")
            with patch.object(common, "RUNS", runs), patch.object(
                    common, "SHARED", shared), patch.object(
                    common, "check_cluster"), patch.object(
                    common.Path, "glob", return_value=[config]):
                with common.prepared_run("cache-unit", True, "buffered", results) as namespace:
                    self.assertTrue(namespace.exists())
                self.assertFalse(namespace.exists())
            snapshot = json.loads((results / "live_config.json").read_text())
            self.assertEqual(snapshot[0]["settings"]["tuneRemoteFSync"], "true")

    def test_live_remote_fsync_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            with patch.object(write.Path, "glob", return_value=[config]):
                for value, expected, opposite in (
                        ("1", "true", "false"), ("true", "true", "false"),
                        ("0", "false", "true"), ("false", "false", "true")):
                    config.write_text(f"tuneRemoteFSync = {value}\n")
                    write.check_remote_fsync(expected)
                    with self.assertRaisesRegex(RuntimeError, f"tuneRemoteFSync={opposite}"):
                        write.check_remote_fsync(opposite)
                config.write_text("tuneRemoteFSync =\ntuneFileCacheType = buffered\n")
                with self.assertRaisesRegex(RuntimeError, "observed"):
                    write.check_remote_fsync("true")

    def test_ior_command_and_traffic_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            samples = [{"client_sent": 0, "server_received": 0, "device_written": 0},
                       {"client_sent": common.SIZE, "server_received": common.SIZE,
                        "device_written": common.SIZE}]

            def fake_run(argv, *, output=None):
                self.assertEqual(argv[:6], ["/usr/bin/mpirun", "-np", "1",
                                            "/usr/local/bin/ior", "-a", "POSIX"])
                for flag in ("-w", "-E", "-k", "-g", "-e"):
                    self.assertIn(flag, argv)
                self.assertNotIn("--posix.odirect", argv)
                summary = Path(argv[argv.index("-O", argv.index("-O") + 1) + 1]
                               .split("=", 1)[1])
                summary.write_text(json.dumps({"summary": [{
                    "API": "POSIX", "operation": "write", "numTasks": 1,
                    "blockSize": common.SIZE, "transferSize": 1024**2,
                    "xsizeMiB": 8192, "bwMeanMIB": 123.0, "MeanTime": 66.0}]}))

            with patch.object(write, "run", side_effect=fake_run), patch.object(
                    write, "counters", side_effect=samples):
                rate, seconds, evidence = write.ior_write(
                    folder / "file", folder, "eth0", "sdb1")
            self.assertEqual((rate, seconds), (123.0, 66.0))
            self.assertEqual(evidence["device_written_ratio"], 1)
            self.assertEqual(json.loads((folder / "command.json").read_text())[6:11],
                             ["-w", "-E", "-k", "-g", "-e"])

    def test_write_labels_follow_policy_and_raw_traffic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            idle = {"client_sent": 0, "server_received": 0, "device_written": 0}
            for index, (state, policy, device_ratio, expected) in enumerate((
                    ("server_ram", "false", 1.0, "server_ram"),
                    ("server_disk", "true", .1, "unverified")), 1):
                path = root / f"file-{index}"
                with path.open("wb") as stream:
                    stream.truncate(common.SIZE)
                evidence = {"client_sent_ratio": 1.0, "server_received_ratio": 1.0,
                            "device_written_ratio": device_ratio}
                with patch.object(write, "target_file", return_value=path), patch.object(
                        write, "drop"), patch.object(write, "counters",
                        side_effect=[idle, idle]), patch.object(write.time, "sleep"), patch.object(
                        write, "ior_write", return_value=(100.0, 80.0, evidence)), patch.object(
                        write, "targets", return_value=[101]):
                    result = write.measure(index, "HDD", state, "buffered", policy,
                                           "eth0", root, root)
                self.assertEqual(result["achieved"], expected)
                self.assertTrue((root / f"{index:02d}-hdd-{state}" / "idle_counters.json").exists())
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
