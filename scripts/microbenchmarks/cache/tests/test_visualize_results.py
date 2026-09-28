"""Cache plots are built from native IOR and state evidence, without CSV."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
sys.path.insert(0, str(HERE))
import run_cache as runner
import visualize_results as plots


class RawCachePlots(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = runner.load_config(HERE / "cache_config.json")
        units = runner.pilot_units(self.config)
        (self.root / "owner.json").write_text(json.dumps({"run_id": self.root.name,
                                                            "domain": "cache-pilot"}))
        plan_hash = plots.fingerprint({"domain": "cache-pilot", "config": self.config, "units": units})
        (self.root / "plan.json").write_text(json.dumps({"domain": "cache-pilot",
            "config": self.config, "units": units, "fingerprint": plan_hash}))
        inventory = {"domain": "cache", "reviewed": True, "client_mode": "native",
                     "available_ram_bytes": 20 * 1024**3,
                     "targets": [{"id": 100 * index + 1, "oss": f"colva{index}",
                                  "media": "HDD", "state": "Online/Good", "device": "/dev/test"}
                                 for index in range(1, 5)], "devices": ["test"],
                     "mount": "/mnt/beegfs", "namespace": "/mnt/beegfs/owned",
                     "restoration": "verified", "restoration_baseline": {"mode": "native"},
                     "watchdog_verified": True,
                     "privilege_verified": True, "tools": {"ior_version": "pinned",
                     "ior": "/usr/bin/ior", "mpirun": "/usr/bin/mpirun",
                     "mpi_binding_pattern": r"^rank (?P<rank>\d+) host (?P<host>\w+) core (?P<core>\d+)$"}}
        (self.root / "inventory.json").write_text(json.dumps(inventory))
        for unit in units:
            folder = self.root / "attempts" / unit["id"] / "attempt-1"
            folder.mkdir(parents=True)
            state = unit["state"]
            owned = inventory["namespace"] + "/file"
            size = 8 * 1024**3
            rate = {runner.STATES[0]: 200, runner.STATES[1]: 400,
                    runner.STATES[2]: 800}[state]
            seconds = 8192 / rate
            summary = folder / "native.json"
            native = {"Version": "pinned", "tests": [{"Parameters": {
                "api": "POSIX", "repetitions": 1, "readFile": 1, "writeFile": 0,
                "testFileName": owned, "blockSize": size, "transferSize": 1048576,
                "filePerProc": 0, "useExistingTestFile": 1},
                "Options": {"tasks": 1, "Results": [{"access": "read", "bwMiB": rate,
                    "iops": rate, "totalTime": seconds, "wrRdTime": seconds - .1}]}}],
                "summary": [{"operation": "read", "API": "POSIX", "numTasks": 1,
                             "blockSize": size, "transferSize": 1048576,
                             "segmentCount": 1, "repetitions": 1, "filePerProc": 0,
                             "xsizeMiB": 8192, "MeanTime": seconds, "bwMeanMIB": rate}]}
            telemetry = {"logical_bytes": size, "network_bytes": 0 if state == "client_hit" else size,
                         "backend_read_bytes": size if state == "client_miss_server_miss" else 0,
                         "client_residency": 1 if state == "client_hit" else 0,
                         "quiescent": True, "netbench_off": True,
                         "telemetry_complete": True, "file_unchanged": True}
            evidence = {"exclusive_allocation": True, "writeback_settled": True,
                        "file_identity_before": "file-1", "file_identity_after": "file-1",
                        "client_residency": telemetry["client_residency"]}
            evidence["steps"] = runner.prepare_cache_state(state, evidence=evidence)
            files = {"native": native, "telemetry": telemetry, "state": evidence,
                     "rank_map": [{"rank": 0, "host": "anjuna2", "core": 1}],
                     "rank_report": "rank 0 host anjuna2 core 1\n",
                     "command": runner.build_command("/usr/bin/ior", "/usr/bin/mpirun", owned, summary),
                     "exit": {"returncode": 0, "timeout": False}}
            attempt_evidence = {"owned_path": owned, "summary_path": str(summary), "sha256": {}}
            for name, data in files.items():
                path = folder / (f"{name}.txt" if name == "rank_report" else f"{name}.json")
                path.write_text(data if isinstance(data, str) else json.dumps(data))
                attempt_evidence[name] = str(path.relative_to(self.root))
                attempt_evidence["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            unit.update(cleanup="completed", attempts=[{"id": "attempt-1", "state": "completed",
                "cleanup": "completed", "evidence": attempt_evidence}])
        run_hash = plots.fingerprint({"domain": "cache-pilot", "plan": plan_hash,
                                      "inventory": inventory})
        restoration = self.root / "state-transitions" / "restoration.json"
        restoration.parent.mkdir()
        restoration.write_text(json.dumps({"run_fingerprint": run_hash,
            "baseline": inventory["restoration_baseline"],
            "restored": inventory["restoration_baseline"],
            "netbench_off": {"anjuna2": True, "anjuna3": True},
            "watchdog_released": True}))
        (self.root / "manifest.json").write_text(json.dumps({"domain": "cache-pilot",
            "units": units, "fingerprint": run_hash, "restoration": "completed",
            "restoration_evidence": {"path": str(restoration.relative_to(self.root)),
                "sha256": hashlib.sha256(restoration.read_bytes()).hexdigest()}}))

    def test_raw_run_yields_verified_bandwidth_and_traffic_figures(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        manifest = json.loads((self.root / "plots" / "plot_manifest.json").read_text())
        self.assertEqual(manifest["measurements"], 4)
        self.assertEqual(manifest["unverified"], 0)
        self.assertEqual(len(manifest["plots"]), 2)
        self.assertTrue(all((self.root / "plots" / name).is_file() for name in manifest["plots"]))
        self.assertFalse((self.root / "analysis").exists())

    def test_corrupt_native_and_path_escape_are_rejected(self):
        native = next(self.root.glob("attempts/*/*/native.json"))
        native.write_text("corrupt")
        self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertFalse((self.root / "plots" / "plot_manifest.json").exists())
        self.assertEqual(plots.main([str(self.root), "--output-dir", str(self.root.parent)]), 1)

    def test_plot_symlink_never_modifies_an_unrelated_file(self):
        victim = self.root / "unrelated"
        victim.write_text("preserve")
        output = self.root / "plots"
        output.mkdir()
        (output / "cache_throughput.png").symlink_to(victim)
        self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertEqual(victim.read_text(), "preserve")
        self.assertTrue((output / "cache_throughput.png").is_symlink())

    def test_render_failure_discards_staging_and_allows_retry(self):
        original = plots.save_figure
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("simulated rendering failure")
            return original(*args)

        with mock.patch.object(plots, "save_figure", side_effect=interrupted):
            self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertEqual(list((self.root / "plots").glob("*.png")), [])
        self.assertEqual(list((self.root / "plots").glob(".plot-stage-*")), [])
        self.assertEqual(plots.main([str(self.root)]), 0)

    def test_publish_failure_restores_existing_cache_figures(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        output = self.root / "plots"
        before = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in output.rglob("*") if path.is_file()}
        replace = plots.os.replace

        def fail_manifest(source, destination):
            if Path(source).name.startswith(".plot-manifest-"):
                raise OSError("simulated manifest publication failure")
            return replace(source, destination)

        with mock.patch.object(plots.os, "replace", side_effect=fail_manifest):
            self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertEqual(before, {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in output.rglob("*") if path.is_file()})

    def test_process_loss_before_manifest_switch_keeps_old_generation(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        output = self.root / "plots"
        old_manifest = (output / "plot_manifest.json").read_bytes()
        old = json.loads(old_manifest)
        hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in old["plots"]}
        script = ("import os,sys; sys.path.insert(0,sys.argv[1]); "
                  "import visualize_results as p; original=p.os.replace; "
                  "exec('def crash(source, target):\\n'"
                  "     ' if str(source).split(\"/\")[-1].startswith(\".plot-manifest-\"): os._exit(72)\\n'"
                  "     ' return original(source,target)\\n'); "
                  "p.os.replace=crash; p.main([sys.argv[2]])")
        env = os.environ.copy()
        cache = self.root / ".mpl-cache"
        cache.mkdir()
        env["MPLCONFIGDIR"] = str(cache)
        crashed = subprocess.run([sys.executable, "-B", "-c", script, str(HERE), str(self.root)],
                                 capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(crashed.returncode, 72, crashed.stderr)
        self.assertEqual((output / "plot_manifest.json").read_bytes(), old_manifest)
        self.assertEqual(hashes, {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in old["plots"]})
        self.assertEqual(plots.main([str(self.root)]), 0)
        current = json.loads((output / "plot_manifest.json").read_text())
        self.assertEqual(len(list((output / "generations").iterdir())), 1)
        self.assertTrue(all((output / name).is_file() for name in current["plots"]))


if __name__ == "__main__":
    unittest.main()
