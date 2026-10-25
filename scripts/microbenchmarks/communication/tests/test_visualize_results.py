"""NetBench figures consume pilot native IOR files rather than derived CSV."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
sys.path.insert(0, str(HERE))
import run_communication as runner
import visualize_results as plots


class RawCommunicationPlots(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = runner.load_config(HERE / "communication_config.json")
        units = runner.pilot_units(config)
        (self.root / "owner.json").write_text(json.dumps({"run_id": self.root.name,
                                                            "domain": "communication-pilot"}))
        plan_hash = plots.fingerprint({"domain": "communication-pilot", "config": config,
                                       "units": units})
        (self.root / "plan.json").write_text(json.dumps({"domain": "communication-pilot",
            "config": config, "units": units, "fingerprint": plan_hash}))
        targets = [{"id": 100 * index + 1, "oss": f"colva{index}", "state": "Online/Good"}
                   for index in range(1, 5)]
        inventory = {"domain": "communication", "reviewed": True, "clients": {"a": 1},
                     "targets": targets, "mount": "/mnt/beegfs",
                     "netbench_paths": {"anjuna2": "/proc/a2", "anjuna3": "/proc/a3"},
                     "pool_baseline": "saved", "chooser_baseline": "saved",
                     "restoration_baseline": {"pool": "saved", "chooser": "saved"},
                     "namespace": "/mnt/beegfs/owned", "watchdog_verified": True,
                     "restoration_tested": True, "tools": {"ior_version": "pinned",
                         "ior": "/usr/bin/ior", "mpirun": "/usr/bin/mpirun",
                         "mpi_binding_pattern": r"^rank (?P<rank>\d+) host (?P<host>\w+) core (?P<core>\d+)$"}}
        (self.root / "inventory.json").write_text(json.dumps(inventory))
        for unit in units:
            folder = self.root / "attempts" / unit["id"] / "attempt-1"
            folder.mkdir(parents=True)
            clients = ("anjuna2", "anjuna3") if unit["placement"] == "dual" else (unit["placement"],)
            tasks = len(clients) * unit["ranks_per_client"]
            owned = inventory["namespace"] + "/file"
            summary = folder / "native.json"
            operation = unit["direction"]
            mib = 8192 * tasks
            bandwidth = mib / 30.1
            params = {"api": "POSIX", "testFileName": owned, "deadlineForStonewall": 30,
                      "repetitions": 1, "blockSize": 16 * 1024**3,
                      "transferSize": 1048576, "segmentCount": 1,
                      "filePerProc": int(unit["organization"] == "fpp"), "fsync": 0,
                      "randomOffset": 0, "useExistingTestFile": int(operation == "read"),
                      "readFile": int(operation == "read"), "writeFile": int(operation == "write")}
            native = {"Version": "pinned", "tests": [{"Parameters": params,
                "Options": {"tasks": tasks, "Results": [{"access": operation,
                    "bwMiB": bandwidth, "iops": bandwidth, "totalTime": 30.1,
                    "wrRdTime": 30.0}]}}],
                "summary": [{"operation": operation, "API": "POSIX", "numTasks": tasks,
                    "blockSize": 16 * 1024**3, "transferSize": 1048576, "segmentCount": 1,
                    "repetitions": 1, "filePerProc": int(unit["organization"] == "fpp"),
                    "xsizeMiB": mib, "MeanTime": 30.1, "bwMeanMIB": bandwidth}]}
            before = {client: 0 for client in runner.CLIENTS}
            during = {client: int(client in clients) for client in runner.CLIENTS}
            layout = [101] if unit["stripes"] == 1 else [101, 201, 301, 401]
            layouts = [layout] * (tasks if unit["organization"] == "fpp" else 1)
            state = {"before_mode": before, "during_mode": during, "after_mode": before,
                     "watchdog_verified": True, "actual_layouts": layouts}
            telemetry = {**state, "layout_verified": True, "telemetry_complete": True,
                         "pool_restored": True, "chooser_restored": True,
                         "rank_completion_seconds": [30] * tasks,
                         "rank_bytes": [8 * 1024**3] * tasks,
                         "backend_read_bytes": 0, "backend_write_bytes": 0}
            files = {"native": native, "telemetry": telemetry, "state": state,
                     "rank_map": [{"rank": rank, "host": client, "core": index}
                                  for rank, (client, index) in enumerate((client, index)
                                      for client in clients for index in range(unit["ranks_per_client"]))],
                     "command": runner.build_command(unit, "/usr/bin/mpirun", "/usr/bin/ior",
                                                     owned, summary),
                     "exit": {"returncode": 0, "timeout": False}}
            files["rank_report"] = "".join(
                f"rank {entry['rank']} host {entry['host']} core {entry['core']}\n"
                for entry in files["rank_map"])
            evidence = {"owned_path": owned, "summary_path": str(summary), "sha256": {}}
            for name, payload in files.items():
                path = folder / (f"{name}.txt" if name == "rank_report" else f"{name}.json")
                path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
                evidence[name] = str(path.relative_to(self.root))
                evidence["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            unit.update(cleanup="completed", attempts=[{"id": "attempt-1", "state": "completed",
                "cleanup": "completed", "evidence": evidence}])
        run_hash = plots.fingerprint({"domain": "communication-pilot", "plan": plan_hash,
                                      "inventory": inventory})
        restoration = self.root / "state-transitions" / "restoration.json"
        restoration.parent.mkdir()
        restoration.write_text(json.dumps({"run_fingerprint": run_hash,
            "baseline": inventory["restoration_baseline"],
            "restored": inventory["restoration_baseline"],
            "netbench_off": {"anjuna2": True, "anjuna3": True},
            "watchdog_released": True}))
        (self.root / "manifest.json").write_text(json.dumps({"domain": "communication-pilot",
            "units": units, "fingerprint": run_hash, "restoration": "completed",
            "restoration_evidence": {"path": str(restoration.relative_to(self.root)),
                "sha256": hashlib.sha256(restoration.read_bytes()).hexdigest()}}))

    def test_raw_pilot_produces_direction_and_backend_figures(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        manifest = json.loads((self.root / "plots" / "plot_manifest.json").read_text())
        self.assertEqual(manifest["measurements"], 8)
        self.assertEqual([Path(name).name for name in manifest["plots"]],
                         ["synthetic_read.png", "synthetic_write.png", "backend_ratio.png"])
        self.assertEqual(manifest["evidence_status"], ["calibration_only"])
        self.assertFalse((self.root / "analysis").exists())

    def test_missing_native_attempt_rejects_plotting(self):
        next(self.root.glob("attempts/*/*/native.json")).unlink()
        self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertFalse((self.root / "plots").exists())


if __name__ == "__main__":
    unittest.main()
