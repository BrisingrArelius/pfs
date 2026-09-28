"""Placement figures derive directly from native IOR, layouts and target state."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
sys.path.insert(0, str(HERE))
import run_placement as runner
import visualize_results as plots


class RawPlacementPlots(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = runner.load_config(HERE / "placement_config.json")
        units = runner.pilot_units(config)
        (self.root / "owner.json").write_text(json.dumps({"run_id": self.root.name,
                                                            "domain": "placement-pilot"}))
        plan_hash = plots.fingerprint({"domain": "placement-pilot", "config": config,
                                       "units": units})
        (self.root / "plan.json").write_text(json.dumps({"domain": "placement-pilot",
            "config": config, "units": units, "fingerprint": plan_hash}))
        hdd = {101, 102, 103, 201, 202, 203, 204, 301, 302, 303, 304, 401, 402, 403}
        targets = [{"id": 100 * host + offset, "oss": f"colva{host}",
                    "media": "HDD" if 100 * host + offset in hdd else "NVMe",
                    "state": "Online/Good"}
                   for host in range(1, 5) for offset in range(1, 8)]
        inventory = {"domain": "placement", "reviewed": True,
                     "targets": targets, "pool_baseline": "reviewed", "namespace": "/mnt/beegfs/owned",
                     "restoration_baseline": {"pool": "original", "chooser": "original"},
                     "tools": {"mpirun": "/usr/bin/mpirun", "ior": "/usr/bin/ior",
                               "ior_version": "pinned",
                               "mpi_binding_pattern": r"^rank (?P<rank>\d+) host (?P<host>\w+) core (?P<core>\d+)$"}}
        (self.root / "inventory.json").write_text(json.dumps(inventory))
        capacity = {str(item["id"]): {"free_bytes": 1000000, "free_percent": 50,
                    "free_inodes": 100000, "byte_class": "Normal", "inode_class": "Normal"}
                    for item in targets}
        for unit in units:
            directory = self.root / "attempts" / unit["id"] / "attempt-1"
            directory.mkdir(parents=True)
            hosts, per_host = runner.CONCURRENCY[unit["concurrency"]]
            ranks = len(hosts) * per_host
            operation = "read" if "read" in unit["workload"] else "write"
            random = unit["workload"].startswith("rand_")
            fpp = unit["workload"].endswith("fpp")
            transfer = 4096 if random else 1048576
            block = config["block_bytes"]
            mib = 16384 * ranks
            throughput = mib / 30
            owned = inventory["namespace"] + "/file"
            summary = directory / "native.json"
            native = {"Version": "pinned", "tests": [{"Parameters": {
                "api": "POSIX", "testFileName": owned, "deadlineForStonewall": 60,
                "blockSize": block, "transferSize": transfer, "segmentCount": 1,
                "repetitions": 1, "filePerProc": int(fpp), "fsync": int(operation == "write"),
                "randomOffset": int(random),
                "useExistingTestFile": int(operation == "read" or unit["workload"] == "rand_write_fpp"),
                "readFile": int(operation == "read"), "writeFile": int(operation == "write")},
                "Options": {"tasks": ranks, "Results": [{"access": operation, "bwMiB": throughput,
                    "iops": throughput * 1048576 / transfer, "totalTime": 30,
                    "wrRdTime": 29.9}]}}], "summary": [{"operation": operation,
                "API": "POSIX", "numTasks": ranks, "blockSize": block,
                "transferSize": transfer, "segmentCount": 1, "repetitions": 1,
                "filePerProc": int(fpp), "xsizeMiB": mib, "MeanTime": 30,
                "bwMeanMIB": throughput}]}
            eligible = runner.eligible_targets(inventory, unit["storage"])
            layout = (eligible[:1] if unit["stripes"] == 1 else eligible[:4])
            state = {"during": {"chooser": unit["chooser"], "eligible_targets": eligible,
                        "netbench": {client: 0 for client in runner.CLIENTS},
                        "restoration_watchdog_active": True},
                     "after": inventory["restoration_baseline"], "watchdog_verified": True}
            telemetry = {"netbench_off": True, "telemetry_complete": True,
                         "synchronized_write": True, "pool_restored": True,
                         "chooser_restored": True,
                         "actual_layouts": [layout for _ in range(ranks if fpp else 1)],
                         "rank_results": [{"rank": rank, "bytes": block, "transfer_seconds": 29.9}
                                          for rank in range(ranks)],
                         "target_capacity_before": capacity, "target_capacity_after": capacity,
                         "random_existing_file_verified": True,
                         "identity_before": "file-1", "identity_after": "file-1",
                         "size_before": block, "size_after": block}
            files = {"native": native, "state": state, "telemetry": telemetry,
                     "rank_map": [{"rank": rank, "host": client, "core": index}
                                  for rank, (client, index) in enumerate((client, index)
                                      for client in hosts for index in range(per_host))],
                     "command": runner.build_command(unit, "/usr/bin/mpirun", "/usr/bin/ior",
                                                     owned, summary),
                     "exit": {"returncode": 0, "timeout": False}}
            files["rank_report"] = "".join(
                f"rank {entry['rank']} host {entry['host']} core {entry['core']}\n"
                for entry in files["rank_map"])
            evidence = {"owned_path": owned, "summary_path": str(summary), "sha256": {}}
            for name, payload in files.items():
                path = directory / (f"{name}.txt" if name == "rank_report" else f"{name}.json")
                path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
                evidence[name] = str(path.relative_to(self.root))
                evidence["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            unit.update(cleanup="completed", attempts=[{"id": "attempt-1", "state": "completed",
                "cleanup": "completed", "evidence": evidence}])
        run_hash = plots.fingerprint({"domain": "placement-pilot", "plan": plan_hash,
                                      "inventory": inventory})
        restoration = self.root / "state-transitions" / "restoration.json"
        restoration.parent.mkdir()
        restoration.write_text(json.dumps({"run_fingerprint": run_hash,
            "baseline": inventory["restoration_baseline"],
            "restored": inventory["restoration_baseline"],
            "netbench_off": {"anjuna2": True, "anjuna3": True},
            "watchdog_released": True}))
        (self.root / "manifest.json").write_text(json.dumps({"domain": "placement-pilot",
            "units": units, "fingerprint": run_hash, "restoration": "completed",
            "restoration_evidence": {"path": str(restoration.relative_to(self.root)),
                "sha256": hashlib.sha256(restoration.read_bytes()).hexdigest()}}))

    def test_pilot_keeps_workloads_and_comparisons_in_separate_figures(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        report = json.loads((self.root / "plots" / "plot_manifest.json").read_text())
        self.assertEqual(report["measurements"], 6)
        self.assertEqual(len(report["plots"]), 6)
        self.assertTrue(all((self.root / "plots" / name).is_file() for name in report["plots"]))
        self.assertFalse((self.root / "analysis").exists())

    def test_invalid_layout_blocks_figures(self):
        telemetry = next(self.root.glob("attempts/*/*/telemetry.json"))
        data = json.loads(telemetry.read_text())
        data["actual_layouts"] = [[999]]
        telemetry.write_text(json.dumps(data))
        self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertFalse((self.root / "plots").exists())


if __name__ == "__main__":
    unittest.main()
