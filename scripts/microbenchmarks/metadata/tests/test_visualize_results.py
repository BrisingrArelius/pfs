"""mdtest visualizer reads all seven native phases from raw attempts."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[2] / "results" / "microbenchmarks" / "runs"
sys.path.insert(0, str(HERE))
import run_mdtest as runner
import visualize_results as plots


class RawMdtestPlots(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = runner.load_config(HERE / "metadata_config.json")
        units = runner.pilot_units(config)
        (self.root / "owner.json").write_text(json.dumps({"run_id": self.root.name,
                                                            "domain": "metadata-pilot"}))
        plan_hash = plots.fingerprint({"domain": "metadata-pilot", "config": config, "units": units})
        (self.root / "plan.json").write_text(json.dumps({"domain": "metadata-pilot",
            "config": config, "units": units, "fingerprint": plan_hash}))
        schema = {"version_signature": r"^mdtest pinned$", "task_signature": r"^tasks={value}$",
                  "item_signature": r"^items={value}$", "path_signature": r"^workdir={value}$",
                  "phases": {phase: rf"^{phase}: (?P<operations>\d+) (?P<elapsed>\d+) (?P<rate>\d+)$"
                             for phase in runner.PHASES}}
        inventory = {"domain": "metadata", "reviewed": True, "clients": {"a": 1},
                     "meta_target": "reviewed", "mount": "/mnt/beegfs",
                     "restoration_baseline": {"mount": "/mnt/beegfs"},
                     "namespace": "/mnt/beegfs/owned",
                     "allowed_cores": {host: list(range(16)) for host in runner.CLIENTS},
                     "netbench_off": True, "shared_mount_verified": True,
                     "tools": {"mpirun": "/usr/bin/mpirun", "mdtest": "/usr/bin/mdtest",
                               "mpi_binding_pattern": r"^rank (?P<rank>\d+) host (?P<host>\w+) core (?P<core>\d+)$",
                               "mdtest_schema": schema}}
        (self.root / "inventory.json").write_text(json.dumps(inventory))
        for unit in units:
            directory = self.root / "attempts" / unit["id"] / "attempt-1"
            directory.mkdir(parents=True)
            clients = runner.CLIENTS if unit["placement"] == "dual" else (unit["placement"],)
            tasks = len(clients) * unit["ranks_per_client"]
            owned = inventory["namespace"] + "/attempt"
            operations = tasks * 100000
            native = (f"mdtest pinned\ntasks={tasks}\nitems=100000\nworkdir={owned}\n"
                      + "".join(f"{phase}: {operations} 4 {operations // 4}\n"
                                for phase in runner.PHASES))
            rank_map = [{"rank": rank, "host": client, "core": index}
                        for rank, (client, index) in enumerate((client, index)
                            for client in clients for index in range(unit["ranks_per_client"]))]
            data = {"native": native, "command": runner.build_command(unit, "/usr/bin/mpirun",
                     "/usr/bin/mdtest", owned), "exit": {"returncode": 0, "timeout": False},
                    "telemetry": {"telemetry_complete": True, "netbench_off": True,
                                  "mount_unchanged": True}, "rank_map": rank_map,
                    "rank_report": "".join(
                        f"rank {entry['rank']} host {entry['host']} core {entry['core']}\n"
                        for entry in rank_map)}
            evidence = {"owned_path": owned, "sha256": {}}
            for name, payload in data.items():
                path = directory / f"{name}.txt" if isinstance(payload, str) else directory / f"{name}.json"
                path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
                evidence[name] = str(path.relative_to(self.root))
                evidence["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            unit.update(cleanup="completed", attempts=[{"id": "attempt-1", "state": "completed",
                "cleanup": "completed", "evidence": evidence}])
        run_hash = plots.fingerprint({"domain": "metadata-pilot", "plan": plan_hash,
                                      "inventory": inventory})
        restoration = self.root / "state-transitions" / "restoration.json"
        restoration.parent.mkdir()
        restoration.write_text(json.dumps({"run_fingerprint": run_hash,
            "baseline": inventory["restoration_baseline"],
            "restored": inventory["restoration_baseline"],
            "netbench_off": {"anjuna2": True, "anjuna3": True},
            "watchdog_released": True}))
        (self.root / "manifest.json").write_text(json.dumps({"domain": "metadata-pilot",
            "units": units, "fingerprint": run_hash, "restoration": "completed",
            "restoration_evidence": {"path": str(restoration.relative_to(self.root)),
                "sha256": hashlib.sha256(restoration.read_bytes()).hexdigest()}}))

    def test_seven_native_phases_get_separate_figures(self):
        self.assertEqual(plots.main([str(self.root)]), 0)
        report = json.loads((self.root / "plots" / "plot_manifest.json").read_text())
        self.assertEqual(report["phase_rows"], 4 * 7)
        self.assertEqual([Path(name).name for name in report["plots"]],
                         [f"{phase}.png" for phase in runner.PHASES])
        self.assertTrue(all((self.root / "plots" / name).is_file() for name in report["plots"]))
        self.assertFalse((self.root / "analysis").exists())

    def test_missing_phase_is_rejected(self):
        native = next(self.root.glob("attempts/*/*/native.txt"))
        native.write_text(native.read_text().replace("file_read:", "missing_read:"))
        self.assertEqual(plots.main([str(self.root)]), 1)
        self.assertFalse((self.root / "plots").exists())


if __name__ == "__main__":
    unittest.main()
