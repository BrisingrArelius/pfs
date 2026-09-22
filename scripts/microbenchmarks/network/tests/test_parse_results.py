"""Read-only parser tests using synthetic native iperf3 artifacts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


NETWORK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NETWORK_DIR))
import parse_results as parser
import run_iperf3 as runner
from test_run_iperf3 import confirmed_inventory, payload


class ParserTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "run"
        self.output = self.root / "analysis"
        self.root.mkdir()
        self.config = json.loads((NETWORK_DIR / "network_config.json").read_text())
        self.inventory = confirmed_inventory()
        units = runner.plan_units(self.config, self.inventory, pilot=True)
        hosts = set(runner.CLIENTS + runner.SERVERS)
        for unit in units:
            relative = f"raw/{unit['id']}/attempt-1"
            attempt = {"id": 1, "state": "completed", "cleanup": "completed",
                       "artifacts": relative, "launch_skew_seconds": 0.0, "members": []}
            folder = self.root / relative
            folder.mkdir(parents=True)
            member = unit["members"][0]
            involved = {(member["client"], member["source_interface"]),
                        (member["server"], member["destination_interface"])}
            for host, interface in involved:
                for phase in ("before", "after"):
                    base = 100 if phase == "before" else 200
                    telemetry = [{"ifname": interface, "stats64": {
                        "rx": {"bytes": base, "packets": base, "errors": 0, "dropped": 0},
                        "tx": {"bytes": base, "packets": base, "errors": 0, "dropped": 0},
                    }}]
                    (folder / f"telemetry-{phase}-{host}-{interface}.json").write_text(json.dumps(telemetry))
            member_folder = folder / member["id"]
            member_folder.mkdir(parents=True)
            (member_folder / "member.json").write_text(json.dumps(member))
            commands = dict(zip(("server", "client"), runner.build_commands(member, self.config)))
            (member_folder / "commands.json").write_text(json.dumps(commands))
            for role in ("client", "server"):
                role_folder = member_folder / role
                role_folder.mkdir()
                (role_folder / "output.json").write_text(
                    json.dumps(payload(member, self.config, role)))
                (role_folder / "stderr").write_text("")
                (role_folder / "exit_status").write_text("0\n")
                (role_folder / "argv").write_text(" ".join(commands[role]) + "\n")
                (role_folder / "identity").write_text(
                    "supervisor_pid=1\nsupervisor_start=1\nbarrier_pid=0\nbarrier_start=0\n"
                    "pid=1\npgid=1\nstart=1\n"
                    "boot=test\nuid=1\nrelease_epoch=1\nlaunched_epoch_ns=1\n")
            attempt["members"].append({"member": member,
                                       "artifacts": f"{relative}/{member['id']}"})
            unit["attempts"] = [attempt]
        observations = {host: {"iperf3_version": "iperf 3.test"} for host in hosts}
        fingerprint, inventory_fingerprint = runner.fingerprints(
            self.config, self.inventory, observations)
        self.manifest = {
            "run_id": "a" * 32, "mode": "pilot", "config": self.config,
            "inventory": self.inventory, "fingerprint": fingerprint,
            "inventory_fingerprint": inventory_fingerprint,
            "tool_observations": observations,
            "units": units, "sessions": [{"outcome": "completed"}],
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_complete_evidence_emits_valid_csv_and_markdown(self):
        result = parser.main([str(self.root), "--output-dir", str(self.output)])
        self.assertEqual(result, 0)
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertEqual((report["measurements"], report["epochs"]), (4, 4))
        self.assertEqual(report["errors"], [])
        self.assertIn("Validation: **PASS**", (self.output / "summary.md").read_text())

    def test_pending_unit_is_reported_without_rerunning(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"][0]["attempts"][0]["state"] = "interrupted"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        result = parser.main([str(self.root), "--output-dir", str(self.output)])
        self.assertEqual(result, 2)
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertTrue(report["errors"])

    def test_corrupt_native_json_is_a_validation_error(self):
        native = next(self.root.glob("raw/**/client/output.json"))
        native.write_text("not JSON")
        result = parser.main([str(self.root), "--output-dir", str(self.output)])
        self.assertEqual(result, 2)
        self.assertIn("Expecting value", (self.output / "summary.md").read_text())

    def test_incomplete_cleanup_is_a_validation_error(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"][0]["attempts"][0]["cleanup"] = "failed"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(parser.main([str(self.root), "--output-dir", str(self.output)]), 2)
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertTrue(any("cleanup" in error for error in report["errors"]))

    def test_truncated_canonical_plan_is_rejected(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"].pop()
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(parser.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_command_provenance_corruption_is_a_validation_error(self):
        commands = next(self.root.glob("raw/**/commands.json"))
        value = json.loads(commands.read_text())
        value["client"].append("--bidir")
        commands.write_text(json.dumps(value))
        self.assertEqual(parser.main([str(self.root), "--output-dir", str(self.output)]), 2)
        self.assertIn("command provenance differs", (self.output / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
