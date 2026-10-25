"""The network visualizer consumes raw iperf3 attempts directly."""

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


NETWORK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NETWORK_DIR))
import visualize_results as plotter
import run_iperf3 as runner
from test_run_iperf3 import confirmed_inventory, payload


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "run"
        self.output = self.root / "plots"
        self.root.mkdir()
        self.config = json.loads((NETWORK_DIR / "network_config.json").read_text())
        self.inventory = confirmed_inventory()
        units = runner.plan_units(self.config, self.inventory, pilot=True)
        hosts = set(runner.CLIENTS + runner.SERVERS)
        for unit in units:
            relative = f"raw/{unit['id']}/attempt-1"
            attempt = {"id": 1, "state": "completed", "cleanup": "completed",
                       "artifacts": relative, "launch_skew_seconds": 0.0,
                       "release_epoch": 0.0, "maximum_release_lateness_seconds": 0.0,
                       "clock_observations": {unit["members"][0]["client"]: {
                           "offset_seconds": 0.0, "uncertainty_seconds": 0.0}}, "members": []}
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

    def test_complete_raw_evidence_produces_isolated_figure(self):
        result = plotter.main([str(self.root)])
        self.assertEqual(result, 0)
        report = json.loads((self.output / "plot_manifest.json").read_text())
        self.assertEqual((report["measurements"], report["epochs"]), (4, 4))
        self.assertEqual([Path(name).name for name in report["plots"]], ["isolated_throughput.png"])
        self.assertTrue((self.output / report["plots"][0]).is_file())

    def test_pending_unit_cannot_be_plotted_as_complete(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"][0]["attempts"][0]["state"] = "interrupted"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertFalse((self.output / "plot_manifest.json").exists())

    def test_corrupt_native_json_is_a_validation_error(self):
        native = next(self.root.glob("raw/**/client/output.json"))
        native.write_text("not JSON")
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_incomplete_cleanup_is_a_validation_error(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"][0]["attempts"][0]["cleanup"] = "failed"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_truncated_canonical_plan_is_rejected(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["units"].pop()
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_unowned_plot_symlink_is_not_followed(self):
        self.output.mkdir()
        victim = self.root / "unrelated"
        victim.write_text("preserve")
        (self.output / "isolated_throughput.png").symlink_to(victim)
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(victim.read_text(), "preserve")

    def test_failed_second_plot_leaves_no_figures_and_retry_succeeds(self):
        with mock.patch.object(plotter, "plot_concurrent_epochs",
                               side_effect=ValueError("simulated rendering failure")):
            self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(list(self.output.glob("*.png")), [])
        self.assertEqual(list(self.output.glob(".plot-stage-*")), [])
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 0)

    def test_manifest_failure_restores_existing_isolated_plot(self):
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 0)
        before = {str(path.relative_to(self.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.output.rglob("*") if path.is_file()}
        replace = plotter.os.replace

        def fail_manifest(source, destination):
            if Path(source).name.startswith(".plot-manifest-"):
                raise OSError("simulated manifest publication failure")
            return replace(source, destination)

        with mock.patch.object(plotter.os, "replace", side_effect=fail_manifest):
            self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(before, {str(path.relative_to(self.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.output.rglob("*") if path.is_file()})

    def test_command_provenance_corruption_is_a_validation_error(self):
        commands = next(self.root.glob("raw/**/commands.json"))
        value = json.loads(commands.read_text())
        value["client"].append("--bidir")
        commands.write_text(json.dumps(value))
        self.assertEqual(plotter.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_concurrent_epoch_with_excessive_launch_skew_is_rejected(self):
        concurrent = next(unit for unit in runner.plan_units(self.config, self.inventory)
                          if unit["mode"] == "one_client_four_oss")
        folder = self.root / "raw" / "concurrent-attempt"
        folder.mkdir(parents=True)
        attempt = {"artifacts": "raw/concurrent-attempt", "members": [],
                   "release_epoch": 0.0,
                   "clock_observations": {concurrent["members"][0]["client"]: {
                       "offset_seconds": 0.0, "uncertainty_seconds": 0.0}}}
        endpoints = {(member["client"], member["source_interface"])
                     for member in concurrent["members"]}
        endpoints |= {(member["server"], member["destination_interface"])
                      for member in concurrent["members"]}
        for host, interface in endpoints:
            for phase in ("before", "after"):
                (folder / f"telemetry-{phase}-{host}-{interface}.json").write_text(json.dumps(
                    [{"ifname": interface}]))
        for index, member in enumerate(concurrent["members"]):
            path = folder / member["id"] / "client"
            path.mkdir(parents=True)
            (path / "identity").write_text(
                "supervisor_pid=1\nsupervisor_start=1\nbarrier_pid=0\nbarrier_start=0\n"
                "pid=1\npgid=1\nstart=1\nboot=test\nuid=1\nrelease_epoch=0\n"
                f"launched_epoch_ns={5_000_000_000 if index == 1 else 1}\n")
            attempt["members"].append({"member": member,
                                       "artifacts": f"raw/concurrent-attempt/{member['id']}"})
        with mock.patch.object(runner, "validate_member_artifacts", return_value={"receiver_bits_per_second": 1}):
            with self.assertRaisesRegex(ValueError, "launch skew"):
                runner.revalidate_attempt_evidence(self.root, concurrent, attempt, self.config)


if __name__ == "__main__":
    unittest.main()
