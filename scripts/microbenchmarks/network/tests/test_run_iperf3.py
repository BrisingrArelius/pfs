"""Pure protocol, planning, command and native-JSON tests for the iperf3 runner."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


NETWORK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NETWORK_DIR))
import run_iperf3 as runner


def confirmed_inventory():
    inventory = json.loads((NETWORK_DIR / "network_inventory.json").read_text())
    inventory["confirmed"] = True
    inventory["confirmed_at"] = "2026-09-22T00:00:00+00:00"
    inventory["evidence"] = "synthetic-test"
    addresses = {
        "anjuna2": "10.1.19.73", "anjuna3": "10.1.19.74",
        "colva1": "192.168.0.2", "colva2": "10.1.19.76",
        "colva3": "10.1.19.77", "colva4": "192.168.0.5",
    }
    for host, item in inventory["hosts"].items():
        item["clock_synchronized"] = True
    for client in runner.CLIENTS:
        for server in runner.SERVERS:
            path = inventory["paths"][client][server]
            path["source_address"] = addresses[client]
            path["source_interface"] = "client0"
            path["source_mtu"] = 1500
            path["source_link_mbps"] = 10000
            path["destination_address"] = addresses[server]
            path["destination_interface"] = "storage0"
            path["destination_mtu"] = 1500
            path["destination_link_mbps"] = 2500
            path["evidence_level"] = "confirmed"
    return inventory


def payload(member, config, perspective="client"):
    if perspective == "client":
        connected = [{"local_host": member["source_address"],
                      "remote_host": member["destination_address"],
                      "remote_port": member["port"]} for _ in range(member["streams"])]
    else:
        connected = [{"local_host": member["destination_address"],
                      "local_port": member["port"],
                      "remote_host": member["source_address"]} for _ in range(member["streams"])]
    streams = [{"sender": {"bits_per_second": 1e9},
                "receiver": {"bits_per_second": 0.99e9}} for _ in range(member["streams"])]
    intervals = [{"sum": {"seconds": config["interval_seconds"],
                           "bits_per_second": 2.34e9,
                           "omitted": index < config["omit_seconds"]}}
                 for index in range(int((config["omit_seconds"] + config["duration_seconds"])
                                        / config["interval_seconds"]))]
    sent = {"bits_per_second": 2.35e9, "bytes": 8_812_500_000,
            "seconds": 30.0, "retransmits": 3}
    received = {"bits_per_second": 2.34e9, "bytes": 8_775_000_000,
                "seconds": 30.01}
    if perspective == "server" and member["direction"] == "client_to_oss":
        sent = {"bits_per_second": 0, "bytes": 0, "seconds": 30.01}
    if perspective == "server" and member["direction"] == "oss_to_client":
        received = {"bits_per_second": 0, "bytes": 0, "seconds": 30.01}
    return {
        "start": {"connected": connected, "test_start": {
            "protocol": "TCP", "num_streams": member["streams"],
            "omit": config["omit_seconds"], "duration": config["duration_seconds"],
            "reverse": 1 if member["direction"] == "oss_to_client" else 0,
        }},
        "intervals": intervals,
        "end": {
            "streams": streams,
            "sum_sent": sent,
            "sum_received": received,
            "cpu_utilization_percent": {"host_total": 12.5, "remote_total": 8.5},
        },
    }


class RunnerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((NETWORK_DIR / "network_config.json").read_text())
        self.inventory = confirmed_inventory()

    def test_unconfirmed_inventory_is_rejected(self):
        inventory = json.loads((NETWORK_DIR / "network_inventory.json").read_text())
        inventory["confirmed"] = False
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            runner.validate_inventory(inventory)

    def test_full_plan_has_190_units_and_320_path_sessions(self):
        runner.validate_config(self.config)
        runner.validate_inventory(self.inventory)
        units = runner.plan_units(self.config, self.inventory)
        self.assertEqual(len(units), 190)
        self.assertEqual(sum(len(unit["members"]) for unit in units), 320)
        self.assertEqual(sum(unit["mode"] == "isolated" for unit in units), 160)
        self.assertEqual(sum(unit["mode"] == "one_client_four_oss" for unit in units), 20)
        self.assertEqual(sum(unit["mode"] == "two_clients_four_oss" for unit in units), 10)
        self.assertEqual(len({unit["id"] for unit in units}), 190)

    def test_plan_is_reproducible_and_dual_ports_are_distinct(self):
        first = runner.plan_units(self.config, self.inventory)
        second = runner.plan_units(self.config, self.inventory)
        self.assertEqual(first, second)
        dual = next(unit for unit in first if unit["mode"] == "two_clients_four_oss")
        for server in runner.SERVERS:
            ports = {member["port"] for member in dual["members"] if member["server"] == server}
            self.assertEqual(ports, {5201, 5202})

    def test_pilot_is_four_isolated_anjuna2_colva2_cases(self):
        units = runner.plan_units(self.config, self.inventory, pilot=True)
        self.assertEqual(len(units), 4)
        self.assertEqual(sum(len(unit["members"]) for unit in units), 4)
        self.assertEqual({member["client"] for unit in units for member in unit["members"]}, {"anjuna2"})
        self.assertEqual({member["server"] for unit in units for member in unit["members"]}, {"colva2"})

    def test_commands_bind_fixed_addresses_and_reverse_only_data_direction(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        member = copy.deepcopy(member)
        member["direction"] = "oss_to_client"
        server, client = runner.build_commands(member, self.config)
        self.assertEqual(server[:4], ["iperf3", "--server", "--one-off", "--json"])
        self.assertIn(member["destination_address"], server)
        self.assertIn(member["source_address"], client)
        self.assertEqual(client[-1], "--reverse")

    def test_remote_bash_script_is_shell_quoted_as_one_ssh_command(self):
        script = "set -u; hostname -s; printf '%s\\n' safe"
        with mock.patch.object(runner, "local_aliases", return_value={"anjuna3"}):
            command = runner.command_for_host(
                "anjuna2", ["bash", "-c", script], self.inventory, self.config)
        self.assertEqual(command[-1], runner.shlex.join(["bash", "-c", script]))
        self.assertEqual(command[-4:-1], ["-o", "ConnectTimeout=10", "anjuna2"])

    def test_explicit_route_source_may_be_reported_as_from(self):
        route = {"dst": "192.168.0.2", "from": "192.168.0.6", "dev": "eno1"}
        self.assertTrue(runner.route_matches(route, "eno1", "192.168.0.6"))
        self.assertFalse(runner.route_matches(route, "enp4s0", "192.168.0.6"))

    def test_remote_artifact_transfer_requests_only_fixed_files(self):
        process = {"host": "anjuna2", "remote_dir": "/tmp/fixed", "role": "client"}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "copied"
            completed = subprocess.CompletedProcess([], 0, "", "")
            with (mock.patch.object(runner, "local_aliases", return_value={"anjuna3"}),
                  mock.patch.object(runner.subprocess, "run", return_value=completed) as run):
                runner.copy_process_artifacts(
                    process, destination, self.inventory, self.config)
            command = run.call_args.args[0]
        self.assertNotIn("anjuna2:/tmp/fixed/.", command)
        for name in runner.PROCESS_FILES:
            self.assertIn(f"anjuna2:/tmp/fixed/{name}", command)

    def test_native_json_validates_forward_and_reverse(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        for direction in self.config["directions"]:
            current = copy.deepcopy(member)
            current["direction"] = direction
            result = runner.validate_iperf_payload(payload(current, self.config), current, self.config)
            self.assertEqual(result["receiver_bits_per_second"], 2.34e9)
            self.assertEqual(result["retransmits"], 3)
            runner.validate_iperf_payload(payload(current, self.config, "server"),
                                          current, self.config, "server")

    def test_server_requires_only_its_direction_active_aggregate(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        for direction, active_field in (("client_to_oss", "sum_received"),
                                        ("oss_to_client", "sum_sent")):
            current = copy.deepcopy(member)
            current["direction"] = direction
            valid = payload(current, self.config, "server")
            runner.validate_iperf_payload(valid, current, self.config, "server")
            valid["end"][active_field]["bytes"] = 0
            with self.assertRaisesRegex(ValueError, "server .* throughput"):
                runner.validate_iperf_payload(valid, current, self.config, "server")

    def test_native_json_rejects_wrong_endpoint_stream_and_duration(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        wrong = payload(member, self.config)
        wrong["start"]["connected"][0]["remote_host"] = "192.0.2.1"
        with self.assertRaisesRegex(ValueError, "endpoints differ"):
            runner.validate_iperf_payload(wrong, member, self.config)
        wrong = payload(member, self.config)
        wrong["end"]["sum_received"]["seconds"] = 20
        with self.assertRaisesRegex(ValueError, "duration"):
            runner.validate_iperf_payload(wrong, member, self.config)

    def test_native_json_rejects_missing_intervals_and_malformed_server_cpu(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        wrong = payload(member, self.config)
        wrong["intervals"] = []
        with self.assertRaisesRegex(ValueError, "interval evidence"):
            runner.validate_iperf_payload(wrong, member, self.config)
        wrong = payload(member, self.config, "server")
        wrong["end"]["cpu_utilization_percent"] = []
        with self.assertRaisesRegex(ValueError, "CPU utilization"):
            runner.validate_iperf_payload(wrong, member, self.config, "server")

    def test_barrier_wait_can_be_cancelled_before_child_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "attempt" / "client"
            process = subprocess.Popen(
                ["bash", "-s", "--", str(directory), "60", str(time.time() + 30),
                 "sleep", "30"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            process.stdin.write(runner.REMOTE_PROCESS_SCRIPT)
            process.stdin.close()
            deadline = time.monotonic() + 3
            while not (directory / "identity").is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((directory / "identity").is_file())
            identity = runner.parse_identity(directory / "identity")
            self.assertEqual(identity["pid"], "0")
            cleanup = subprocess.run(
                ["bash", "-s", "--", str(directory)], input=runner.CLEANUP_SCRIPT,
                text=True, capture_output=True, timeout=10, check=False)
            self.assertEqual(cleanup.returncode, 0, cleanup.stderr)
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()

    def test_launch_handoff_failure_reaps_child_and_attempts_remote_cleanup(self):
        class BrokenInput:
            def write(self, _value):
                raise BrokenPipeError("synthetic handoff failure")

            def close(self):
                pass

        class FakeProcess:
            pid = 12345
            stdin = BrokenInput()

            def wait(self):
                return 1

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            with (mock.patch.object(runner.subprocess, "Popen", return_value=FakeProcess()),
                  mock.patch.object(runner.os, "killpg") as killpg,
                  mock.patch.object(runner, "remote_cleanup") as cleanup,
                  mock.patch.object(runner, "remove_remote_artifacts") as remove):
                with self.assertRaises(BrokenPipeError):
                    runner.start_process("anjuna2", "client", ["iperf3"],
                                         "/tmp/pfs-network-bench/run/unit/attempt/member/client",
                                         artifact, self.inventory, self.config)
            killpg.assert_called_once_with(12345, runner.signal.SIGKILL)
            cleanup.assert_called_once()
            remove.assert_called_once()

    def test_interrupted_attempt_cleanup_is_recovered_without_traffic(self):
        member = runner.plan_units(self.config, self.inventory, pilot=True)[0]["members"][0]
        attempt = {"state": "interrupted", "cleanup": "failed", "members": [{
            "member": member, "server_remote": "/tmp/server", "client_remote": "/tmp/client",
        }]}
        run = {"config": self.config, "inventory": self.inventory, "manifest": {}}
        with (mock.patch.object(runner, "remote_cleanup") as cleanup,
              mock.patch.object(runner, "remove_remote_artifacts") as remove,
              mock.patch.object(runner, "save")):
            runner.recover_attempt_cleanup(run, attempt)
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(remove.call_count, 2)
        self.assertEqual(attempt["cleanup"], "completed")


if __name__ == "__main__":
    unittest.main()
