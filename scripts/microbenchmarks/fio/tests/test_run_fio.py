"""Lifecycle checks with tiny temporary files and a fake FIO, never real devices."""

import configparser
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_fio as runner
from run_support import BudgetExpired, atomic_json, read_json


class RunnerTests(unittest.TestCase):
    """Exercise the real orchestration while replacing every cluster operation."""

    def setUp(self):
        """Create an isolated fake host; FIO and findmnt cannot execute."""
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "results"
        self.config = read_json(runner.HERE / "fio_config.json")
        self.config["fio"]["size"] = 1048576
        self.config["planning"]["prepare_seconds"] = {"hdd": 1, "nvme": 1}
        self.config["planning"]["measurement_seconds"] = {
            media: {workload["name"]: 1 for workload in self.config["workloads"]}
            for media in ("hdd", "nvme")}
        self.targets = [dict(target_id=101, media="hdd", mount=str(self.base / "hdd"), device="/dev/fake1"),
                        dict(target_id=104, media="nvme", mount=str(self.base / "nvme"), device="/dev/fake2")]
        for target in self.targets:
            Path(target["mount"]).mkdir()
        self.write_config()
        atomic_json(self.base / "target_inventory.json", {"colva1": self.targets})
        self.calls = []
        self.fail_at = None
        self.failure = BudgetExpired("fake reservation end")
        self.real_check_target = runner.check_target
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "HERE", self.base))
        self.stack.enter_context(patch.object(runner.socket, "gethostname", return_value="colva1"))
        self.stack.enter_context(patch.object(runner.subprocess, "run", side_effect=self.metadata_command))
        self.stack.enter_context(patch.object(runner, "check_target", side_effect=self.snapshot))
        self.stack.enter_context(patch.object(runner, "run_command", side_effect=self.fake_fio))
        self.stack.enter_context(patch.object(runner.signal, "signal"))

    def write_config(self):
        """Persist this fixture's config without modifying the project config."""
        atomic_json(self.base / "fio_config.json", self.config)

    def metadata_command(self, argv, **kwargs):
        """Reject all external probes except the fake version query."""
        self.assertEqual(argv, ["fio", "--version"])
        return subprocess.CompletedProcess(argv, 0, stdout="fio-3.39\n", stderr="")

    def snapshot(self, target):
        """Return ample fake capacity, without inspecting any actual mount/device."""
        return {"free_bytes": 10**12, "free_inodes": 10000, "total_bytes": 2 * 10**12,
                "free_percent": 50, "mount": target["mount"], "diskstats": "fake"}

    def fake_fio(self, argv, stdout, stderr, deadline, cleanup, timeout, cwd=None):
        """Simulate accounting and sparse fixture-file growth, not benchmark I/O."""
        options = configparser.ConfigParser(interpolation=None)
        options.read(argv[-1])
        name = options.sections()[0]
        job = options[name]
        path = Path(job["filename"])
        self.assertTrue(path.is_relative_to(self.base))
        self.assertEqual(Path(cwd), Path(argv[-1]).parent)
        self.assertIn(f"--aux-path={cwd}", argv)
        self.assertEqual(path.name, "data")
        self.assertEqual(path.parent.name, f"target-{name.removeprefix('target-')}")
        self.assertEqual(path.parents[2].name, ".local-fio")
        is_prepare = "runtime" not in job
        self.calls.append((is_prepare, path, job["rw"]))
        Path(stdout).write_text("fake stdout\n")
        Path(stderr).write_text("")
        if self.fail_at == len(self.calls):
            self.fail_at = None
            raise self.failure
        if is_prepare:
            with path.open("xb") as data:
                data.truncate(int(job["size"]))
        else:
            self.assertEqual(job["allow_file_create"], "0")
            self.assertEqual(job["overwrite"], "1")
            self.assertEqual(path.stat().st_size, int(job["size"]))
        operation = "write" if "write" in job["rw"] else "read"
        payload = {"jobs": [{"jobname": name, "error": 0, operation: {
            "io_bytes": int(job["size"]), "runtime": 1000, "total_ios": 8}}]}
        output = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--output="))
        Path(output).write_text(json.dumps(payload))
        return 1.1

    def invoke(self, *extra):
        """Execute the real entry point with fake hardware and capture its messages."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(["--results-dir", str(self.root), "--time-limit", "2h", *extra])

    def manifest(self):
        """Read the actual durable checkpoint generated by the runner."""
        return read_json(self.root / "manifest.json")

    def test_one_preparation_per_target_and_balanced_order(self):
        """Two targets get 50 measurements, two setups and one deletion each."""
        self.assertEqual(self.invoke("--targets", "101,104"), 0)
        manifest = self.manifest()
        self.assertEqual(len(self.calls), 52)
        self.assertEqual(sum(prepare for prepare, _, _ in self.calls), 2)
        for target in self.targets:
            cases = [case for case in manifest["cases"] if case["target_id"] == target["target_id"]]
            rounds = [[case["workload"]["name"] for case in cases if case["repetition"] == repetition]
                      for repetition in range(1, 6)]
            for position in range(5):
                self.assertEqual(len({order[position] for order in rounds}), 5)
            state = manifest["targets"][str(target["target_id"])]
            identities = [case["attempts"][0]["file_before"] for case in cases]
            self.assertTrue(all(identity == identities[0] for identity in identities))
            self.assertFalse(Path(state["path"]).exists())
            self.assertEqual(state["cleanup"], "completed")

    def test_pilot_runs_one_round_and_resume_remembers_mode(self):
        """Pilot covers every workload once per target without weakening full runs."""
        self.fail_at = 3
        self.assertEqual(self.invoke("--targets", "101,104", "--pilot"), 0)
        stopped = self.manifest()
        self.assertEqual(stopped["mode"], "pilot")
        self.assertEqual(len(stopped["cases"]), 10)
        self.assertEqual({case["repetition"] for case in stopped["cases"]}, {1})
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(len(self.calls), 13)  # two preparations, ten completions, one interrupted attempt
        self.assertEqual(sum(prepare for prepare, _, _ in self.calls), 2)

    def test_pilot_cannot_convert_an_existing_full_run(self):
        """Pilot mode is part of scientific compatibility, not a resume shortcut."""
        self.fail_at = 2
        self.invoke("--targets", "101")
        before = len(self.calls)
        self.assertEqual(self.invoke("--resume", "--pilot"), 1)
        self.assertEqual(len(self.calls), before)

    def test_resume_retains_file_and_skips_completed_measurements(self):
        """A stop during the second measurement retries it, not setup or the first."""
        self.fail_at = 3
        self.assertEqual(self.invoke("--targets", "101"), 0)
        before = self.manifest()
        path = Path(before["targets"]["101"]["path"])
        inode = path.stat().st_ino
        self.assertEqual(before["sessions"][-1]["outcome"], "budget_stop")
        self.assertEqual(self.invoke("--resume"), 0)
        after = self.manifest()
        self.assertEqual(sum(prepare for prepare, _, _ in self.calls), 1)
        self.assertEqual(len(after["cases"][0]["attempts"]), 1)
        self.assertEqual(len(after["cases"][1]["attempts"]), 2)
        self.assertEqual(after["cases"][1]["attempts"][-1]["file_before"]["inode"], inode)
        self.assertFalse(path.exists())

    def test_missing_retained_file_is_prepared_once_again(self):
        """A lost dataset does not discard completed measurement evidence."""
        self.fail_at = 3
        self.invoke("--targets", "101")
        Path(self.manifest()["targets"]["101"]["path"]).unlink()
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(sum(prepare for prepare, _, _ in self.calls), 2)
        self.assertEqual(len(self.manifest()["cases"][0]["attempts"]), 1)

    def test_interrupted_preparation_is_not_reused(self):
        """File existence without successful setup evidence is insufficient."""
        self.fail_at = 1
        self.assertEqual(self.invoke("--targets", "101"), 0)
        self.assertEqual(self.invoke("--resume"), 0)
        preparations = self.manifest()["targets"]["101"]["preparations"]
        self.assertEqual([item["state"] for item in preparations], ["interrupted", "completed"])

    def test_hard_timeout_abandons_session(self):
        """Unexpected timeout never retries or proceeds to another measurement."""
        self.fail_at, self.failure = 2, TimeoutError("fake hard timeout")
        self.assertEqual(self.invoke("--targets", "101"), 1)
        self.assertEqual(len(self.calls), 2)
        manifest = self.manifest()
        self.assertEqual(manifest["sessions"][-1]["outcome"], "failed")
        self.assertEqual(manifest["cases"][0]["attempts"][-1]["state"], "failed")
        self.assertTrue(Path(manifest["targets"]["101"]["path"]).exists())

    def test_bad_deadline_does_not_prevent_failure_checkpoint(self):
        """Even non-finite JSON deadline values cannot poison the manifest write."""
        def invalid_update(*args, **kwargs):
            """Inject malformed live state while a measured attempt is active."""
            elapsed = self.fake_fio(*args, **kwargs)
            if len(self.calls) == 2:
                Path(args[3]).write_text('{"deadline_epoch": NaN}')
                raise ValueError("invalid live deadline")
            return elapsed

        with patch.object(runner, "run_command", side_effect=invalid_update):
            self.assertEqual(self.invoke("--targets", "101"), 1)
        self.assertEqual(self.manifest()["sessions"][-1]["outcome"], "failed")
        self.assertEqual(self.manifest()["cases"][0]["attempts"][-1]["state"], "failed")

    def test_cleanup_failure_does_not_repeat_measurements(self):
        """A completed target with failed unlink resumes cleanup only."""
        original = Path.unlink

        def fail_data_unlink(path, *args, **kwargs):
            """Inject a cleanup error only for the synthetic dataset."""
            if path.name == "data":
                raise PermissionError("fake cleanup failure")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_data_unlink):
            self.assertEqual(self.invoke("--targets", "101"), 1)
        count = len(self.calls)
        self.assertTrue(all(runner.completed(case["attempts"]) for case in self.manifest()["cases"]))
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(len(self.calls), count)

    def test_crash_after_unlink_before_checkpoint(self):
        """An absent completed target file is cleanup success, never a new setup."""
        self.assertEqual(self.invoke("--targets", "101"), 0)
        manifest = self.manifest()
        manifest["targets"]["101"]["cleanup"] = "pending"
        atomic_json(self.root / "manifest.json", manifest)
        count = len(self.calls)
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(len(self.calls), count)

    def test_corrupt_completed_result_retries_only_that_case(self):
        """Missing native evidence invalidates one case, not the whole target."""
        self.invoke("--targets", "101")
        manifest = self.manifest()
        artifact = self.root / manifest["cases"][0]["attempts"][0]["artifacts"] / "fio.json"
        artifact.write_text("broken JSON")
        before = len(self.calls)
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(len(self.calls) - before, 2)  # replacement file + one measurement
        self.assertEqual(len(self.manifest()["cases"][1]["attempts"]), 1)

    def test_resume_compatibility_and_planning_updates(self):
        """Planning may change; scientific settings and selected targets may not."""
        self.fail_at = 2
        self.invoke("--targets", "101")
        before = len(self.calls)
        self.config["fio"]["iodepth"] = 16
        self.write_config()
        self.assertEqual(self.invoke("--resume"), 1)
        self.assertEqual(len(self.calls), before)
        self.config["fio"]["iodepth"] = 32
        self.config["planning"]["overhead_seconds"] = 6
        self.write_config()
        self.assertEqual(self.invoke("--resume", "--targets", "104"), 1)
        self.assertEqual(self.invoke("--resume"), 0)

    def test_identity_change_never_deletes_unknown_file(self):
        """A replacement at the owned pathname requires investigation."""
        self.fail_at = 2
        self.invoke("--targets", "101")
        path = Path(self.manifest()["targets"]["101"]["path"])
        path.rename(path.with_name("original"))
        path.write_text("unrelated")
        self.assertEqual(self.invoke("--resume"), 1)
        self.assertEqual(path.read_text(), "unrelated")

    def test_mount_mismatch_is_rejected_before_capacity_or_io(self):
        """An ancestor mount, different device or non-XFS filesystem is not a target."""
        target = self.targets[0]
        good = {"target": target["mount"], "source": target["device"], "fstype": "xfs"}
        for field, value in (("target", "/"), ("source", "/dev/wrong"), ("fstype", "ext4")):
            with self.subTest(field=field):
                output = subprocess.CompletedProcess([], 0, stdout=json.dumps({
                    "filesystems": [dict(good, **{field: value})]}))
                with patch.object(runner.subprocess, "run", return_value=output), \
                        patch.object(runner.os, "statvfs", side_effect=AssertionError("must not inspect capacity")):
                    with self.assertRaises(ValueError):
                        self.real_check_target(target)

    def test_admission_uses_estimates_not_hard_timeout(self):
        """A short allocation can admit quick work despite a 600-second failure bound."""
        self.config["prepare"]["timeout_seconds"] = 600
        self.write_config()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(["--results-dir", str(self.root), "--targets", "101",
                                "--time-limit", "2m", "--cleanup-buffer", "10s"])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 26)

    def test_abandoned_measurement_retries_but_preserves_raw_output(self):
        """Raw output without a completion checkpoint is an interrupted attempt."""
        self.fail_at = 3
        self.invoke("--targets", "101")
        manifest = self.manifest()
        attempt = manifest["cases"][0]["attempts"][0]
        artifact = self.root / attempt["artifacts"] / "fio.json"
        original = artifact.read_bytes()
        attempt["state"] = "running"
        atomic_json(self.root / "manifest.json", manifest)
        self.assertEqual(self.invoke("--resume"), 0)
        self.assertEqual(artifact.read_bytes(), original)
        attempts = self.manifest()["cases"][0]["attempts"]
        self.assertEqual([item["state"] for item in attempts], ["interrupted", "completed"])

    def test_runtime_cap_validation_and_job_policy(self):
        """A normal cap is successful, a short incomplete result is not."""
        output = self.base / "native.json"
        expected = {"jobname": "target-101", "operation": "read", "size": 1048576,
                    "runtime": 60, "hard_timeout": 75}
        payload = {"jobs": [{"jobname": "target-101", "error": 0,
                            "read": {"io_bytes": 4096, "runtime": 60001, "total_ios": 1}}]}
        atomic_json(output, payload)
        self.assertEqual(runner.validate_result(output, expected)[1]["completion_reason"], "time_limit")
        payload["jobs"][0]["read"]["runtime"] = 2000
        atomic_json(output, payload)
        with self.assertRaises(ValueError):
            runner.validate_result(output, expected)
        job = runner.build_job(self.config, self.targets[0], dict(self.config["workloads"][0], repetition=1),
                               self.base / "data", "measure")
        self.assertIn("time_based=0\n", job)
        self.assertIn("ramp_time=0\n", job)
        self.assertIn("runtime=60\n", job)
        self.assertIn("fallocate=none\n", job)

    def test_jobs_never_name_devices_or_beegfs_storage(self):
        """Every phase writes only the exact generated private data pathname."""
        target = self.targets[0]
        run_id = "a" * 32
        path = runner.target_data_path(target, run_id)
        self.assertEqual(path, Path(target["mount"]) / ".local-fio" / run_id / "target-101" / "data")
        for phase in ("prepare", "measure"):
            for workload in self.config["workloads"]:
                job = runner.build_job(self.config, target, dict(workload, repetition=1), path, phase)
                options = configparser.ConfigParser(interpolation=None)
                options.read_string(job)
                filename = options[options.sections()[0]]["filename"]
                self.assertEqual(filename, str(path))
                self.assertNotEqual(filename, target["device"])
                self.assertNotIn("beegfs_storage", Path(filename).parts)

    def test_adjacent_target_files_are_never_modified(self):
        """Preparation and cleanup cannot touch BeeGFS data or unrelated siblings."""
        mount = Path(self.targets[0]["mount"])
        protected = {
            mount / "beegfs_storage" / "important": b"beegfs sentinel",
            mount / ".local-fio" / "unrelated": b"unrelated sentinel",
            mount / "ordinary-file": b"ordinary sentinel",
        }
        for path, content in protected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.assertEqual(self.invoke("--targets", "101", "--pilot"), 0)
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content)

    def test_tampered_manifest_data_path_cannot_delete_victim(self):
        """A resumed manifest cannot redirect FIO or cleanup to an arbitrary file."""
        self.fail_at = 2
        self.invoke("--targets", "101", "--pilot")
        victim = self.base / "victim"
        victim.write_text("do not touch")
        manifest = self.manifest()
        manifest["targets"]["101"]["path"] = str(victim)
        atomic_json(self.root / "manifest.json", manifest)
        self.assertEqual(self.invoke("--resume"), 1)
        self.assertEqual(victim.read_text(), "do not touch")

    def test_symlinked_data_file_cannot_redirect_io_or_cleanup(self):
        """Replacing the private file with a symlink stops before following it."""
        self.fail_at = 2
        self.invoke("--targets", "101", "--pilot")
        data = Path(self.manifest()["targets"]["101"]["path"])
        victim = self.base / "victim"
        victim.write_text("do not touch")
        data.unlink()
        data.symlink_to(victim)
        self.assertEqual(self.invoke("--resume"), 1)
        self.assertEqual(victim.read_text(), "do not touch")
        self.assertTrue(data.is_symlink())

    def test_tampered_artifact_path_cannot_escape_results(self):
        """Resume validation never reads evidence through traversal or outside symlinks."""
        self.fail_at = 3
        self.invoke("--targets", "101", "--pilot")
        victim = self.base / "outside.json"
        victim.write_text("do not touch")
        manifest = self.manifest()
        manifest["cases"][0]["attempts"][0]["artifacts"] = "../../outside.json"
        atomic_json(self.root / "manifest.json", manifest)
        self.assertEqual(self.invoke("--resume"), 1)
        self.assertEqual(victim.read_text(), "do not touch")


if __name__ == "__main__":
    unittest.main()
