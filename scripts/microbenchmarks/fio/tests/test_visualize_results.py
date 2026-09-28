"""The FIO visualizer consumes native host manifests directly."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualize_results as visualizer
import run_fio as runner


RUNS = Path(__file__).resolve().parents[4] / "results" / "microbenchmarks" / "runs"


class RawVisualizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.host = self.root / "colva1"
        self.host.mkdir()
        self.output = self.root / "plots"
        config = json.loads((Path(runner.__file__).parent / "fio_config.json").read_text())
        config["fio"]["size"] = 1048576
        config["repetitions"] = 1
        config["prepare"]["timeout_seconds"] = 30
        fio = config["fio"]
        inventory = [{"target_id": target, "media": media, "device": "/dev/test",
                      "mount": "/mnt/test"} for target, media in ((101, "hdd"), (104, "nvme"))]
        cases = []
        targets = {}
        for item in inventory:
            target = item["target_id"]
            preparation = self.host / f"raw/target-{target}/prepare-1"
            preparation.mkdir(parents=True)
            (preparation / "fio.json").write_text(json.dumps({"jobs": [{"jobname": f"target-{target}",
                "error": 0, "read": {"io_bytes": 0}, "write": {"io_bytes": 4 * 1048576,
                "runtime": 1000, "total_ios": 4, "bw_bytes": 4 * 1048576, "iops": 4}}]}))
            targets[str(target)] = {"cleanup": "completed", "preparations": [{
                "state": "completed", "artifacts": str(preparation.relative_to(self.host))}]}
        cases = runner.plan_cases(config, inventory)
        for case in cases:
            target = case["target_id"]
            name = case["workload"]["name"]
            relative = f"raw/{target}-{name}-r1/attempt-1"
            directory = self.host / relative
            directory.mkdir(parents=True)
            operation = "write" if "write" in name else "read"
            job_bytes = (2 if target == 101 else 4) * 1048576
            block_bytes = {"1m": 1048576, "4k": 4096, "128k": 131072}[case["workload"]["bs"]]
            jobs = []
            for index in range(4):
                stats = {"io_bytes": job_bytes, "runtime": 60000,
                         "total_ios": job_bytes // block_bytes,
                         "bw_bytes": job_bytes / 60,
                         "iops": job_bytes / block_bytes / 60}
                jobs.append({"jobname": f"target-{target}-job-{index + 1}", "error": 0,
                             operation: stats,
                             ("read" if operation == "write" else "write"): {"io_bytes": 0}})
            (directory / "fio.json").write_text(json.dumps({"jobs": jobs}))
            case["attempts"] = [{"id": 1, "state": "completed", "artifacts": relative,
                                 "completion_reason": "time_limit", "io_bytes": 4 * job_bytes}]
        scientific = {key: value for key, value in config.items()
                      if key not in {"planning", "measurement_timeout_seconds", "prepare"}}
        scientific["prepare"] = {key: value for key, value in config["prepare"].items()
                                 if key != "timeout_seconds"}
        scientific.update(host="colva1", inventory=inventory, fio_version="fio-3.test",
                          job_policy={"overwrite": 1, "fallocate": "none",
                                      "unique_filename": 0, "region_layout": "disjoint_offsets"})
        manifest = {"run_id": "a" * 32, "mode": "pilot", "host": "colva1",
                    "fingerprint": hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest(),
                    "fio_version": "fio-3.test", "config": config,
                    "inventory": inventory, "targets": targets, "cases": cases}
        (self.host / "manifest.json").write_text(json.dumps(manifest))

    def test_native_files_produce_five_figures_without_modification(self):
        before = {str(path.relative_to(self.host)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.host.rglob("*") if path.is_file()}
        self.assertEqual(visualizer.main([str(self.root)]), 0)
        self.assertEqual(before, {str(path.relative_to(self.host)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.host.rglob("*") if path.is_file()})
        manifest = json.loads((self.output / "plot_manifest.json").read_text())
        self.assertEqual(manifest["measurements"], 10)
        self.assertEqual(len(manifest["plots"]), 5)
        self.assertTrue(all((self.output / name).is_file() for name in manifest["plots"]))

    def test_corrupt_native_or_output_escape_fails(self):
        native = next(self.host.glob("raw/*/attempt-1/fio.json"))
        native.write_text("broken")
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertFalse((self.output / "plot_manifest.json").exists())
        with self.assertRaises(ValueError):
            visualizer.load_rows(self.root / "missing")
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.root.parent)]), 1)

    def test_missing_case_and_mixed_host_settings_cannot_be_plotted(self):
        manifest_file = self.host / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["cases"].pop()
        manifest_file.write_text(json.dumps(manifest))
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertFalse(self.output.exists())
        manifest_file.write_text(json.dumps({**manifest,
            "cases": runner.plan_cases(manifest["config"], manifest["inventory"])}))
        # Restore the original case attempts from the native fixture.
        restored = json.loads(manifest_file.read_text())
        for case in restored["cases"]:
            relative = f"raw/{case['target_id']}-{case['workload']['name']}-r1/attempt-1"
            case["attempts"] = [{"id": 1, "state": "completed", "artifacts": relative,
                                 "completion_reason": "time_limit",
                                 "io_bytes": 4 * (2 if case["target_id"] == 101 else 4) * 1048576}]
        manifest_file.write_text(json.dumps(restored))
        other = self.root / "colva2"
        shutil.copytree(self.host, other)
        changed = json.loads((other / "manifest.json").read_text())
        changed["host"] = "colva2"
        changed["config"]["fio"]["iodepth"] = 64
        (other / "manifest.json").write_text(json.dumps(changed))
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)

    def test_unowned_plot_symlink_is_not_followed(self):
        directory = self.output / "by_access_pattern"
        directory.mkdir(parents=True)
        victim = self.root / "unrelated"
        victim.write_text("preserve")
        (directory / "seq_read_fio_3_test.png").symlink_to(victim)
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(victim.read_text(), "preserve")

    def test_different_pinned_fio_versions_get_separate_figures(self):
        other = self.root / "colva2"
        shutil.copytree(self.host, other)
        manifest_path = other / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["host"] = "colva2"
        manifest["fio_version"] = "fio-3.other"
        config = manifest["config"]
        scientific = {key: value for key, value in config.items()
                      if key not in {"planning", "measurement_timeout_seconds", "prepare"}}
        scientific["prepare"] = {key: value for key, value in config["prepare"].items()
                                 if key != "timeout_seconds"}
        scientific.update(host="colva2", inventory=manifest["inventory"],
                          fio_version=manifest["fio_version"],
                          job_policy={"overwrite": 1, "fallocate": "none",
                                      "unique_filename": 0, "region_layout": "disjoint_offsets"})
        manifest["fingerprint"] = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 0)
        plotted = json.loads((self.output / "plot_manifest.json").read_text())
        self.assertEqual(len(plotted["plots"]), 10)
        self.assertEqual(plotted["fio_versions"], ["fio-3.other", "fio-3.test"])
        self.assertTrue(all("fio_3_other" in name or "fio_3_test" in name
                            for name in plotted["plots"]))
        shutil.rmtree(other)
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 0)
        self.assertFalse(any((self.output / name).exists() for name in plotted["plots"]
                             if "fio_3_other" in name))

    def test_failed_second_plot_leaves_no_figures_and_retry_succeeds(self):
        original = visualizer.plot_workload_by_ost
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("simulated rendering failure")
            return original(*args)

        with mock.patch.object(visualizer, "plot_workload_by_ost", side_effect=interrupted):
            self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(list(self.output.rglob("*.png")), [])
        self.assertEqual(list(self.output.glob(".plot-stage-*")), [])
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 0)

    def test_publish_failure_restores_previously_owned_figures(self):
        self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 0)
        before = {str(path.relative_to(self.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.output.rglob("*") if path.is_file()}
        replace = visualizer.os.replace

        def fail_manifest(source, destination):
            if Path(source).name.startswith(".plot-manifest-"):
                raise OSError("simulated manifest publication failure")
            return replace(source, destination)

        with mock.patch.object(visualizer.os, "replace", side_effect=fail_manifest):
            self.assertEqual(visualizer.main([str(self.root), "--output-dir", str(self.output)]), 1)
        self.assertEqual(before, {str(path.relative_to(self.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.output.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
