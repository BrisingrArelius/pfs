"""Read-only parser tests built from small synthetic native-FIO evidence."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parse_results as parser


class ParserTests(unittest.TestCase):
    """Validate normalization, completeness reporting and source immutability."""

    def setUp(self):
        """Create one host with one target, preparation and two repetitions."""
        temporary = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run" / "colva1"
        self.output = self.root / "derived"
        target = {"target_id": 101, "media": "hdd", "device": "/dev/sdb1", "mount": "/mnt/hdd2"}
        config = {"fio": {"size": 1048576, "runtime": 60}}
        preparation = self.attempt("raw/target-101/prepare-1", "byte_limit", 1048576, 1000)
        preparation["preparation_generation"] = 1
        cases = []
        for repetition, bandwidth in ((1, 104857600), (2, 125829120)):
            relative = f"raw/101__seq_read__r{repetition}/attempt-1"
            attempt = self.attempt(relative, "byte_limit", 1048576, 1000)
            cases.append({"id": f"101__seq_read__r{repetition}", "target_id": 101,
                          "workload": {"name": "seq_read", "rw": "read", "bs": "1m"},
                          "repetition": repetition, "attempts": [attempt]})
            self.native(relative, "read", 1048576, 1000, bandwidth)
        self.native("raw/target-101/prepare-1", "write", 1048576, 1000, 104857600)
        manifest = {
            "run_id": "a" * 32, "mode": "full", "host": "colva1", "config": config,
            "inventory": [target], "sessions": [{"outcome": "completed"}],
            "targets": {"101": {"cleanup": "completed", "preparations": [preparation]}},
            "cases": cases,
        }
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / "manifest.json").write_text(json.dumps(manifest))

    def attempt(self, artifact, reason, io_bytes, runtime):
        """Return one completed manifest attempt."""
        return {"id": 1, "state": "completed", "session": 1, "artifacts": artifact,
                "completion_reason": reason, "io_bytes": io_bytes, "fio_runtime_ms": runtime,
                "command_wall_seconds": runtime / 1000 + 0.5, "preparation_generation": 1}

    def native(self, relative, operation, io_bytes, runtime, bandwidth):
        """Write one minimal but complete native FIO JSON artifact."""
        folder = self.run / relative
        folder.mkdir(parents=True, exist_ok=True)
        other = "write" if operation == "read" else "read"
        stats = {"io_bytes": io_bytes, "runtime": runtime, "bw_bytes": bandwidth,
                 "iops": bandwidth / 1048576, "total_ios": 1,
                 "clat_ns": {"mean": 1000, "percentile": {
                     "50.000000": 1000, "95.000000": 2000,
                     "99.000000": 3000, "99.900000": 4000}}}
        payload = {"jobs": [{"jobname": "target-101", "error": 0, operation: stats,
                             other: {"io_bytes": 0}}]}
        (folder / "fio.json").write_text(json.dumps(payload))

    def hashes(self):
        """Hash every source file to prove parsing cannot modify evidence."""
        return {path.relative_to(self.run): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.run.rglob("*") if path.is_file()}

    def test_complete_run_emits_csv_markdown_and_pass_report(self):
        """Normalize repetitions and summarize their median/range accurately."""
        before = self.hashes()
        self.assertEqual(parser.main([str(self.run), "--output-dir", str(self.output)]), 0)
        self.assertEqual(self.hashes(), before)
        with (self.output / "measurements.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["completion_reason"] for row in rows}, {"byte_limit"})
        with (self.output / "summary.csv").open(newline="") as source:
            summary = list(csv.DictReader(source))
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary[0]["bw_mib_s_median"]), 110)
        self.assertEqual(summary[0]["runs"], "2")
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertEqual((report["measurements"], report["preparations"]), (2, 1))
        self.assertEqual(report["hosts"][0]["errors"], [])
        self.assertIn("Validation: **PASS**", (self.output / "summary.md").read_text())

    def test_incomplete_run_returns_two_but_keeps_valid_rows(self):
        """One pending measurement is reported without discarding valid evidence."""
        manifest = json.loads((self.run / "manifest.json").read_text())
        manifest["cases"][1]["attempts"][-1]["state"] = "interrupted"
        manifest["sessions"][-1]["outcome"] = "budget_stop"
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(parser.main([str(self.run), "--output-dir", str(self.output)]), 2)
        with (self.output / "measurements.csv").open(newline="") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertTrue(report["hosts"][0]["errors"])
        self.assertTrue(report["hosts"][0]["warnings"])
        self.assertIn("Validation: **FAIL**", (self.output / "summary.md").read_text())

    def test_corrupt_native_output_is_an_error_not_a_rerun(self):
        """Parsing failure changes only derived output and identifies the case."""
        native = self.run / "raw/101__seq_read__r1/attempt-1/fio.json"
        native.write_text("not JSON")
        before = self.hashes()
        self.assertEqual(parser.main([str(self.run), "--output-dir", str(self.output)]), 2)
        self.assertEqual(self.hashes(), before)
        report = json.loads((self.output / "parse_report.json").read_text())
        self.assertIn("101__seq_read__r1", report["hosts"][0]["errors"][0])

    def test_traversal_artifact_is_rejected_without_reading_victim(self):
        """A tampered manifest cannot make the parser read outside the run."""
        victim = self.root / "victim.json"
        victim.write_text("do not read or modify")
        manifest = json.loads((self.run / "manifest.json").read_text())
        manifest["cases"][0]["attempts"][-1]["artifacts"] = "../../../victim.json"
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(parser.main([str(self.run), "--output-dir", str(self.output)]), 2)
        self.assertEqual(victim.read_text(), "do not read or modify")

    def test_output_inside_input_is_not_rediscovered(self):
        """A derived directory beneath a collection root cannot become evidence."""
        output = self.run.parent / "derived"
        self.assertEqual(parser.main([str(self.run.parent), "--output-dir", str(output)]), 0)
        self.assertEqual(parser.main([str(self.run.parent), "--output-dir", str(output)]), 0)


if __name__ == "__main__":
    unittest.main()
