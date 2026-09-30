"""Focused check for the installed mdtest summary-table format."""

import sys
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import visualize_results as visualizer


class NativeSummaryTest(unittest.TestCase):
    def test_parses_all_phases_and_zero_single_iteration_stddev(self):
        rows = "".join(
            f"   {operation:<24} 1000.000 900.000 950.000 0.000\n"
            for operation in visualizer.OPERATIONS)
        output = ("SUMMARY rate (in ops/sec): (of 1 iterations)\n"
                  "   Operation                     Max            Min           Mean        Std Dev\n"
                  + rows + "SUMMARY time (in ms/op): (of 1 iterations)\n"
                  "   Operation                     Max            Min           Mean        Std Dev\n"
                  + rows)
        rates = visualizer.parse_table(output, "rate (in ops/sec):")
        times = visualizer.parse_table(output, "time (in ms/op):")
        self.assertEqual(set(rates), set(visualizer.OPERATIONS))
        self.assertEqual(rates["File creation"]["mean"], 950.0)
        self.assertEqual(times["Directory stat"]["stddev"], 0.0)

    def test_rejects_duplicate_phase_rows(self):
        row = "Directory stat 1000.000 900.000 950.000 0.000\n"
        output = ("SUMMARY rate (in ops/sec): (of 1 iterations)\n" + row + row)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            visualizer.parse_table(output, "rate (in ops/sec):")

    def test_writes_one_metric_figure(self):
        summary = {name: {"min": 0.9, "mean": 1.0, "max": 1.1, "stddev": 0.0}
                   for name in visualizer.OPERATIONS}
        case = {"unit": {"placement": "anjuna2", "ranks": 1,
                          "layout": "flat", "repetition": 1},
                "rate": summary}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rate.png"
            visualizer.plot_metric([case], "rate", "mean", "rate", path, "pilot")
            self.assertGreater(path.stat().st_size, 0)

    def test_time_per_operation_uses_mean_of_each_invocations_reciprocal_rate(self):
        self.assertAlmostEqual(
            visualizer.mean_time_per_operation_ms([500.0, 1000.0]), 1.5)
        self.assertAlmostEqual(
            visualizer.time_per_operation_ms(768.393), 1000.0 / 768.393)

    def test_writes_per_operation_figure_from_rates(self):
        summary = {name: {"min": 0.0, "mean": 0.0, "max": 0.0, "stddev": 0.0}
                   for name in visualizer.OPERATIONS}
        rates = {name: {"min": 900.0, "mean": 1000.0, "max": 1100.0,
                        "stddev": 0.0}
                 for name in visualizer.OPERATIONS}
        case = {"unit": {"placement": "anjuna2", "ranks": 1,
                          "layout": "flat", "repetition": 1},
                "time": summary, "rate": rates}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time.png"
            visualizer.plot_metric([case], "time", "mean", "time", path, "pilot")
            self.assertGreater(path.stat().st_size, 0)

    def test_plots_hdd_and_ssd_target_assignments_as_one_pair_per_configuration(self):
        rates = {name: {"min": 900.0, "mean": 1000.0, "max": 1100.0,
                        "stddev": 0.0}
                 for name in visualizer.OPERATIONS}
        cases = [{"unit": {"placement": "anjuna2", "ranks": 1,
                            "layout": "flat", "repetition": 1,
                            "target_class": target_class},
                  "rate": rates}
                 for target_class in ("hdd", "ssd")]
        pools = {"classes": {
            "hdd": {"name": "hdd-singleton", "pool_id": 3, "target_id": 101},
            "ssd": {"name": "ssd-singleton", "pool_id": 8, "target_id": 104}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rate.png"
            visualizer.plot_metric(cases, "rate", "mean", "rate", path,
                                   "targeted-pilot", pools)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
