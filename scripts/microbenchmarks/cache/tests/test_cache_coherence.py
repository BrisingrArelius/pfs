"""Cache runner and visualizer use one explicit traffic-evidence contract."""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_cache_read as runner
import visualize_results as plots


class CacheEvidenceContract(unittest.TestCase):
    def test_case_counts_match_documented_matrix(self):
        self.assertEqual(len(runner.cases(True, "buffered")), 4)
        self.assertEqual(len(runner.cases(False, "buffered")), 20)
        self.assertEqual(len(runner.cases(True, "native")), 8)
        self.assertEqual(len(runner.cases(False, "native")), 30)

    def test_native_preflight_requires_effective_two_mib_threshold(self):
        config = "tuneFileCacheType = native\ntuneFileCacheBufSize = {}\n"
        self.assertFalse(runner.cache_config_matches(config.format(524288), "native"))
        self.assertTrue(runner.cache_config_matches(config.format(2097152), "native"))
        self.assertFalse(runner.cache_config_matches("tuneFileCacheType = native\n", "native"))
        self.assertFalse(runner.cache_config_matches(config.format(2097152), "buffered"))

    def test_current_native_states_use_traffic_only(self):
        label = plots.expected_label("native", "client_ram", .01, .01, 0,
                                     None, True, runner.VERIFICATION)
        self.assertEqual(label, "client_ram")
        label = plots.expected_label("native", "backend", 1.0, 1.0, 1.0,
                                     None, True, runner.VERIFICATION)
        self.assertEqual(label, "backend")

    def test_transitional_native_traffic_without_marker_is_checked(self):
        self.assertEqual(plots.expected_label("native", "client_ram", .01, .01, 0,
                                              None, True, None), "client_ram")
        with self.assertRaisesRegex(ValueError, "verification method"):
            plots.expected_label("native", "client_ram", 0, 0, 0,
                                 None, True, "unknown")

    def test_legacy_failed_probe_is_not_reclassified(self):
        label = plots.expected_label("native", "client_ram", 0, 0, 0,
                                     0.0, True, None)
        self.assertEqual(label, "unverified")

    def test_buffered_mode_rejects_client_ram(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            plots.expected_label("buffered", "client_ram", 0, 0, 0,
                                 None, True, runner.VERIFICATION)

    def test_two_read_pilot_reports_arithmetic_median(self):
        owner = {"run_id": "cache-native-pilot", "mode": "native", "pilot": True}
        cases = [{"medium": "HDD", "state": "client_ram", "label": "client_ram", "rate": rate}
                 for rate in (9537.0, 9573.0)]
        self.assertIn("median=9555.0", plots.summarize(owner, cases)[1])


if __name__ == "__main__":
    unittest.main()
