"""Plot-generation tests using synthetic normalized measurements."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualize_results as visualizer


class VisualizerTests(unittest.TestCase):
    """Ensure compact per-OST views are complete and source CSV stays immutable."""

    def test_one_per_ost_plot_is_created_for_each_access_pattern(self):
        """Each workload gets one bandwidth plot without aggregating OSTs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "measurements.csv", root / "plots"
            fields = ["host", "target_id", "media", "workload", "repetition", "bw_mib_s", "iops"]
            with source.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for media, targets, multiplier in (("hdd", (101, 102), 1), ("nvme", (104, 105), 100)):
                    for workload_index, workload in enumerate(visualizer.WORKLOAD_ORDER, 1):
                        for target_id in targets:
                            for repetition in range(1, 6):
                                writer.writerow({"host": "colva1", "target_id": target_id,
                                    "media": media, "workload": workload, "repetition": repetition,
                                    "bw_mib_s": multiplier * workload_index + repetition / 10,
                                    "iops": multiplier * workload_index * 100 + repetition})
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(visualizer.main([str(source), "--output-dir", str(output)]), 0)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            manifest = json.loads((output / "plot_manifest.json").read_text())
            self.assertEqual(manifest["measurements"], 100)
            self.assertEqual(manifest["plots"], [
                f"by_access_pattern/{workload}.png" for workload in visualizer.WORKLOAD_ORDER
            ])
            self.assertTrue(all((output / path).is_file() for path in manifest["plots"]))
            self.assertIn("never combined", manifest["semantics"]["sampling_unit"])


if __name__ == "__main__":
    unittest.main()
