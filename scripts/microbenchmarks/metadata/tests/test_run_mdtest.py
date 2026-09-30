"""Focused local checks for the fixed mdtest plan, ownership and raw resume."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_mdtest as md


class MetadataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=md.RUNS)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_fixed_plans_and_command(self):
        self.assertEqual(md.HDD_POOL_NAME, "hdd_meta_101")
        self.assertEqual(md.SSD_POOL_NAME, "ssd_meta_104")
        self.assertEqual(md.duration_text(3661), "1h 01m")
        self.assertEqual(len(md.legacy_units(True)), 4)
        self.assertEqual(len(md.units(True)), 8)
        pilot_units = md.units(True)
        self.assertEqual([unit["target_class"] for unit in pilot_units[:4]],
                         ["hdd", "ssd", "ssd", "hdd"])
        full = md.units(False)
        self.assertEqual(len(md.legacy_units(False)), 90)
        self.assertEqual(len(full), 180)
        self.assertEqual(len({row["id"] for row in full}), 180)
        self.assertEqual({row["target_class"] for row in full}, {"hdd", "ssd"})
        self.assertEqual({row["ranks"] for row in full}, {1, 4, 16})
        first_pair_orders = {}
        for row in full:
            base_id = row["id"].rsplit("-", 1)[0]
            first_pair_orders.setdefault(base_id, []).append(row["target_class"])
        self.assertTrue(all(order in (["hdd", "ssd"], ["ssd", "hdd"])
                            for order in first_pair_orders.values()))
        config_orders = {}
        for base_id, order in first_pair_orders.items():
            config = base_id.split("-", 1)[1]
            config_orders.setdefault(config, []).append(order[0])
        self.assertTrue(all(len(order) == 5 and 2 <= order.count("hdd") <= 3
                            and 2 <= order.count("ssd") <= 3
                            for order in config_orders.values()))
        argv = md.command(md.units(True)[-1], "/work", 1000)
        self.assertEqual(argv[1:7], ["-wdir", "/work", "-np", "8", "-hosts",
                                     "anjuna2:4,anjuna3:4"])
        self.assertEqual(argv[argv.index("-n") + 1], "1000")
        self.assertIn("-u", md.command(md.legacy_units(True)[2], "/work", 1000))

    def test_pool_listing_resolution_requires_exact_singleton_targets(self):
        pools = ("Pool ID   Pool Description                      Targets\n"
                 "=======   ==================                   =======\n"
                 "      3   hdd_meta_target_101                  101\n"
                 "      8   ssd_meta_target_104                  104\n")
        targets = ("TargetID   St. Pool   NodeID\n"
                   "========   ========   ======\n"
                   "     101          3        1\n"
                   "     104          8        1\n")
        with (patch.object(md, "HDD_POOL_NAME", "hdd_meta_target_101"),
              patch.object(md, "SSD_POOL_NAME", "ssd_meta_target_104"),
              patch.object(md, "TARGET_POOLS", {
                  "hdd": {"name": "hdd_meta_target_101", "target_id": 101},
                  "ssd": {"name": "ssd_meta_target_104", "target_id": 104}}),
              patch.object(md, "beegfs_ctl", side_effect=[pools, targets])):
            state = md.storage_pool_state()
        self.assertEqual(state["classes"]["hdd"]["pool_id"], 3)
        self.assertEqual(state["classes"]["ssd"]["members"], [104])
        md.validate_storage_pools(state)

    def test_pool_parser_accepts_two_column_listing_and_rejects_extra_target(self):
        pools = "Pool ID  Description\n-------  -----------\n3 hdd_meta_target_101\n8 ssd_meta_target_104\n"
        targets = ("TargetID   St. Pool   NodeID\n"
                   "========   ========   ======\n"
                   "     101          3        1\n"
                   "     105          3        1   Online\n"
                   "     104          8        1\n")
        with (patch.object(md, "HDD_POOL_NAME", "hdd_meta_target_101"),
              patch.object(md, "SSD_POOL_NAME", "ssd_meta_target_104"),
              patch.object(md, "TARGET_POOLS", {
                  "hdd": {"name": "hdd_meta_target_101", "target_id": 101},
                  "ssd": {"name": "ssd_meta_target_104", "target_id": 104}}),
              patch.object(md, "beegfs_ctl", side_effect=[pools, targets])):
            with self.assertRaisesRegex(ValueError, "must contain only target 101"):
                md.storage_pool_state()

    def test_cleanup_rejects_links_and_preserves_siblings(self):
        shared = self.root / "metadata-pilot-test"
        shared.mkdir()
        unit = md.units(True)[0]
        owner = {"run_id": shared.name}
        work = shared / unit["id"]
        work.mkdir()
        md.save(work / "owner.json", owner)
        (work / "work").mkdir()
        sibling = shared / "keep"
        sibling.write_text("untouched")
        (work / "work" / "link").symlink_to(self.root)
        with self.assertRaises(ValueError):
            md.cleanup(work, shared, unit, owner)
        (work / "work" / "link").unlink()
        md.cleanup(work, shared, unit, owner)
        self.assertEqual(sibling.read_text(), "untouched")

    def test_pilot_raw_checkpoint_and_resume(self):
        runs = self.root / "results" / "microbenchmarks" / "runs"
        runs.mkdir(parents=True)
        namespace = self.root / "beegfs" / "pfs" / ".metadata-mdtest"
        namespace.parent.mkdir(parents=True)
        state = {"mount": "beegfs beegfs_nodev /mnt/beegfs rw", "mount_device": 1,
                 "client_config_sha256": "local"}
        baseline = {**state, "tools": {name: {"path": f"/mock/{name}", "sha256": name}
                                      for name in ("mpirun", "mdtest")},
                    "storage_pools": {
                        "classes": {
                            "hdd": {"name": "hdd_single", "pool_id": 3,
                                    "target_id": 101, "members": [101]},
                            "ssd": {"name": "ssd_single", "pool_id": 8,
                                    "target_id": 104, "members": [104]}},
                        "pool_listing": "pools", "target_listing": "targets"}}
        remote = {"client_config_sha256": {"anjuna2": "a2", "anjuna3": "a3"}}
        launched = []

        class SetPattern:
            returncode = 0
            stdout = "pattern set\n"
            stderr = ""

        def fake_entryinfo(argv):
            pool_id = 3 if str(argv[-1]).endswith("-hdd/work") else 8
            pool_name = "hdd_single" if pool_id == 3 else "ssd_single"
            return ("Stripe pattern details:\n"
                    "+ Number of storage targets: desired: 1\n"
                    f"+ Storage Pool: {pool_id} ({pool_name})\n")

        def fake_launch(argv, output, _timeout):
            launched.append(argv)
            (output / "stdout.txt").write_text("native mdtest output\n")
            (output / "stderr.txt").write_text("")
            return {"returncode": 0, "interrupted": False}

        with (patch.object(md, "RUNS", runs), patch.object(md, "NAMESPACE", namespace),
              patch.object(md, "MPIRUN", Path("/mock/mpirun")),
              patch.object(md, "MDTEST", Path("/mock/mdtest")),
              patch.object(md, "preflight", return_value=baseline),
              patch.object(md, "mount_state", return_value=state),
              patch.object(md, "storage_pool_state", return_value=baseline["storage_pools"]),
              patch.object(md, "beegfs_ctl", side_effect=fake_entryinfo),
              patch.object(md.subprocess, "run", return_value=SetPattern()),
              patch.object(md, "remote_probe", return_value=remote),
              patch.object(md, "launch", side_effect=fake_launch)):
            md.benchmark("metadata-pilot-local", True)
            md.benchmark("metadata-pilot-local", True)
            self.assertEqual(len(launched), 8)
            case = runs / "metadata-pilot-local" / "cases" / "pilot-01-hdd"
            self.assertEqual(json.loads((case / "result.json").read_text())["cleanup"], "completed")
            (case / "stdout.txt").write_text("changed")
            with self.assertRaises(ValueError):
                md.benchmark("metadata-pilot-local", True)

    def test_directory_pattern_parses_native_entryinfo(self):
        output = ("Stripe pattern details:\n"
                  "+ Number of storage targets: desired: 1\n"
                  "+ Storage Pool: 3 (hdd_meta_target_101)\n")
        self.assertEqual(md.directory_pattern(output), {"pool_id": 3, "num_targets": 1})

    def test_placement_evidence_must_match_pool_and_pattern(self):
        unit = md.units(True)[0]
        pool = {"name": "hdd_single", "pool_id": 3,
                "target_id": 101, "members": [101]}
        pools = {"classes": {"hdd": pool, "ssd": {
            "name": "ssd_single", "pool_id": 8, "target_id": 104,
            "members": [104]}}}
        work = Path("/work")
        record = {"target_class": "hdd", "pool": pool,
                  "setpattern_command": md.pattern_command(work, 3),
                  "verified_pattern": {"pool_id": 3, "num_targets": 1},
                  "directory_entryinfo": (
                      "+ Number of storage targets: desired: 1\n"
                      "+ Storage Pool: 3 (hdd_single)\n")}
        md.validate_placement(record, unit, work, pools)
        record["pool"] = {**pool, "pool_id": 4}
        with self.assertRaisesRegex(ValueError, "placement differs"):
            md.validate_placement(record, unit, work, pools)

    def test_remote_probe_shell_syntax(self):
        class Result:
            returncode = 0
            stdout = "anjuna2 hash2\nanjuna3 hash3\n"
            stderr = ""

        with patch.object(md.subprocess, "run", return_value=Result()) as called:
            md.remote_probe(self.root / "owner.json", "mdtest-hash")
        argv = called.call_args.args[0]
        self.assertEqual(argv[1:3], ["-wdir", str(self.root)])
        script = argv[9]
        self.assertEqual(subprocess.run(["/bin/sh", "-n", "-c", script]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
