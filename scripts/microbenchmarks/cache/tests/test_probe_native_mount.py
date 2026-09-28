"""The native-cache mount probe must remain isolated and reversible."""

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import probe_native_mount as probe


class NativeMountProbe(unittest.TestCase):
    def test_only_explicit_buffered_setting_is_changed(self):
        original = "sysMgmtdHost = host\ntuneFileCacheType = buffered\nfoo = unchanged\n"
        self.assertEqual(probe.native_config(original),
                         original.replace("= buffered", "= native"))
        with self.assertRaises(ValueError):
            probe.native_config("tuneFileCacheType = native\n")
        with self.assertRaises(ValueError):
            probe.native_config("tuneFileCacheType = buffered\ntuneFileCacheType = buffered\n")

    def test_never_mount_in_host_namespace(self):
        with tempfile.TemporaryDirectory(dir=probe.RUNS) as root:
            with mock.patch.object(probe.os, "geteuid", return_value=0), mock.patch.object(
                    probe.os, "readlink", return_value="mnt:[same]"):
                with self.assertRaisesRegex(ValueError, "host mount namespace"):
                    probe.inside(root)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_private_mount_cleanup_after_success_and_failure(self):
        with tempfile.TemporaryDirectory(dir=probe.RUNS) as root:
            (Path(root) / "owner.json").write_text(
                '{"run_id": "' + Path(root).name + '", "domain": "cache-native-probe"}')
            original_read = Path.read_text

            def config_or_file(path, *args, **kwargs):
                if path == probe.CLIENT_CONFIG:
                    return "tuneFileCacheType = buffered\n"
                return original_read(path, *args, **kwargs)

            with ExitStack() as patches:
                patches.enter_context(mock.patch.object(probe.os, "geteuid", return_value=0))
                patches.enter_context(mock.patch.object(
                    probe.os, "readlink", side_effect=["mnt:[new]", "mnt:[host]"] * 2))
                patches.enter_context(mock.patch.object(
                    Path, "read_text", autospec=True, side_effect=config_or_file))
                patches.enter_context(mock.patch.object(
                    probe, "native_instances", side_effect=[set(), {"new-client"}] * 2))
                commands = []

                def successful(*argv):
                    commands.append(argv)
                    if argv[0] == "findmnt" and argv[-1] == "/mnt/beegfs":
                        return "beegfs_nodev beegfs cfgFile=/etc/beegfs/beegfs-client.conf\n"
                    if argv[0] == "findmnt":
                        return f"beegfs {Path(root) / 'native-mount'}\n"
                    return ""

                with mock.patch.object(probe, "run_command", side_effect=successful):
                    result = probe.inside(root)
                self.assertTrue(result["native_mount_supported"])
                self.assertTrue(result["original_mount_untouched"])
                self.assertIn("umount", [command[0] for command in commands])
                self.assertEqual([path.name for path in Path(root).iterdir()], ["owner.json"])

                def no_native(*argv):
                    if argv[0] == "findmnt" and argv[-1] == "/mnt/beegfs":
                        return "beegfs_nodev beegfs cfgFile=/etc/beegfs/beegfs-client.conf\n"
                    if argv[0] == "findmnt":
                        return f"beegfs {Path(root) / 'native-mount'}\n"
                    commands.append(argv)
                    return ""

                with mock.patch.object(probe, "run_command", side_effect=no_native), mock.patch.object(
                        probe, "native_instances", side_effect=[set(), set()]):
                    failure = probe.inside(root)
                self.assertFalse(failure["native_mount_supported"])
                self.assertIn("no new native", failure["error"])
                self.assertEqual([path.name for path in Path(root).iterdir()], ["owner.json"])


if __name__ == "__main__":
    unittest.main()
