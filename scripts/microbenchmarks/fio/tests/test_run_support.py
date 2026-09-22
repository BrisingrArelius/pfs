"""Exercise real process control with harmless Python children, never FIO."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_support as support


class ProcessTests(unittest.TestCase):
    """Check deadline updates, output retention and child-group termination."""

    def setUp(self):
        """Create a private artifact directory and a fresh deadline."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.deadline = self.root / "deadline.json"
        support.set_deadline(self.deadline, time_limit=30)

    def run_python(self, code, timeout=5):
        """Run a harmless child through the exact production process wrapper."""
        return support.run_command([sys.executable, "-c", code], self.root / "stdout",
                                   self.root / "stderr", self.deadline, 0, timeout)

    @contextmanager
    def update_after(self, delay, callback):
        """Schedule a live deadline change and always join the updater thread."""
        timer = threading.Timer(delay, callback)
        timer.start()
        try:
            yield
        finally:
            timer.join()

    def assert_reaped(self):
        """A failed child must no longer exist, not merely have received a signal."""
        pid = int((self.root / "stdout").read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_success_and_nonzero_are_distinct(self):
        """Return actual wall time and preserve both streams on failure."""
        elapsed = self.run_python("print('ok')")
        self.assertGreater(elapsed, 0)
        self.assertEqual((self.root / "stdout").read_text(), "ok\n")
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.run_python("import sys; print('bad', file=sys.stderr); sys.exit(7)")
        self.assertEqual(error.exception.returncode, 7)
        self.assertGreater(error.exception.command_wall_seconds, 0)
        self.assertIn("bad", (self.root / "stderr").read_text())

    def test_hard_timeout_reaps_child(self):
        """Hard timeout is a failure, independent of a long reservation."""
        with self.assertRaises(TimeoutError):
            self.run_python("import os,time; print(os.getpid(), flush=True); time.sleep(20)", timeout=0.3)
        self.assert_reaped()

    def test_deadline_stop_is_not_hard_timeout(self):
        """The reservation boundary raises the separately resumable condition."""
        support.set_deadline(self.deadline, time_limit=0.4)
        with self.assertRaises(support.BudgetExpired):
            self.run_python("import os,time; print(os.getpid(), flush=True); time.sleep(20)")
        self.assert_reaped()

    def test_live_extension_allows_existing_child_to_finish(self):
        """An active child observes a later deadline without restarting."""
        support.set_deadline(self.deadline, time_limit=0.5)
        with self.update_after(0.1, lambda: support.set_deadline(self.deadline, extend=2)):
            self.run_python("import time; time.sleep(0.8); print('finished')")
        self.assertEqual((self.root / "stdout").read_text(), "finished\n")

    def test_extension_does_not_disable_hard_timeout(self):
        """More reservation time cannot turn a stuck command into an endless run."""
        with self.update_after(0.05, lambda: support.set_deadline(self.deadline, extend=20)):
            with self.assertRaises(TimeoutError):
                self.run_python("import os,time; print(os.getpid(), flush=True); time.sleep(20)", timeout=0.3)
        self.assert_reaped()

    def test_malformed_live_deadline_kills_child(self):
        """Bad updates never silently remove the allocation bound."""
        with self.update_after(0.1, lambda: self.deadline.write_text("broken")):
            with self.assertRaises(json.JSONDecodeError):
                self.run_python("import os,time; print(os.getpid(), flush=True); time.sleep(20)")
        self.assert_reaped()

    def test_interrupt_reaps_child(self):
        """SIGINT and the runner's SIGTERM handler share KeyboardInterrupt cleanup."""
        original = support.remaining_seconds
        calls = 0

        def interrupt(path, buffer):
            """Interrupt only after the child has had one poll interval to start."""
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt
            return original(path, buffer)

        with patch.object(support, "remaining_seconds", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_python("import os,time; print(os.getpid(), flush=True); time.sleep(20)")
        self.assert_reaped()

    def test_descendant_is_killed_with_process_group(self):
        """Do not leave a TERM-ignoring child behind after its leader exits."""
        pidfile = self.root / "descendant.pid"
        child = ("import os,signal,time; from pathlib import Path; "
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(20)")
        leader = ("import os,subprocess,sys,time; print(os.getpid(), flush=True); "
                  f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(20)")
        with self.assertRaises(TimeoutError):
            self.run_python(leader, timeout=0.5)
        self.assert_reaped()
        child_pid = int(pidfile.read_text())
        for _ in range(100):
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break  # Orphan zombies are reaped by init; they cannot perform I/O.
            time.sleep(0.01)
        else:
            self.fail("descendant still running after group termination")

    def test_actual_signals_reap_active_child(self):
        """Both real SIGINT and SIGTERM reach the process cleanup path."""
        module_path = str(Path(support.__file__).parent)
        code = (
            "import signal,sys; from pathlib import Path\n"
            f"sys.path.insert(0, {module_path!r})\n"
            "from run_support import run_command\n"
            "signal.signal(signal.SIGTERM, signal.default_int_handler)\n"
            f"root = Path({str(self.root)!r})\n"
            "try:\n"
            " run_command([sys.executable, '-c', 'import os,time; print(os.getpid(), flush=True); time.sleep(30)'], "
            "root/'stdout', root/'stderr', root/'deadline.json', 0, 20)\n"
            "except KeyboardInterrupt:\n"
            " sys.exit(130)\n")
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                for name in ("stdout", "stderr"):
                    (self.root / name).unlink(missing_ok=True)
                with subprocess.Popen([sys.executable, "-B", "-c", code]) as wrapper:
                    for _ in range(200):
                        if (self.root / "stdout").exists() and (self.root / "stdout").read_text().strip():
                            break
                        time.sleep(0.01)
                    wrapper.send_signal(signum)
                    self.assertEqual(wrapper.wait(timeout=5), 130)
                self.assert_reaped()

    def test_atomic_replace_failure_preserves_previous_checkpoint(self):
        """A failed publication cannot expose truncated progress or leave temp files."""
        path = self.root / "manifest.json"
        support.atomic_json(path, {"completed": 1})
        with patch.object(support.os, "replace", side_effect=OSError("fake failure")):
            with self.assertRaises(OSError):
                support.atomic_json(path, {"completed": 2})
        self.assertEqual(support.read_json(path), {"completed": 1})
        self.assertEqual(list(self.root.glob(".manifest.json-*")), [])


if __name__ == "__main__":
    unittest.main()
