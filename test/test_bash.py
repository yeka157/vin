"""Bash execution, bounded output, timeout, and cancellation tests."""

from pathlib import Path
import tempfile
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch
from vin import tools


class BashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = tools
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(patch.object(self.tools, "LAUNCH_DIR", self.root))


    def test_closed_output_does_not_skip_waiting_for_command_timeout(self) -> None:
        start = time.monotonic()
        result = self.tools.bash("exec 1>&- 2>&-; sleep 10", timeout=1)
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, .8)
        self.assertLess(elapsed, 4)
        self.assertIn("timed out", result.lower())

    def test_cancel_before_launch_does_not_spawn_a_process(self) -> None:
        cancel = Event()
        cancel.set()
        with self.tools._tool_cancellation(cancel), patch("subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.tools.bash("true")
        spawn.assert_not_called()

    def test_cancel_during_command_kills_descendants(self) -> None:
        cancel = Event()
        errors: list[Exception] = []

        def execute() -> None:
            try:
                with self.tools._tool_cancellation(cancel):
                    self.tools.bash("printf ready > started; (sleep 1.5; printf leaked > late) & wait", timeout=5)
            except Exception as error:
                errors.append(error)

        worker = Thread(target=execute, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while not (self.root / "started").exists() and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue((self.root / "started").exists(), errors)
            cancel.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn("cancelled", str(errors[0]))
            time.sleep(1.7)
            self.assertFalse((self.root / "late").exists())
        finally:
            cancel.set()
            worker.join(6)
