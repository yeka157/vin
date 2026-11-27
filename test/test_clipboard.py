"""Copying a transcript selection: pbcopy, the OSC 52 fallback and the status line note."""

import base64
import subprocess
import unittest
from unittest.mock import Mock, patch

from vin.tui import clipboard, model_menu
from vin.tui.app import VinApp


class ClipboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.write_raw = Mock()
        self.flush = Mock()
        output = Mock(write_raw=self.write_raw, flush=self.flush)
        self.enterContext(patch.object(clipboard, "get_app", return_value=Mock(output=output)))

    def osc52(self, text: str) -> str:
        return f"\x1b]52;c;{base64.b64encode(text.encode()).decode()}\x07"

    def test_pbcopy_gets_the_text_when_it_exists(self) -> None:
        with patch.object(clipboard.shutil, "which", return_value="/usr/bin/pbcopy"), \
                patch.object(clipboard.subprocess, "run") as run:
            clipboard.copy_to_clipboard("héllo\nworld")
        run.assert_called_once_with(["/usr/bin/pbcopy"], input="héllo\nworld".encode(), check=True, timeout=2)
        self.write_raw.assert_not_called()

    def test_without_pbcopy_or_when_it_fails_the_terminal_gets_osc_52(self) -> None:
        for which, error in ((None, None), ("/usr/bin/pbcopy", subprocess.TimeoutExpired("pbcopy", 2)),
                             ("/usr/bin/pbcopy", OSError("broken"))):
            with self.subTest(which=which, error=error):
                self.write_raw.reset_mock()
                self.flush.reset_mock()
                with patch.object(clipboard.shutil, "which", return_value=which), \
                        patch.object(clipboard.subprocess, "run", side_effect=error):
                    clipboard.copy_to_clipboard("héllo")
                self.write_raw.assert_called_once_with(self.osc52("héllo"))
                self.flush.assert_called_once_with()


class CopiedNoteTests(unittest.TestCase):
    def test_status_line_says_copied_for_two_seconds(self) -> None:
        app = VinApp()
        invalidate = Mock()
        app.app = Mock(invalidate=invalidate)
        with patch("vin.tui.app.copy_to_clipboard") as copy, patch("vin.tui.app.monotonic", return_value=100.0):
            app.copy_selection("text")
        copy.assert_called_once_with("text")
        invalidate.assert_called_once_with()
        for now, expected in ((101.9, " Copied to clipboard"), (102.1, " Not signed in · /login")):
            with self.subTest(now=now), patch.object(model_menu, "monotonic", return_value=now):
                self.assertEqual(app.model_indicator_text(), expected)


if __name__ == "__main__":
    _ = unittest.main()
