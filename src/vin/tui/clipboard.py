"""Copy text to the system clipboard from the terminal app."""

import base64
import shutil
import subprocess

from prompt_toolkit.application.current import get_app


def copy_to_clipboard(text: str) -> None:
    """Use pbcopy on macOS; elsewhere send OSC 52, which most terminals accept."""
    pbcopy = shutil.which("pbcopy")
    if pbcopy is not None:
        try:
            _ = subprocess.run([pbcopy], input=text.encode(), check=True, timeout=2)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    output = get_app().output
    output.write_raw(f"\x1b]52;c;{base64.b64encode(text.encode()).decode()}\x07")
    output.flush()
