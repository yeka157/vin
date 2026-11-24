"""Assistant replies render Markdown headings at the left edge, like the rest of the text."""

from io import StringIO
import unittest

from rich.console import Console

from vin.tui.transcript import print_assistant_message


class HeadingTests(unittest.TestCase):
    def test_assistant_headings_start_at_the_left_edge(self) -> None:
        output = StringIO()
        console = Console(file=output, width=40, color_system=None)
        print_assistant_message(console, "# Vin\n\n## Two")
        lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines, [" Vin", " Two"])


if __name__ == "__main__":
    unittest.main()
