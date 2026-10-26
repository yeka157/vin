"""Visible edit diffs in the existing Rich/prompt-toolkit transcript."""

import unittest

from prompt_toolkit.formatted_text import to_formatted_text
from rich.text import Text

from vin.tui.transcript import render_transcript


class DiffRenderingTests(unittest.TestCase):
    def row(self, diff):
        return {"role": "tool", "name": "edit", "detail": "f", "status": "finished", "diff": diff,
                "expanded": True,
                "content": Text("● edit f · finished", style="dim")}

    def render(self, entries, width=80):
        fragments = to_formatted_text(render_transcript(entries, width, 100))
        return fragments, "".join(fragment[1] for fragment in fragments)


    def test_diff_preserves_literal_markup_leading_indentation_and_newline_marker(self):
        diff = ("--- f\n+++ f\n@@ -1,2 +1,2 @@\n     context\n"
                "-    [red]before[/red]\n+    **after**\n\\ No newline at end of file\n")
        _, text = self.render([self.row(diff)])
        lines = [line.rstrip() for line in text.splitlines()]
        self.assertIn("  2 │ -     [red]before[/red]", lines)
        self.assertIn("  2 │ +     **after**", lines)
        self.assertIn("  1 │       context", lines)
        self.assertIn("\\ No newline at end of file", text)

    def test_long_diff_lines_wrap_without_losing_text(self):
        addition = "+" + "abcdefghij" * 8
        _, text = self.render([self.row("--- f\n+++ f\n@@ -1 +1 @@\n-old\n" + addition + "\n")], width=24)
        self.assertIn(addition, "".join(text.split()))
        self.assertTrue(all(len(line) <= 24 for line in text.splitlines()))


    def test_malformed_diff_falls_back_to_literal_text_without_invented_numbers(self):
        diff = "--- f\n+++ f\n@@ broken @@\n-[red]before[/red]\n+**after**\n"
        _, text = self.render([self.row(diff)])
        self.assertIn("@@ broken @@", text)
        self.assertIn("-[red]before[/red]", text)
        self.assertIn("+**after**", text)
        self.assertNotIn("│", text)
