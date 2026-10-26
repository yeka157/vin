"""Automatic activity collapse; independent click controls follow this increment."""

import unittest

from prompt_toolkit.formatted_text import to_formatted_text
from rich.text import Text

from vin.tui.app import VinApp
from vin.tui.transcript import render_transcript


class ToolGroupRenderingTests(unittest.TestCase):
    def setUp(self):
        self.owner = VinApp()

    def tool(self, name, detail, status="finished"):
        return {
            "role": "tool", "name": name, "detail": detail, "status": status,
            "content": Text(f"● {name} {detail} · {status}", style="dim"),
        }

    def visible(self, width=100):
        fragments = to_formatted_text(
            render_transcript(self.owner.get_transcript_entries(), width, 100)
        )
        return "".join(fragment[1] for fragment in fragments)

    def test_active_group_keeps_completed_calls_visible_until_next_phase(self):
        self.owner.request_started_at = 1.0
        self.owner.transcript.extend([
            self.tool("read", "src/first.py"),
            self.tool("bash", "python -m unittest", "running"),
        ])
        text = self.visible()
        self.assertIn("read src/first.py · finished", text)
        self.assertIn("bash python -m unittest · running", text)
        self.assertGreater(text.index("Thinking…"), text.index("python -m unittest"))
        self.assertNotIn("2 tool calls", text)

    def test_new_assistant_update_collapses_group_while_turn_remains_active(self):
        self.owner.request_started_at = 1.0
        self.owner.transcript.extend([
            self.tool("read", "src/first.py"),
            self.tool("bash", "exit 7", "failed"),
        ])
        self.assertIn("src/first.py", self.visible())
        self.owner.transcript.append({"role": "assistant", "content": "Checking the failure."})
        text = self.visible()
        self.assertIn("2 tool calls · 1 finished · 1 failed", text)
        self.assertNotIn("src/first.py", text)
        self.assertNotIn("exit 7", text)
        self.assertLess(text.index("2 tool calls"), text.index("Checking the failure."))
        self.assertGreater(text.index("Thinking…"), text.index("Checking the failure."))
        self.assertEqual(len(self.owner.transcript), 3, "Collapse must retain original records.")


    def test_manually_reopened_closed_group_reveals_retained_diff(self):
        row = self.tool("edit", "src/example.py")
        row["diff"] = "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"
        self.owner.transcript.append(row)
        self.assertNotIn("+ new", self.visible())
        row["expanded"] = True
        text = self.visible()
        self.assertIn("▾ 1 tool call · 1 finished", text)
        self.assertIn("src/example.py", text)
        self.assertIn("- old", text)
        self.assertIn("+ new", text)
        self.assertLess(text.index("▾ 1 tool call"), text.index("src/example.py"))
        row["expanded"] = False
        collapsed = self.visible()
        self.assertIn("▸ 1 tool call · 1 finished", collapsed)
        self.assertNotIn("+ new", collapsed)
        self.assertIn("+new", row["diff"], "Closing details must retain the diff.")

    def test_opening_one_group_leaves_other_finished_group_collapsed(self):
        first = self.tool("read", "src/first.py")
        first["expanded"] = True
        self.owner.transcript.extend([
            first,
            {"role": "assistant", "content": "Next phase."},
            self.tool("bash", "python -m unittest"),
        ])
        text = self.visible()
        self.assertIn("src/first.py", text)
        self.assertNotIn("python -m unittest", text)
        self.assertEqual(text.count("▾ 1 tool call · 1 finished"), 1)
        self.assertEqual(text.count("▸ 1 tool call · 1 finished"), 1)
