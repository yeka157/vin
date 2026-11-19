"""The agent panel under the input, its keys, and the read-only subagent view."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown
from rich.text import Text

from vin.sessions import storage
from vin.sessions.records import new_session
from vin.tui import subagents
from vin.tui.app import VinApp
from vin.tui.layout import build_workspace
from vin.types import Subagent
from tui_typing_helpers import key_event


def subagent(task_id: str, description: str) -> Subagent:
    return {"id": task_id, "description": description, "status": "finished", "started_at": 0.0,
            "duration": 4.0, "tokens": 900,
            "transcript": [{"role": "user", "content": f"Prompt for {description}"},
                           {"role": "assistant", "content": f"Answer for {description}"}]}


def plain(fragments: StyleAndTextTuples) -> list[str]:
    return "".join(fragment[1] for fragment in fragments).split("\n")


class PanelTextTests(unittest.TestCase):
    def test_rows_show_the_viewed_agent_activity_time_and_tokens(self) -> None:
        running = subagent("call_a", "Find sessions")
        running.update(status="running", duration=None, started_at=10.0, tokens=12400,
                       transcript=[{"role": "user", "content": "p"},
                                   {"role": "tool", "name": "read", "label": "read storage.py", "status": "running",
                                    "content": Text("● read storage.py · running")}])
        thinking = subagent("call_b", "Plan")
        thinking.update(status="running", duration=None, started_at=10.0, transcript=[{"role": "user", "content": "p"}])
        failed = subagent("call_c", "Run tests")
        failed.update(status="failed", duration=65.0, tokens=50)

        with patch.object(subagents, "monotonic", return_value=50.0):
            fragments = subagents.panel_text([running, thinking, failed], "call_a", 3, "Review the queue")

        self.assertEqual(plain(fragments), [
            " ○ Review the queue",
            " ● task Find sessions · read storage.py 40s · ↓ 12.4k tokens",
            " ○ task Plan · working 40s · ↓ 900 tokens",
            " ○ task Run tests · failed 1m 5s · ↓ 50 tokens",
        ])
        styles = [fragment[0] for fragment in fragments if fragment[1] != "\n"]
        self.assertEqual(styles, ["class:placeholder"] * 3 + ["fg:#a5b4fc bold"])

    def test_main_is_marked_when_no_subagent_is_viewed(self) -> None:
        self.assertEqual(plain(subagents.panel_text([subagent("call_a", "Find")], None, None, "main"))[0], " ● main")


class AgentPanelKeyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", Path(temporary.name) / "sessions"))
        self.enterContext(create_app_session(input=DummyInput(), output=DummyOutput()))
        self.owner = VinApp()
        self.owner.project_dir = self.project
        self.owner.header = Markdown("Vin")
        build_workspace(self.owner)
        self.enterContext(set_app(self.owner.app))
        self.buffer = self.owner.session.default_buffer
        self.buffer.accept_handler = self.owner.accept_input
        self.owner.transcript = [{"role": "user", "content": "Main prompt"}]
        self.owner.subagents = [subagent("call_old", "Earlier"), subagent("call_a", "Find sessions"),
                                subagent("call_b", "Run tests")]
        self.owner.panel_start = 1
        self.buffer.document = Document("line one\nline two", cursor_position=len("line one\nline two"))

    async def asyncTearDown(self) -> None:
        await self.owner.app.cancel_and_wait_for_background_tasks()

    def press(self, *keys: Keys) -> None:
        for key in keys:
            self.owner.app.key_processor.feed(KeyPress(key))
        self.owner.app.key_processor.process_keys()

    def escape(self) -> None:
        bindings = self.owner.app.key_bindings
        assert bindings is not None
        binding = next(item for item in bindings.get_bindings_for_keys((Keys.Escape,))
                       if item.handler.__name__ == "return_to_main")
        self.assertTrue(binding.filter())
        self.assertFalse(binding.eager(), "An eager Escape would break Alt+Enter.")
        _ = binding.handler(key_event(self.owner.app, Keys.Escape))

    def visible(self) -> list[str]:
        return [str(entry["content"]) for entry in self.owner.get_transcript_entries()]

    async def test_panel_lists_only_the_latest_turn_under_the_session_title(self) -> None:
        self.owner.session_record = new_session(self.project, "Review the queue")
        self.assertEqual([agent["id"] for agent in self.owner.panel_agents()], ["call_a", "call_b"])
        window_text = plain(subagents.panel_text(self.owner.panel_agents(), None, None, "Review the queue"))
        self.assertEqual(window_text[0], " ● Review the queue")

    async def test_down_on_the_last_line_enters_the_panel_and_enter_opens_a_read_only_subagent(self) -> None:
        self.press(Keys.Down)
        self.assertEqual(self.owner.panel_index, 0)
        self.press(Keys.Down, Keys.Down, Keys.Down)
        self.assertEqual(self.owner.panel_index, 2, "Down stops at the last agent.")
        self.press(Keys.Up)
        self.assertEqual(self.owner.panel_index, 1)

        self.press(Keys.ControlM)
        self.assertEqual((self.owner.viewed, self.owner.panel_index), ("call_a", None))
        self.assertEqual(self.visible(), ["Prompt for Find sessions", "Answer for Find sessions"])
        self.assertIn("Viewing task · Find sessions", self.owner.model_indicator_text())
        self.assertEqual(self.buffer.text, "line one\nline two")

        self.press(Keys.ControlM)
        self.assertEqual(self.buffer.text, "line one\nline two", "Enter in a subagent view sends nothing.")
        self.assertTrue(self.owner.submitted.empty())

        self.press(Keys.Down)
        self.assertEqual(self.owner.panel_index, 1, "Re-entering the panel selects the viewed agent.")

    async def test_up_from_the_top_row_and_escape_return_to_the_main_input(self) -> None:
        self.press(Keys.Down, Keys.Up)
        self.assertIsNone(self.owner.panel_index)
        self.assertEqual(self.buffer.text, "line one\nline two")

        self.owner.viewed = "call_b"
        self.owner.panel_index = 2
        self.escape()
        self.assertEqual((self.owner.viewed, self.owner.panel_index), (None, None))
        self.assertEqual(self.visible(), ["Main prompt"])

    async def test_alt_enter_still_inserts_a_newline_in_a_subagent_view(self) -> None:
        self.owner.viewed = "call_a"
        self.press(Keys.Escape, Keys.ControlM)
        self.assertEqual(self.buffer.text, "line one\nline two\n")
        self.assertEqual(self.owner.viewed, "call_a")

    async def test_down_keeps_moving_the_cursor_when_not_on_the_last_line_or_with_no_panel(self) -> None:
        self.buffer.document = Document("line one\nline two", cursor_position=0)
        self.press(Keys.Down)
        self.assertIsNone(self.owner.panel_index)
        self.assertEqual(self.buffer.document.cursor_position_row, 1)

        self.owner.panel_start = len(self.owner.subagents)
        self.press(Keys.Down)
        self.assertIsNone(self.owner.panel_index)

    async def test_down_with_a_completion_menu_open_does_not_enter_the_panel(self) -> None:
        self.buffer.document = Document("/he")
        self.buffer.start_completion(select_first=False)
        for _ in range(100):
            if self.buffer.complete_state is not None and self.buffer.complete_state.completions:
                break
            await asyncio.sleep(0)
        assert self.buffer.complete_state is not None
        self.press(Keys.Down)
        self.assertIsNone(self.owner.panel_index)

    async def test_clear_resets_the_panel_and_view(self) -> None:
        self.owner.viewed = "call_a"
        self.owner.handle_command("/clear")
        self.assertEqual((self.owner.subagents, self.owner.panel_start, self.owner.viewed), ([], 0, None))


if __name__ == "__main__":
    unittest.main()
