"""Subagent chats in the transcript and the agent panel under the composer."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event
from typing import Literal, cast
import unittest
from unittest.mock import patch

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown
from rich.text import Text

from tui_typing_helpers import key_event
from vin.provider.events import ReplyEvent, ReplyEvents, ToolStarted
from vin.sessions import storage
from vin.sessions.records import new_session
from vin.tui import app as app_module, bindings, layout, subagents, transcript
from vin.tui.app import VinApp
from vin.tui.layout import build_workspace
from vin.types import Credentials, History, Subagent, TranscriptEntry


TASK_STARTED: ToolStarted = {"name": "task", "detail": "Find sessions", "task_id": "call_task",
                             "prompt": "Find where sessions are saved."}


def tagged(event: ReplyEvent, task_id: str = "call_task") -> ReplyEvent:
    return ("subagent", {"task_id": task_id, "event": event})


def rows(entries: list[TranscriptEntry] | None = None, *, of: Subagent | None = None) -> list[tuple[str, str]]:
    source = of["transcript"] if of is not None else entries or []
    return [(entry["role"], str(entry["content"])) for entry in source]


class SubagentTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.owner = app_module.VinApp()
        self.owner.session = PromptSession(input=DummyInput(), output=DummyOutput())
        self.owner.session.default_buffer = Buffer(document=Document("draft"))
        self.now = 100.0
        for module in (app_module, bindings, layout, transcript, subagents):
            self.enterContext(patch.object(module, "monotonic", side_effect=lambda: self.now, create=True))

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def start_turn(self) -> tuple[asyncio.Task[Literal["error", "rate_limit"] | None], ReplyEvents]:
        entered, release, stopped = Event(), Event(), Event()
        queues: SimpleQueue[ReplyEvents] = SimpleQueue()

        def worker(_credentials: Credentials | None, _model: str | None, _messages: History,
                   events: ReplyEvents, _cancel_event: Event, **_options: object) -> None:
            try:
                queues.put(events)
                entered.set()
                release.wait(5)
            finally:
                stopped.set()

        self.stream = self.enterContext(patch.object(app_module, "stream_chatgpt_response", side_effect=worker))
        self.owner.transcript.append({"role": "user", "content": "Delegate it."})
        task = asyncio.create_task(self.owner.stream_reply("Delegate it."))
        self.owner.pending_reply = task

        async def cleanup() -> None:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.wait_until(stopped.is_set)

        self.addAsyncCleanup(cleanup)
        await self.wait_until(entered.is_set)
        return task, queues.get_nowait()

    async def emit(self, events: ReplyEvents, *items: ReplyEvent) -> None:
        for item in items:
            events.put(item)
        await self.wait_until(events.empty)
        await asyncio.sleep(0)

    async def test_subagent_events_build_its_own_chat_and_stay_out_of_the_main_one(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", TASK_STARTED),
                        tagged(("text", "Look")), tagged(("text", "ing.")),
                        tagged(("tool_started", {"name": "edit", "detail": "storage.py"})),
                        tagged(("tool_diff", "-old\n+new")), tagged(("tool_finished", "edit")),
                        tagged(("usage", {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100})),
                        tagged(("text", "Found it.")))
        self.now = 105.0
        await self.emit(events, ("tool_finished", "task"), ("text", "Done"), ("complete", None))
        await asyncio.wait_for(task, 2)

        subagent, = self.owner.subagents
        self.assertEqual(rows(of=subagent), [
            ("user", "Find where sessions are saved."),
            ("assistant", "Looking."),
            ("tool", "● edit storage.py · finished"),
            ("assistant", "Found it."),
        ])
        tool = subagent["transcript"][2]
        assert tool["role"] == "tool"
        self.assertEqual(tool.get("diff"), "-old\n+new")
        self.assertFalse(any(entry.get("streaming") for entry in subagent["transcript"]))
        self.assertEqual((subagent["id"], subagent["description"], subagent["status"]),
                         ("call_task", "Find sessions", "finished"))
        self.assertEqual((subagent["duration"], subagent["tokens"]), (5.0, 100))

        main = rows(self.owner.transcript)
        self.assertIn(("tool", "● task Find sessions · finished"), main)
        self.assertFalse(any(text in content for _, content in main for text in ("Looking.", "Found it.", "storage.py")))
        self.assertIsNone(self.owner.last_usage)

    async def test_a_new_turn_hides_the_previous_turns_subagents_but_compaction_does_not(self) -> None:
        old: Subagent = {"id": "call_old", "description": "x", "status": "finished", "started_at": 0.0,
                         "duration": 1.0, "tokens": 0, "transcript": []}
        self.owner.subagents = [old]
        self.owner.panel_index = 1
        task, events = await self.start_turn()

        self.assertEqual((self.owner.panel_start, self.owner.panel_index), (1, None))
        self.assertEqual(self.owner.panel_agents(), [])
        await self.emit(events, ("tool_started", TASK_STARTED))
        self.assertEqual([agent["id"] for agent in self.owner.panel_agents()], ["call_task"])
        await self.emit(events, ("tool_finished", "task"), ("text", "Done"), ("complete", None))
        await asyncio.wait_for(task, 2)

        self.owner.panel_start = 0
        compaction = asyncio.create_task(self.owner.stream_reply("", compact_only=True))
        self.owner.pending_reply = compaction
        await asyncio.sleep(0)
        self.assertEqual(self.owner.panel_start, 0)
        await self.wait_until(lambda: self.stream.call_count == 2)
        self.assertNotIn("subagents", self.stream.call_args.kwargs)
        compaction.cancel()
        await asyncio.gather(compaction, return_exceptions=True)

    async def test_a_turn_passes_the_saved_subagent_settings(self) -> None:
        self.owner.subagent_settings = {"enabled": False, "budget": 3, "fetch": True}
        await self.start_turn()

        options = cast(dict[str, object], self.stream.call_args.kwargs)
        self.assertEqual((options["subagents"], options["subagent_budget"]), (False, 3))

    async def test_a_turn_passes_web_search_on_when_it_is_on(self) -> None:
        self.owner.search_settings = {"key": "tvly-key", "enabled": True}
        await self.start_turn()
        self.assertIs(cast(dict[str, object], self.stream.call_args.kwargs)["search"], True)

    async def test_a_turn_passes_web_search_off_when_it_is_off(self) -> None:
        self.owner.search_settings = {"key": "tvly-key", "enabled": False}
        await self.start_turn()
        self.assertIs(cast(dict[str, object], self.stream.call_args.kwargs)["search"], False)

    async def test_a_turn_passes_web_fetch_on_by_default(self) -> None:
        await self.start_turn()
        self.assertIs(cast(dict[str, object], self.stream.call_args.kwargs)["fetch"], True)

    async def test_a_turn_passes_web_fetch_off_when_it_is_off(self) -> None:
        self.owner.subagent_settings = {"enabled": True, "budget": 8, "fetch": False}
        await self.start_turn()
        self.assertIs(cast(dict[str, object], self.stream.call_args.kwargs)["fetch"], False)

    async def test_a_turn_passes_web_search_off_without_a_key(self) -> None:
        self.owner.search_settings = None
        await self.start_turn()
        self.assertIs(cast(dict[str, object], self.stream.call_args.kwargs)["search"], False)

    async def test_a_failed_task_marks_its_subagent_failed(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", TASK_STARTED), tagged(("text", "Trying")),
                        ("tool_failed", "task"), ("text", "It failed."), ("complete", None))
        await asyncio.wait_for(task, 2)

        subagent, = self.owner.subagents
        self.assertEqual(subagent["status"], "failed")
        self.assertIn(("tool", "● task Find sessions · failed"), rows(self.owner.transcript))

    async def test_cancelling_during_a_subagent_marks_it_and_its_running_tool_interrupted(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", TASK_STARTED),
                        tagged(("tool_started", {"name": "bash", "detail": "make test"})))
        self.now = 103.0
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        subagent, = self.owner.subagents
        self.assertEqual((subagent["status"], subagent["duration"]), ("interrupted", 3.0))
        self.assertEqual(rows(of=subagent)[-1], ("tool", "● bash make test · interrupted"))

    async def test_events_for_an_unknown_task_are_ignored(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, tagged(("text", "stray"), "call_other"), ("text", "Done"), ("complete", None))
        await asyncio.wait_for(task, 2)

        self.assertEqual(self.owner.subagents, [])
        self.assertNotIn("stray", str(rows(self.owner.transcript)))

    def test_clear_forgets_subagents(self) -> None:
        self.owner.subagents.append({"id": "call_task", "description": "x", "status": "finished", "started_at": 0.0,
                                     "duration": 1.0, "tokens": 0, "transcript": []})
        self.owner.header = Markdown("Vin")
        self.owner.handle_command("/clear")
        self.assertEqual(self.owner.subagents, [])


class ApplySubagentEventTests(unittest.TestCase):
    def subagent(self) -> Subagent:
        return {"id": "t", "description": "d", "status": "running", "started_at": 0.0, "duration": None,
                "tokens": 0, "transcript": [{"role": "user", "content": "p"}]}

    def test_failed_tool_and_non_edit_diff(self) -> None:
        subagent = self.subagent()
        events: list[ReplyEvent] = [("tool_started", "bash"), ("tool_diff", "ignored"), ("tool_failed", "bash")]
        for event in events:
            subagents.apply_subagent_event(subagent, event)

        self.assertEqual(rows(of=subagent)[-1], ("tool", "● bash · failed"))
        self.assertNotIn("diff", subagent["transcript"][-1])

    def test_text_after_finish_starts_a_new_message(self) -> None:
        subagent = self.subagent()
        subagents.apply_subagent_event(subagent, ("text", "One"))
        subagents.finish_subagent(subagent, "finished")
        subagents.apply_subagent_event(subagent, ("text", "Two"))

        self.assertEqual(rows(of=subagent), [("user", "p"), ("assistant", "One"), ("assistant", "Two")])


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
