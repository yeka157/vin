"""Tool status in the editable composer, including cancellation and layout."""

import asyncio
from threading import Event
from collections.abc import Callable
from typing import Literal
import unittest
from unittest.mock import patch

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from vin.tui import app as app_module, bindings
from vin.provider.events import ReplyEvents, ToolEventKind
from vin.types import Credentials, History
from rich.text import Text
from test.tui_typing_helpers import key_event


class ToolStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.owner = app_module.VinApp()
        self.assertTrue(hasattr(self.owner, "tool_status"), "Initialize VinApp.tool_status to None.")
        self.assertIsNone(self.owner.tool_status)
        self.draft = Document("next draft", cursor_position=3)
        self.owner.session = PromptSession(input=DummyInput(), output=DummyOutput())
        self.owner.session.default_buffer = Buffer(document=self.draft)

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    def start(self) -> asyncio.Task[str | None]:
        task = asyncio.create_task(self.owner.stream_reply("Inspect the file"))
        self.owner.pending_reply = task
        return task

    async def test_repeated_edits_retain_their_own_diffs_separately_from_assistant_text(self) -> None:
        first = "--- f\n+++ f\n@@ -1 +1 @@\n-before\n+middle\n"
        second = "--- f\n+++ f\n@@ -1 +1 @@\n-middle\n+after\n"

        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
            events.tools_used.set()
            for diff in (first, second):
                events.put(("tool_started", {"name": "edit", "detail": "f"}))
                events.put(("tool_diff", diff))
                events.put(("tool_finished", "edit"))
            events.put(("text", "Done"))
            events.put(("complete", None))

        with patch.object(app_module, "stream_chatgpt_response", side_effect=worker):
            await asyncio.wait_for(self.start(), 2)
        rows = [row for row in self.owner.transcript if row["role"] == "tool"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.get("diff") for row in rows], [first, second])
        self.assertTrue(all(str(row["content"]) == "● edit f · finished" for row in rows))
        answers = [row["content"] for row in self.owner.transcript if row["role"] == "assistant"]
        self.assertEqual(answers, ["Done"])
        self.assertEqual(self.owner.messages[-1]["content"], "Done")
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


    async def test_running_and_terminal_status_preserve_editable_draft(self) -> None:
        terminals: list[tuple[Literal["tool_finished", "tool_failed"], str]] = [("tool_finished", "finished"), ("tool_failed", "failed")]
        terminal: ToolEventKind
        for terminal, expected in terminals:
            with self.subTest(terminal=terminal):
                previous_rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                release_status, release_answer, stopped = Event(), Event(), Event()
                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
                    try:
                        events.tools_used.set()
                        events.put(("tool_started", "read"))
                        release_status.wait(2)
                        terminal_event: tuple[ToolEventKind, str] = (terminal, "read")
                        events.put(terminal_event)
                        release_answer.wait(2)
                        events.put(("text", "Done"))
                        events.put(("complete", None))
                    finally:
                        stopped.set()
                with patch.object(app_module, "stream_chatgpt_response", side_effect=worker):
                    task = self.start()
                    try:
                        await self.wait_until(lambda: self.owner.tool_status == ("read", "running"))
                        rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                        self.assertEqual(len(rows), len(previous_rows) + 1)
                        tool_row = rows[-1]
                        self.assertEqual(tool_row.get("status"), "running")
                        self.assertIn("read", str(tool_row["content"]))
                        self.assertIn("running", str(tool_row["content"]))
                        self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                        typed = Document("typing while tool runs", cursor_position=5)
                        self.owner.session.default_buffer.set_document(typed)
                        release_status.set()
                        await self.wait_until(lambda: self.owner.tool_status == ("read", expected))
                        rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                        self.assertEqual(len(rows), len(previous_rows) + 1, "Update the existing row rather than appending another.")
                        self.assertIs(rows[-1], tool_row)
                        self.assertIn(expected, str(tool_row["content"]))
                        self.assertEqual(tool_row.get("status"), expected)
                        self.assertNotIn("running", str(tool_row["content"]))
                        self.assertEqual(self.owner.session.default_buffer.document, typed)
                        release_answer.set()
                        await asyncio.wait_for(task, 2)
                        self.assertEqual(self.owner.tool_status, ("read", expected))
                        self.assertEqual(self.owner.session.default_buffer.document, typed)
                    finally:
                        release_status.set()
                        release_answer.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        await self.wait_until(stopped.is_set)
                        self.owner.session.default_buffer.set_document(self.draft)

    async def test_task_and_keyboard_cancellation_finish_running_status(self) -> None:
        for keyboard in (False, True):
            with self.subTest(keyboard=keyboard):
                release, stopped = Event(), Event()
                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
                    try:
                        events.tools_used.set()
                        events.put(("tool_started", {"name": "bash", "detail": "printf '[red]hello[/red]'"}))
                        release.wait(2)
                        events.put(("tool_finished", "bash"))
                    finally:
                        stopped.set()
                with patch.object(app_module, "stream_chatgpt_response", side_effect=worker):
                    task = self.start()
                    try:
                        await self.wait_until(lambda: self.owner.tool_status == ("bash", "running"))
                        tool_row = [entry for entry in self.owner.transcript if entry["role"] == "tool"][-1]
                        if keyboard:
                            keys = bindings.create_bindings(self.owner)
                            keys.get_bindings_for_keys((Keys.ControlC,))[-1].handler(
                                key_event(self.owner.session.app, Keys.ControlC))
                        else:
                            task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        self.assertEqual(self.owner.tool_status, ("bash", "failed"))
                        self.assertIn("interrupted", str(tool_row["content"]))
                        self.assertEqual(tool_row.get("status"), "interrupted")
                        self.assertIn("printf '[red]hello[/red]'", str(tool_row["content"]))
                        self.assertNotIn("running", str(tool_row["content"]))
                        content = tool_row["content"]
                        assert isinstance(content, Text)
                        self.assertEqual(content.style, "yellow")
                        interrupted_content = str(tool_row["content"])
                        release.set()
                        await self.wait_until(stopped.is_set)
                        self.assertEqual(self.owner.tool_status, ("bash", "failed"))
                        self.assertEqual(str(tool_row["content"]), interrupted_content)
                        self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        await self.wait_until(stopped.is_set)
