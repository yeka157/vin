"""Approved activity/timing UI contracts; application implementation stays with Kevin.

These tests use the existing worker queue and read the visible transcript. They
do not prescribe the shape of tool/timing records inside the application.
"""

import asyncio
from queue import SimpleQueue
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from vin.tui import app as app_module, bindings, layout, transcript


class ActivityTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner = app_module.VinApp()
        self.owner.session = SimpleNamespace(
            app=Mock(), default_buffer=Buffer(document=Document("next draft", cursor_position=3)),
        )
        self.now = 100.0
        for module in (app_module, bindings, layout, transcript):
            self.enterContext(patch.object(module, "monotonic", side_effect=lambda: self.now, create=True))

    async def wait_until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def start_turn(self, text="Inspect the file"):
        entered, release, stopped = Event(), Event(), Event()
        queues = SimpleQueue()

        def worker(credentials, model, messages, events, cancel_event):
            try:
                queues.put(events)
                entered.set()
                release.wait(5)
            finally:
                stopped.set()

        self.enterContext(patch.object(app_module, "stream_chatgpt_response", side_effect=worker))
        self.owner.transcript.append({"role": "user", "content": text})
        task = asyncio.create_task(self.owner.stream_reply(text))
        self.owner.pending_reply = task

        async def cleanup():
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.wait_until(stopped.is_set)

        self.addAsyncCleanup(cleanup)
        await self.wait_until(entered.is_set)
        return task, queues.get_nowait()

    async def emit(self, events, *items):
        for kind, value in items:
            if kind == "tool_started":
                events.tools_used.set()
            events.put((kind, value))
        await self.wait_until(events.empty)

    def visible(self, width=100):
        content = self.owner.transcript_control.create_content(width, 500)
        return "\n".join(
            "".join(fragment[1] for fragment in content.get_line(line))
            for line in range(content.line_count)
        )

    def assert_live_tail(self, after, elapsed):
        text = self.visible()
        self.assertEqual(text.count("Thinking…"), 1, "Keep exactly one live indicator in the transcript.")
        self.assertIn(f"Thinking… ({elapsed})", text)
        self.assertGreater(text.index("Thinking…"), text.index(after))
        self.assertNotIn("Worked for", text, "The completed footer must not appear during work.")
        self.assertEqual(self.owner.session.default_buffer.text, "next draft")
        return text


    async def test_live_indicator_stays_below_streaming_text_and_tool_activity(self):
        _, events = await self.start_turn()
        await self.emit(events, ("text", "Checking the workspace."))
        self.now = 102.0
        self.assert_live_tail("Checking the workspace.", "2s")
        await self.emit(events, ("tool_started", "read"))
        self.now = 104.0
        self.assert_live_tail("read", "4s")
        await self.emit(events, ("tool_finished", "read"), ("text", "The file is valid."))
        self.now = 106.0
        self.assert_live_tail("The file is valid.", "6s")

    async def test_completed_turn_replaces_indicator_with_one_frozen_footer(self):
        task, events = await self.start_turn()
        await self.emit(events, ("text", "Verified."))
        self.now = 1542.0
        await self.emit(events, ("complete", None))
        await asyncio.wait_for(task, 2)
        finished = self.visible()
        self.assertNotIn("Thinking…", finished)
        self.assertEqual(finished.count("Worked for 24m 2s"), 1)
        self.assertGreater(finished.index("Worked for"), finished.index("Verified."))
        self.assertRegex(finished, r"done \d{1,2}:\d{2} [AP]M")
        self.now = 2000.0
        self.assertEqual(self.visible(), finished, "Finished timing must not keep ticking.")


    async def test_failed_tool_and_request_do_not_show_a_success_footer(self):
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", "read"), ("tool_failed", "read"))
        self.now = 107.0
        await self.emit(events, ("error", "The request failed."))
        await asyncio.wait_for(task, 2)
        text = self.visible().lower()
        self.assertIn("1 tool call · 1 failed", text)
        self.assertIn("failed", text)
        self.assertIn("7s", text)
        self.assertNotIn("worked for", text)
        self.assertNotIn("thinking…", text)


    async def test_keyboard_cancellation_freezes_interruption_footer_and_discards_late_events(self):
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", "bash"))
        self.now = 109.0
        keys = bindings.create_bindings(self.owner)
        keys.get_bindings_for_keys((Keys.ControlC,))[-1].handler(
            SimpleNamespace(app=self.owner.session.app)
        )
        with self.assertRaises(asyncio.CancelledError):
            await task
        cancelled = self.visible()
        self.assertIn("interrupted", cancelled.lower())
        self.assertEqual(cancelled.count("Interrupted after"), 1)
        self.assertIn("9s", cancelled)
        self.assertNotIn("Worked for", cancelled)
        self.assertNotIn("Thinking…", cancelled)
        events.put(("tool_finished", "bash"))
        events.put(("text", "Late result must stay hidden."))
        self.now = 300.0
        self.assertEqual(self.visible(), cancelled)
        self.assertEqual(self.owner.session.default_buffer.document, Document("next draft", cursor_position=3))


if __name__ == "__main__":
    unittest.main()
