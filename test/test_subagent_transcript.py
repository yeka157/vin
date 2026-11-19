"""Subagent events build each subagent's own chat while the main transcript keeps one task row."""

import asyncio
from collections.abc import Callable
from queue import SimpleQueue
from threading import Event
from typing import Literal, cast
import unittest
from unittest.mock import patch

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown

from vin.provider.events import ReplyEvent, ReplyEvents, ToolStarted
from vin.tui import app as app_module, bindings, layout, subagents, transcript
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
        self.owner.subagent_settings = {"enabled": False, "budget": 3}
        await self.start_turn()

        options = cast(dict[str, object], self.stream.call_args.kwargs)
        self.assertEqual((options["subagents"], options["subagent_budget"]), (False, 3))

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


if __name__ == "__main__":
    unittest.main()
