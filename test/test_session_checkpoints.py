"""Conversation checkpoints using real temporary storage and no provider calls."""

from copy import deepcopy
import asyncio
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit.buffer import Buffer
from rich.text import Text

from vin.sessions import storage
from vin.sessions.records import new_session, snapshot_session
from vin.tui import app, conversations


class SessionCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.destination = self.root / "session-data"
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)
        self.owner = app.VinApp()
        self.owner.project_dir = self.project

    def checkpoint(self, **kwargs):
        helper = getattr(self.owner, "checkpoint_session", None)
        self.assertTrue(callable(helper), "Add VinApp.checkpoint_session(self) in tui/app.py.")
        return helper(**kwargs)


    def test_checkpoint_saves_current_state_and_keeps_independent_baseline_for_updates(self):
        self.owner.session_record = new_session(self.project, "Review the queue")
        self.owner.messages = [{"metadata": {"nested": ["original"]}}]
        self.owner.transcript = [
            {"role": "user", "content": "Review the queue"},
            {"role": "notice", "content": Text("Notice", style="yellow")},
            {"role": "assistant", "content": "Checking", "streaming": True},
        ]
        self.owner.active_turn = {"prompt": "Review the queue", "tools_used": False}
        original_messages = deepcopy(self.owner.messages)
        original_transcript = deepcopy(self.owner.transcript)

        with patch.object(conversations, "monotonic", return_value=10.0):
            self.assertTrue(self.checkpoint())

        session_id = self.owner.session_record["id"]
        saved = storage.load_session(self.project, session_id)
        baseline = deepcopy(saved)
        self.assertEqual(saved["messages"], original_messages)
        self.assertEqual(saved["active_turn"], self.owner.active_turn)
        self.assertEqual(saved, self.owner.saved_session_record)
        self.assertEqual(self.owner.messages, original_messages)
        self.assertEqual(self.owner.transcript, original_transcript)
        self.assertEqual(self.owner.last_session_checkpoint, 10.0)

        self.owner.messages[0]["metadata"]["nested"].append("later")
        self.owner.transcript[-1]["content"] += " more"
        self.owner.session_record["name"] = "Renamed in memory"
        self.owner.active_turn["tools_used"] = True
        self.assertEqual(self.owner.saved_session_record, baseline)
        self.assertEqual(storage.load_session(self.project, session_id), baseline)

        with patch.object(conversations, "monotonic", return_value=11.0):
            self.assertTrue(self.checkpoint())

        saved = storage.load_session(self.project, session_id)
        self.assertEqual(saved["id"], session_id)
        self.assertEqual(saved["name"], "Renamed in memory")
        self.assertEqual(saved["messages"], self.owner.messages)
        self.assertEqual(saved["transcript"][-1]["content"], "Checking more")
        self.assertTrue(saved["active_turn"]["tools_used"])
        self.assertEqual(self.owner.saved_session_record, saved)
        self.assertEqual(self.owner.last_session_checkpoint, 11.0)

    def test_routine_checkpoints_coalesce_for_500ms_without_extending_the_deadline(self):
        self.owner.session_record = new_session(self.project, "Review the queue")
        self.owner.transcript = [{"role": "assistant", "content": "First", "streaming": True}]
        self.owner.active_turn = {"prompt": "Review the queue", "tools_used": False}
        with patch.object(conversations, "monotonic", return_value=0.0):
            self.assertTrue(self.checkpoint())
        path = storage.session_path(self.project, self.owner.session_record["id"])
        previous = path.read_bytes()
        baseline = deepcopy(self.owner.saved_session_record)
        self.owner.transcript[0]["content"] = "Latest partial response"

        for now in (0.25, 0.499):
            with self.subTest(now=now):
                with patch.object(conversations, "monotonic", return_value=now):
                    with patch.object(conversations, "snapshot_session", wraps=conversations.snapshot_session) as builder:
                        self.assertFalse(self.checkpoint())
                        builder.assert_not_called()
                self.assertEqual(path.read_bytes(), previous)
                self.assertEqual(self.owner.saved_session_record, baseline)
                self.assertEqual(self.owner.last_session_checkpoint, 0.0)

        with patch.object(conversations, "monotonic", return_value=0.5):
            self.assertTrue(self.checkpoint())
        saved = storage.load_session(self.project, self.owner.session_record["id"])
        self.assertEqual(saved["transcript"][0]["content"], "Latest partial response")
        self.assertEqual(self.owner.last_session_checkpoint, 0.5)

    def test_forced_checkpoint_flushes_final_state_inside_the_500ms_window(self):
        self.owner.session_record = new_session(self.project, "Review the queue")
        self.owner.transcript = [{"role": "assistant", "content": "Partial", "streaming": True}]
        self.owner.active_turn = {"prompt": "Review the queue", "tools_used": False}
        with patch.object(conversations, "monotonic", return_value=10.0):
            self.assertTrue(self.checkpoint())

        self.owner.transcript[0].update(content="Finished response", streaming=False)
        self.owner.messages = [{"role": "assistant", "content": "Finished response"}]
        self.owner.active_turn = None
        with patch.object(conversations, "monotonic", return_value=10.1):
            self.assertTrue(self.checkpoint(force=True))

        saved = storage.load_session(self.project, self.owner.session_record["id"])
        self.assertEqual(saved["transcript"][0]["content"], "Finished response")
        self.assertFalse(saved["transcript"][0]["streaming"])
        self.assertEqual(saved["messages"], self.owner.messages)
        self.assertIsNone(saved["active_turn"])
        self.assertEqual(self.owner.last_session_checkpoint, 10.1)

    def start_saved_conversation(self):
        self.owner.session_record = new_session(self.project, "Review the queue")
        self.owner.messages = [{"role": "user", "content": "Earlier request"}]
        self.owner.transcript = [{"role": "assistant", "content": "Earlier reply"}]
        self.owner.active_turn = {"prompt": "Review the queue", "tools_used": False}
        with patch.object(conversations, "monotonic", return_value=1.0):
            self.assertTrue(self.checkpoint(force=True))
        return storage.session_path(self.project, self.owner.session_record["id"])

    def test_failed_save_preserves_checkpoint_suppresses_duplicate_notices_and_can_retry(self):
        path = self.start_saved_conversation()
        previous_bytes = path.read_bytes()
        previous_baseline = deepcopy(self.owner.saved_session_record)
        self.owner.transcript[0]["content"] = "Unsaved progress"
        current_messages = deepcopy(self.owner.messages)
        current_active = deepcopy(self.owner.active_turn)

        with patch.object(conversations, "save_session", side_effect=RuntimeError("Disk unavailable")):
            for now in (2.0, 3.0):
                with patch.object(conversations, "monotonic", return_value=now):
                    self.assertFalse(self.checkpoint(force=True))

        self.assertEqual(path.read_bytes(), previous_bytes)
        self.assertEqual(self.owner.saved_session_record, previous_baseline)
        self.assertEqual(self.owner.session_record, previous_baseline)
        self.assertEqual(self.owner.messages, current_messages)
        self.assertEqual(self.owner.active_turn, current_active)
        self.assertEqual(self.owner.transcript[0]["content"], "Unsaved progress")
        notices = [entry for entry in self.owner.transcript if entry["role"] == "notice"]
        self.assertEqual(len(notices), 1)
        self.assertIn("not saved", notices[0]["content"].plain.lower())
        self.assertIn("Disk unavailable", notices[0]["content"].plain)
        self.assertEqual(self.owner.session_save_error, notices[0]["content"].plain)

        with patch.object(conversations, "monotonic", return_value=4.0):
            self.assertTrue(self.checkpoint(force=True))
        saved = storage.load_session(self.project, self.owner.session_record["id"])
        self.assertEqual(saved["transcript"][0]["content"], "Unsaved progress")
        self.assertEqual(self.owner.saved_session_record, saved)
        self.assertIsNone(self.owner.session_save_error)

    def test_conflict_leaves_other_writer_file_and_live_conversation_intact(self):
        path = self.start_saved_conversation()
        baseline = deepcopy(self.owner.saved_session_record)
        external = dict(baseline, name="Saved elsewhere")
        storage.save_session(self.project, external, expected_record=baseline)
        external_bytes = path.read_bytes()
        self.owner.transcript[0]["content"] = "Keep live progress"
        self.owner.queued_prompts.append("Next request")

        with patch.object(conversations, "monotonic", return_value=2.0):
            self.assertFalse(self.checkpoint(force=True))

        self.assertEqual(path.read_bytes(), external_bytes)
        self.assertEqual(self.owner.saved_session_record, baseline)
        self.assertEqual(self.owner.transcript[0]["content"], "Keep live progress")
        self.assertEqual(list(self.owner.queued_prompts), ["Next request"])
        self.assertIn("changed", self.owner.session_save_error.lower())


class SessionPromptCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()
        self.destination = Path(temporary.name) / "session-data"
        self.enterContext(patch.object(storage, "SESSIONS_DIR", self.destination))
        self.enterContext(patch.object(app, "patch_stdout", side_effect=lambda **kwargs: nullcontext()))
        self.owner = app.VinApp()
        self.owner.project_dir = self.project
        self.owner.header = Text("Vin")
        self.owner.session = SimpleNamespace(message="", default_buffer=Buffer())

    async def run_prompts(self, prompts, reply):
        receiver = None
        fake = SimpleNamespace(is_done=False, result=None, invalidate=lambda: None)

        def start(coroutine):
            nonlocal receiver
            receiver = asyncio.create_task(coroutine)
            return receiver

        def exit_app(*, result):
            fake.is_done = True
            fake.result = result

        async def run_async(*, pre_run):
            pre_run()
            await asyncio.wait_for(receiver, timeout=2)
            return fake.result

        fake.create_background_task = start
        fake.exit = exit_app
        fake.run_async = run_async
        self.owner.app = fake
        self.owner.session.app = fake
        for prompt in prompts:
            self.owner.submitted.put_nowait(prompt)
        with patch.object(self.owner, "stream_reply", side_effect=reply):
            return await self.owner.chat_loop()

    async def test_prompt_is_saved_before_reply_and_clear_starts_another_identity(self):
        observed = []

        async def reply(text):
            records = storage.list_sessions(self.project)
            self.assertTrue(records, "The prompt must be saved before the reply starts.")
            record = storage.load_session(self.project, self.owner.session_record["id"])
            self.assertEqual(record["active_turn"], {"prompt": text, "tools_used": False})
            self.assertEqual(record["transcript"][-1], {"role": "user", "content": text})
            self.assertEqual(record["messages"], self.owner.messages)
            observed.append(record)
            self.owner.messages.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Reply"},
            ])
            self.owner.transcript.append({"role": "assistant", "content": "Reply"})
            self.owner.pending_reply = None

        result = await self.run_prompts(
            ["First request", "Follow-up", "/clear", "New conversation", "/quit"], reply
        )

        self.assertEqual(result, "/quit")
        self.assertEqual(len(observed), 3)
        self.assertEqual(observed[0]["id"], observed[1]["id"])
        self.assertNotEqual(observed[0]["id"], observed[2]["id"])
        self.assertEqual([record["name"] for record in observed],
                         ["First request", "First request", "New conversation"])
        self.assertEqual(observed[0]["messages"], [])
        self.assertEqual(len(observed[1]["messages"]), 2)
        self.assertEqual(observed[2]["messages"], [])
        self.assertEqual(len(storage.list_sessions(self.project)), 2)

    async def test_commands_alone_do_not_create_a_saved_conversation(self):
        async def unexpected_reply(text):
            self.fail(f"A command started a reply: {text}")

        self.assertEqual(await self.run_prompts(["/clear", "/quit"], unexpected_reply), "/quit")
        self.assertFalse(self.destination.exists())
        self.assertIsNone(self.owner.session_record)


    async def test_resume_arguments_reach_lookup_and_preserve_full_title_before_followup(self):
        history = [{"role": "user", "content": "Previous request"},
                   {"role": "assistant", "content": "Previous answer"}]
        record = snapshot_session(
            new_session(self.project, "Saved conversation with spaces"), history,
            [{"role": "user", "content": "Previous request"},
             {"role": "assistant", "content": "Previous answer"}], active_turn=None,
        )
        storage.save_session(self.project, record)
        self.owner.session.default_buffer.text = "Unsent draft"
        draft = self.owner.session.default_buffer.document
        observed = []

        async def reply(text):
            self.assertEqual(text, "Follow up", "Resume arguments must not become model prompts.")
            self.assertEqual(self.owner.messages, history)
            self.assertEqual(self.owner.session_record["id"], record["id"])
            observed.append(text)
            self.owner.pending_reply = None

        with patch.object(self.owner, "choose_inline", side_effect=AssertionError("Exact lookup must not open a picker")):
            result = await self.run_prompts([
                f"/resume {record['name']}",
                f"/resume {record['id']}",
                f"/resume\t  {record['name']}  ",
                "Follow up", "/quit",
            ], reply)

        self.assertEqual(result, "/quit")
        self.assertEqual(observed, ["Follow up"])
        self.assertEqual(self.owner.session.default_buffer.document, draft)
        self.assertEqual(len(storage.list_sessions(self.project)), 1)


    async def test_rename_command_persists_title_and_can_resume_it_after_clear(self):
        history = [{"role": "user", "content": "Previous request"},
                   {"role": "assistant", "content": "Previous answer"}]
        record = snapshot_session(
            new_session(self.project, "Original title"), history,
            [{"role": "user", "content": "Previous request"},
             {"role": "assistant", "content": "Previous answer"}], active_turn=None,
        )
        storage.save_session(self.project, record)
        self.owner.session.default_buffer.text = "Unsent draft"
        draft = self.owner.session.default_buffer.document

        async def unexpected_reply(text):
            self.fail(f"A session command started a model reply: {text}")

        result = await self.run_prompts([
            f"/resume {record['id']}", "/rename\t  New   session title  ", "/quit",
        ], unexpected_reply)

        self.assertEqual(result, "/quit")
        saved = storage.load_session(self.project, record["id"])
        self.assertEqual(saved["name"], "New session title")
        self.assertEqual(saved["messages"], history)
        self.assertTrue(any(str(entry["content"]) == "Renamed: New session title"
                            for entry in self.owner.transcript))

        async def followup(text):
            self.assertEqual(text, "Follow up")
            self.assertEqual(self.owner.messages, history)
            self.assertEqual(self.owner.session_record["id"], record["id"])
            self.assertEqual(self.owner.session_record["name"], "New session title")
            self.owner.pending_reply = None

        with patch.object(self.owner, "choose_inline", side_effect=AssertionError("An exact renamed title must not open a picker")):
            result = await self.run_prompts(
                ["/clear", "/resume New session title", "Follow up", "/quit"], followup,
            )
        self.assertEqual(result, "/quit")
        self.assertEqual(self.owner.session.default_buffer.document, draft)
        self.assertEqual([item["id"] for item in storage.list_sessions(self.project)], [record["id"]])


class SessionReplyCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", Path(temporary.name) / "sessions"))
        # Finalization must force a save even within the initial checkpoint window.
        self.clock = self.enterContext(patch.object(conversations, "monotonic", return_value=100.0))
        self.enterContext(patch.object(app, "monotonic", self.clock))
        self.prompt = "Review the queue"
        self.prepare_conversation()

    def prepare_conversation(self):
        self.owner = app.VinApp()
        self.owner.project_dir = self.project
        buffer = Buffer()
        buffer.text = "next draft"
        buffer.cursor_position = 4
        self.owner.session = SimpleNamespace(app=Mock(), default_buffer=buffer)
        self.owner.messages = [{"role": "user", "content": "Earlier question"},
                               {"role": "assistant", "content": "Earlier answer"}]
        self.previous = deepcopy(self.owner.messages)
        self.draft = buffer.document
        self.owner.transcript = [{"role": "user", "content": self.prompt}]
        self.owner.session_record = new_session(self.project, self.prompt)
        self.owner.active_turn = {"prompt": self.prompt, "tools_used": False}
        self.assertTrue(self.owner.checkpoint_session(force=True))
        self.session_id = self.owner.session_record["id"]

    async def run_turn(self, emitted):
        def worker(credentials, model, request, events, cancel_event):
            if any(kind == "tool_started" for kind, _ in emitted):
                events.tools_used.set()
            for event in emitted:
                events.put(event)

        with patch.object(app, "stream_chatgpt_response", side_effect=worker):
            task = asyncio.create_task(self.owner.stream_reply(self.prompt))
            self.owner.pending_reply = task
            return await asyncio.wait_for(task, timeout=2)

    def saved(self):
        return storage.load_session(self.project, self.session_id)

    async def test_completed_reply_persists_full_context_footer_and_idle_marker(self):
        context = self.previous + [
            {"role": "user", "content": self.prompt},
            {"type": "reasoning", "encrypted_content": "fake-reasoning"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Final response"},
            ]},
        ]
        self.assertIsNone(await self.run_turn([("text", "Final response"), ("complete", context)]))

        saved = self.saved()
        self.assertIsNone(saved["active_turn"])
        self.assertIsNone(self.owner.active_turn)
        self.assertEqual(saved["messages"], context)
        assistant = next(entry for entry in saved["transcript"] if entry["role"] == "assistant")
        self.assertEqual(assistant["content"], "Final response")
        self.assertFalse(assistant["streaming"])
        self.assertIn("Worked for", saved["transcript"][-1]["content"]["text"])
        self.assertEqual(self.owner.saved_session_record, saved)
        self.assertIsNone(self.owner.pending_reply)
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)

    async def test_error_and_rate_limit_persist_failed_tool_diff_and_caution_context(self):
        for kind in ("error", "rate_limit"):
            with self.subTest(kind=kind):
                self.prepare_conversation()
                diff = "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"
                result = await self.run_turn([
                    ("tool_started", {"name": "edit", "detail": "example.py"}),
                    ("tool_diff", diff), ("text", "Partial response"),
                    (kind, "Provider unavailable"),
                ])
                self.assertEqual(result, kind)
                saved = self.saved()
                self.assertIsNone(saved["active_turn"])
                tool = next(entry for entry in saved["transcript"] if entry["role"] == "tool")
                self.assertEqual(tool["status"], "failed")
                self.assertEqual(tool["diff"], diff)
                self.assertEqual(saved["messages"][:-2], self.previous)
                self.assertEqual(saved["messages"][-2], {"role": "user", "content": self.prompt})
                self.assertIn("Changes may already exist", saved["messages"][-1]["content"])
                self.assertIn("Failed after", saved["transcript"][-1]["content"]["text"])
                self.assertEqual(self.owner.session.default_buffer.document, self.draft)

    async def test_cancelled_reply_saves_interruption_and_ignores_late_worker_events(self):
        stopped = Event()

        def worker(credentials, model, request, events, cancel_event):
            try:
                events.tools_used.set()
                events.put(("tool_started", {"name": "read", "detail": "example.py"}))
                events.put(("text", "Partial response"))
                cancel_event.wait(2)
                events.put(("text", "late result"))
                events.put(("complete", None))
            finally:
                stopped.set()

        with patch.object(app, "stream_chatgpt_response", side_effect=worker):
            task = asyncio.create_task(self.owner.stream_reply(self.prompt))
            self.owner.pending_reply = task
            try:
                async with asyncio.timeout(2):
                    while not any(entry.get("content") == "Partial response"
                                  for entry in self.owner.transcript):
                        await asyncio.sleep(0.001)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                async with asyncio.timeout(2):
                    while not stopped.is_set():
                        await asyncio.sleep(0.001)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        saved = self.saved()
        self.assertIsNone(saved["active_turn"])
        tool = next(entry for entry in saved["transcript"] if entry["role"] == "tool")
        self.assertEqual(tool["status"], "interrupted")
        assistant = [entry["content"] for entry in saved["transcript"] if entry["role"] == "assistant"]
        self.assertEqual(assistant, ["Partial response"])
        self.assertIn("Changes may already exist", saved["messages"][-1]["content"])
        self.assertIn("Interrupted after", saved["transcript"][-1]["content"]["text"])
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


    async def start_controlled_turn(self):
        queues = SimpleQueue()
        release, stopped = Event(), Event()

        def worker(credentials, model, request, events, cancel_event):
            try:
                queues.put(events)
                release.wait(3)
            finally:
                stopped.set()

        self.enterContext(patch.object(app, "stream_chatgpt_response", side_effect=worker))
        task = asyncio.create_task(self.owner.stream_reply(self.prompt))
        self.owner.pending_reply = task

        async def cleanup():
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.wait_until(stopped.is_set)

        self.addAsyncCleanup(cleanup)
        await self.wait_until(lambda: not queues.empty())
        return task, queues.get()

    async def wait_until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.001)

    async def test_tool_start_diff_and_finish_are_saved_while_reply_is_active(self):
        task, events = await self.start_controlled_turn()
        events.tools_used.set()
        events.put(("tool_started", {"name": "edit", "detail": "example.py"}))
        await self.wait_until(lambda: self.owner.tool_status == ("edit", "running"))

        saved = self.saved()
        self.assertEqual(saved["active_turn"], {"prompt": self.prompt, "tools_used": True})
        self.assertEqual(saved["transcript"][-1]["status"], "running")
        self.assertEqual(saved["messages"], self.previous)
        self.assertFalse(task.done())

        diff = "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"
        events.put(("tool_diff", diff))
        await self.wait_until(lambda: self.owner.transcript[-1].get("diff") == diff)
        self.assertEqual(self.saved()["transcript"][-1]["diff"], diff)

        events.put(("tool_finished", {"name": "edit", "detail": "example.py"}))
        await self.wait_until(lambda: self.owner.tool_status == ("edit", "finished"))
        saved = self.saved()
        self.assertEqual(saved["transcript"][-1]["status"], "finished")
        self.assertEqual(saved["transcript"][-1]["diff"], diff)
        self.assertTrue(saved["active_turn"]["tools_used"])
        self.assertFalse(task.done())
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)

        events.put(("complete", None))
        await asyncio.wait_for(task, timeout=2)
        self.assertIsNone(self.saved()["active_turn"])


    def observe_idle_poll(self, events):
        idle = asyncio.Event()
        get_nowait = events.get_nowait

        def observed_get():
            try:
                return get_nowait()
            except Empty:
                idle.set()
                raise

        events.get_nowait = observed_get
        return idle

    async def test_paused_stream_flushes_buffered_text_once_at_the_deadline(self):
        task, events = await self.start_controlled_turn()
        idle = self.observe_idle_poll(events)
        self.clock.return_value = 100.1
        with patch.object(conversations, "save_session", wraps=conversations.save_session) as writer:
            events.put(("text", "A buffered burst"))
            await self.wait_until(lambda: self.owner.reply_entry["content"] == "A buffered burst")
            writer.assert_not_called()

            self.clock.return_value = 100.5
            idle.clear()
            await asyncio.wait_for(idle.wait(), timeout=2)
            saved = self.saved()
            self.assertEqual(saved["transcript"][-1]["content"], "A buffered burst")
            self.assertIsNotNone(saved["active_turn"])
            self.assertEqual(writer.call_count, 1)
            self.assertFalse(task.done())

            for now in (101.0, 101.5):
                self.clock.return_value = now
                idle.clear()
                await asyncio.wait_for(idle.wait(), timeout=2)
                self.assertEqual(writer.call_count, 1, "Clean idle polls must not rewrite saved text.")

            events.put(("complete", None))
            await asyncio.wait_for(task, timeout=2)
            self.assertEqual(writer.call_count, 2)
            self.assertIsNone(self.saved()["active_turn"])
