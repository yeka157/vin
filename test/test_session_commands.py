"""Session commands with temporary storage and no provider calls."""

import asyncio
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from prompt_toolkit.buffer import Buffer
from rich.text import Text

from vin.sessions import storage
from vin.sessions.records import new_session, snapshot_session
from vin.sessions.transcript import encode_transcript
from vin.tui import app
from vin.tui.app import VinApp


class SessionResumeCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        location = patch.object(storage, "SESSIONS_DIR", self.root / "sessions")
        location.start()
        self.addCleanup(location.stop)
        self.owner = VinApp()
        self.owner.project_dir = self.project
        self.owner.session_record = new_session(self.project, "Current conversation")
        self.owner.messages = [{"role": "user", "content": "Keep current context"}]
        self.owner.transcript = [{"role": "assistant", "content": Text("Keep display")}]
        self.owner.last_session_checkpoint = 10.0
        self.owner.session_save_error = "Previous save warning"
        self.owner.buffer = Buffer()
        self.owner.buffer.text = "Unsent draft"
        self.owner.buffer.cursor_position = 4
        self.picker = AsyncMock(side_effect=AssertionError("No picker should open here"))
        picker = patch.object(self.owner, "choose_inline", self.picker)
        picker.start()
        self.addCleanup(picker.stop)

    def current_state(self):
        return deepcopy((
            self.owner.session_record, self.owner.saved_session_record,
            self.owner.messages, self.owner.transcript, self.owner.active_turn,
            self.owner.last_session_checkpoint, self.owner.session_save_error,
            self.owner.buffer.text, self.owner.buffer.cursor_position,
        ))

    async def resume(self, *args):
        helper = getattr(self.owner, "resume_conversation", None)
        self.assertTrue(callable(helper), "Add async VinApp.resume_conversation(self).")
        # Keep a missing query parameter an intentional assertion failure.
        if args:
            import inspect
            try:
                inspect.signature(helper).bind(*args)
            except TypeError:
                self.fail("Allow an optional title/ID argument in resume_conversation.")
        return await helper(*args)


    async def test_picker_distinguishes_duplicate_names_and_cancel_keeps_current_conversation(self):
        older = new_session(self.project, "Same name")
        older.update(created_at="2026-10-24T10:00:00+00:00",
                     updated_at="2026-10-24T10:00:00+00:00")
        newer = new_session(self.project, "Same name")
        newer.update(created_at="2026-10-24T10:00:00+00:00",
                     updated_at="2026-10-25T12:00:00+00:00")
        for record in (older, newer):
            storage.save_session(self.project, record)
        other_project = self.root / "another-project"
        other_project.mkdir()
        storage.save_session(other_project, new_session(other_project, "Another project"))
        before = self.current_state()
        saved_bytes = {
            record["id"]: storage.session_path(self.project, record["id"]).read_bytes()
            for record in (older, newer)
        }
        shown = []

        async def cancel_selection(message, options, default=None):
            shown.extend(options)
            raise asyncio.CancelledError()

        self.picker.side_effect = cancel_selection

        self.assertIsNone(await self.resume())

        self.assertEqual([value for value, _ in shown], [newer["id"], older["id"]])
        for (_, label), record in zip(shown, (newer, older)):
            label = str(label)
            self.assertIn("Same name", label)
            self.assertIn(record["updated_at"], label)
            self.assertIn(record["id"][:8], label)
        self.assertNotEqual(str(shown[0][1]), str(shown[1][1]))
        self.assertEqual(self.current_state(), before)
        for session_id, original in saved_bytes.items():
            self.assertEqual(storage.session_path(self.project, session_id).read_bytes(), original)

    async def test_app_task_cancellation_during_picker_is_not_swallowed(self):
        storage.save_session(self.project, new_session(self.project, "Saved conversation"))
        before = self.current_state()
        entered = asyncio.Event()

        async def wait_for_selection(message, options, default=None):
            entered.set()
            await asyncio.Future()

        self.picker.side_effect = wait_for_selection
        task = asyncio.create_task(self.resume())
        try:
            # An absent picker returns immediately; fail without an arbitrary timeout.
            for _ in range(5):
                if entered.is_set() or task.done():
                    break
                await asyncio.sleep(0)
            self.assertTrue(entered.is_set(), "Resume must open the existing inline picker.")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.current_state(), before)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


    def save_restorable_record(self, *, active=False):
        messages = [
            {"role": "user", "content": "Previous request"},
            {"type": "reasoning", "encrypted_content": "opaque-test-reasoning"},
            {"type": "function_call", "name": "read", "call_id": "saved-call",
             "arguments": '{"path":"example.py"}'},
            {"type": "function_call_output", "call_id": "saved-call", "output": "Saved output"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        transcript = [
            {"role": "user", "content": "Previous request"},
            {"role": "tool", "name": "read", "status": "finished",
             "content": Text("Read example.py", style="dim"), "expanded": True},
            {"role": "assistant", "content": "Previous answer", "streaming": False},
        ]
        if active:
            transcript.extend([
                {"role": "user", "content": "Pending edit"},
                {"role": "tool", "name": "edit", "detail": "example.py",
                 "status": "running", "content": Text("Editing"), "diff": "-old\n+new\n"},
                {"role": "assistant", "content": "Partial reply", "streaming": True},
            ])
        record = snapshot_session(
            new_session(self.project, "Saved conversation"), messages, transcript,
            active_turn={"prompt": "Pending edit", "tools_used": True} if active else None,
        )
        storage.save_session(self.project, record)
        self.picker.side_effect = None
        self.picker.return_value = record["id"]
        return record

    async def test_selected_conversation_restores_context_display_and_followup_save_baseline(self):
        record = self.save_restorable_record()
        path = storage.session_path(self.project, record["id"])
        original_bytes = path.read_bytes()
        self.owner.selected_provider = "chatgpt"
        self.owner.selected_model = "current-model"
        credentials = {"access_token": "fake-test-token"}
        self.owner.credentials = credentials
        self.owner.transcript_control.top_line = 12

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertEqual(self.owner.messages, record["messages"])
        self.assertEqual(encode_transcript(self.owner.transcript), record["transcript"])
        self.assertIsInstance(self.owner.transcript[1]["content"], Text)
        self.assertEqual(self.owner.session_record, record)
        self.assertEqual(self.owner.saved_session_record, record)
        self.assertIsNot(self.owner.session_record, self.owner.saved_session_record)
        self.assertIsNone(self.owner.active_turn)
        self.assertIsNone(self.owner.last_session_checkpoint)
        self.assertIsNone(self.owner.session_save_error)
        self.assertIsNone(self.owner.transcript_control.top_line)
        self.assertIsNone(self.owner.pending_reply)
        self.assertIs(self.owner.credentials, credentials)
        self.assertEqual(self.owner.selected_model, "current-model")
        self.assertEqual(self.owner.buffer.text, "Unsent draft")
        self.assertEqual(self.owner.buffer.cursor_position, 4)
        self.assertEqual(path.read_bytes(), original_bytes)

        self.owner.messages.append({"role": "user", "content": "Follow up"})
        self.owner.transcript.append({"role": "user", "content": "Follow up"})
        self.owner.session_record["name"] = "Changed live label"
        self.assertEqual(self.owner.saved_session_record, record)
        self.assertTrue(self.owner.checkpoint_session(force=True))
        saved = storage.load_session(self.project, record["id"])
        self.assertEqual(saved["name"], "Changed live label")
        self.assertEqual(saved["messages"][-1]["content"], "Follow up")
        self.assertEqual(saved["id"], record["id"])

    async def test_selected_partial_conversation_recovers_once_without_replaying_tools(self):
        record = self.save_restorable_record(active=True)
        path = storage.session_path(self.project, record["id"])
        original_bytes = path.read_bytes()

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertIsNone(self.owner.active_turn)
        self.assertEqual(len(self.owner.messages), 7, "Restore prior context and the pending request/caution.")
        self.assertEqual(self.owner.messages[-2], {"role": "user", "content": "Pending edit"})
        self.assertIn("Changes may already exist", self.owner.messages[-1]["content"])
        self.assertEqual(self.owner.transcript[-3]["status"], "interrupted")
        self.assertEqual(self.owner.transcript[-3]["diff"], "-old\n+new\n")
        self.assertFalse(self.owner.transcript[-2]["streaming"])
        self.assertEqual(self.owner.transcript[-1]["content"].plain, "Previous turn was interrupted.")
        self.assertEqual(self.owner.saved_session_record, record)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertTrue(self.owner.checkpoint_session(force=True))
        await self.resume()
        notices = [entry for entry in self.owner.transcript
                   if str(entry["content"]) == "Previous turn was interrupted."]
        self.assertEqual(len(notices), 1)


    async def test_exact_title_and_full_id_resume_directly_without_opening_picker(self):
        record = self.save_restorable_record()
        self.picker.side_effect = AssertionError("An exact title/ID must bypass the picker.")
        for query in (record["name"], record["id"], f"  {record['name']}  "):
            with self.subTest(query=query):
                await self.resume(query)
                self.assertEqual(self.owner.session_record["id"], record["id"])
                self.assertEqual(self.owner.messages, record["messages"])
                self.assertEqual(self.owner.saved_session_record, record)

    async def test_missing_title_and_partial_id_preserve_current_conversation(self):
        record = self.save_restorable_record()
        self.picker.side_effect = AssertionError("Unknown queries must not open the picker.")
        before = self.current_state()
        for query in ("Unknown title", "Saved", record["id"][:8]):
            with self.subTest(query=query):
                with self.assertRaisesRegex(RuntimeError, "No saved conversation matches"):
                    await self.resume(query)
                self.assertEqual(self.current_state(), before)

    async def test_duplicate_title_reports_ambiguity_but_full_id_selects_the_right_session(self):
        first = self.save_restorable_record()
        second = new_session(self.project, first["name"])
        storage.save_session(self.project, second)
        self.picker.side_effect = AssertionError("Direct lookup must not open the picker.")
        before = self.current_state()
        with self.assertRaisesRegex(RuntimeError, "Multiple.*Use /resume"):
            await self.resume(first["name"])
        self.assertEqual(self.current_state(), before)

        await self.resume(first["id"])

        self.assertEqual(self.owner.session_record["id"], first["id"])
        self.assertEqual(self.owner.messages, first["messages"])
