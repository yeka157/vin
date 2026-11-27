"""/resume, /rename and session completion in the composer."""

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.text import Text

from session_typing_helpers import present, string_value, text_value
from vin.types.events import TokenUsage
from vin.sessions import storage
from vin.sessions.records import new_session, snapshot_session
from vin.sessions.transcript import encode_transcript
from vin.tui import app, conversations
from vin.tui.app import VinApp
from vin.tui.completion import SessionCompleter
from vin.tui.layout import build_workspace
from vin.types.auth import Credentials
from vin.types.data import Pathish
from vin.types.history import History
from vin.types.sessions import ActiveTurn, SessionRecord
from vin.types.transcript import Subagent, TranscriptEntry


class SessionResumeCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
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
        self.owner.last_usage = {
            "input_tokens": 1200, "output_tokens": 100, "total_tokens": 1300,
        }
        self.owner.transcript = [{"role": "assistant", "content": "Keep display"}]
        self.owner.last_session_checkpoint = 10.0
        self.owner.session_save_error = "Previous save warning"
        self.buffer = Buffer()
        self.buffer.text = "Unsent draft"
        self.buffer.cursor_position = 4
        self.picker = AsyncMock(side_effect=AssertionError("No picker should open here"))
        picker = patch.object(self.owner, "choose_inline", self.picker)
        picker.start()
        self.addCleanup(picker.stop)

    def current_state(self) -> tuple[SessionRecord | None, SessionRecord | None, History, list[TranscriptEntry], ActiveTurn | None, float | None, str | None, str, int, TokenUsage | None]:
        return deepcopy((
            self.owner.session_record, self.owner.saved_session_record,
            self.owner.messages, self.owner.transcript, self.owner.active_turn,
            self.owner.last_session_checkpoint, self.owner.session_save_error,
            self.buffer.text, self.buffer.cursor_position,
            self.owner.last_usage,
        ))

    async def resume(self, query: str = "") -> str | None:
        return await self.owner.resume_conversation(query)

    async def test_picker_distinguishes_duplicate_names_and_cancel_keeps_current_conversation(self) -> None:
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
        shown: list[tuple[str, str]] = []

        async def cancel_selection(message: str, options: Sequence[tuple[str, str]], default: str | None = None) -> str:
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

    async def test_app_task_cancellation_during_picker_is_not_swallowed(self) -> None:
        storage.save_session(self.project, new_session(self.project, "Saved conversation"))
        before = self.current_state()
        entered = asyncio.Event()

        async def wait_for_selection(message: str, options: Sequence[tuple[str, str]], default: str | None = None) -> str:
            entered.set()
            return await asyncio.Future[str]()

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


    def save_restorable_record(self, *, active: bool = False) -> SessionRecord:
        messages: History = [
            {"role": "user", "content": "Previous request"},
            {"type": "reasoning", "encrypted_content": "opaque-test-reasoning"},
            {"type": "function_call", "name": "read", "call_id": "saved-call",
             "arguments": '{"path":"example.py"}'},
            {"type": "function_call_output", "call_id": "saved-call", "output": "Saved output"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        transcript: list[TranscriptEntry] = [
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

    async def test_selected_conversation_restores_context_display_and_followup_save_baseline(self) -> None:
        record = self.save_restorable_record()
        path = storage.session_path(self.project, record["id"])
        original_bytes = path.read_bytes()
        self.owner.selected_provider = "chatgpt"
        self.owner.selected_model = "current-model"
        credentials: Credentials = {"issuer": "test", "subject": "user", "client_id": "test", "ext_agent_host_id": "test", "access_token": "fake-test-token"}
        self.owner.credentials = credentials
        self.owner.transcript_control.top_line = 12

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertEqual(self.owner.messages, record["messages"])
        self.assertIsNone(self.owner.last_usage)
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
        self.assertEqual(self.buffer.text, "Unsent draft")
        self.assertEqual(self.buffer.cursor_position, 4)
        self.assertEqual(path.read_bytes(), original_bytes)

        self.owner.messages.append({"role": "user", "content": "Follow up"})
        self.owner.transcript.append({"role": "user", "content": "Follow up"})
        present(self.owner.session_record)["name"] = "Changed live label"
        self.assertEqual(self.owner.saved_session_record, record)
        self.assertTrue(self.owner.checkpoint_session(force=True))
        saved = present(storage.load_session(self.project, record["id"]))
        self.assertEqual(saved["name"], "Changed live label")
        self.assertEqual(saved["messages"][-1]["content"], "Follow up")
        self.assertEqual(saved["id"], record["id"])

    async def test_selected_partial_conversation_recovers_once_without_replaying_tools(self) -> None:
        record = self.save_restorable_record(active=True)
        path = storage.session_path(self.project, record["id"])
        original_bytes = path.read_bytes()

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertIsNone(self.owner.active_turn)
        self.assertEqual(len(self.owner.messages), 7, "Restore prior context and the pending request/caution.")
        self.assertEqual(self.owner.messages[-2], {"role": "user", "content": "Pending edit"})
        self.assertIn("Changes may already exist", string_value(self.owner.messages[-1]["content"]))
        self.assertEqual(self.owner.transcript[-3].get("status"), "interrupted")
        self.assertEqual(self.owner.transcript[-3].get("diff"), "-old\n+new\n")
        assert "streaming" in self.owner.transcript[-2]
        self.assertFalse(self.owner.transcript[-2]["streaming"])
        self.assertEqual(text_value(self.owner.transcript[-1]["content"]).plain, "Previous turn was interrupted.")
        self.assertEqual(self.owner.saved_session_record, record)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertTrue(self.owner.checkpoint_session(force=True))
        await self.resume()
        notices = [entry for entry in self.owner.transcript
                   if str(entry["content"]) == "Previous turn was interrupted."]
        self.assertEqual(len(notices), 1)


    async def test_resume_restores_saved_subagents_and_later_checkpoints_keep_them(self) -> None:
        running: Subagent = {
            "id": "call_running", "description": "Run tests", "status": "running", "started_at": 50.0,
            "duration": None, "tokens": 40, "transcript": [
                {"role": "user", "content": "Run the tests."},
                {"role": "assistant", "content": "Starting", "streaming": False},
                {"role": "tool", "name": "bash", "label": "bash make test", "status": "running",
                 "content": Text("● bash make test · running", style="dim")},
            ]}
        finished: Subagent = {
            "id": "call_done", "description": "Find sessions", "status": "finished", "started_at": 10.0,
            "duration": 4.5, "tokens": 100, "transcript": [{"role": "user", "content": "Find sessions."}]}
        record = snapshot_session(new_session(self.project, "Delegated work"), [],
                                  [{"role": "user", "content": "Delegate"}], active_turn=None,
                                  subagents=[running, finished])
        storage.save_session(self.project, record)
        self.picker.side_effect = None
        self.picker.return_value = record["id"]
        self.owner.viewed, self.owner.panel_index = "stale", 1

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertEqual((self.owner.panel_start, self.owner.viewed, self.owner.panel_index), (2, None, None))
        self.assertEqual(self.owner.panel_agents(), [], "Restored subagents stay out of the panel.")
        restored_running, restored_finished = self.owner.subagents
        self.assertEqual([(item["id"], item["status"], item["duration"], item["tokens"])
                          for item in self.owner.subagents],
                         [("call_running", "interrupted", None, 40), ("call_done", "finished", 4.5, 100)])
        tool = restored_running["transcript"][2]
        self.assertEqual(tool.get("status"), "interrupted")
        self.assertEqual(text_value(tool["content"]).plain, "● bash make test · interrupted")
        self.assertEqual(restored_finished["transcript"], [{"role": "user", "content": "Find sessions."}])

        self.assertTrue(self.owner.checkpoint_session(force=True))
        saved = present(storage.load_session(self.project, record["id"]))
        self.assertEqual([(item["id"], item["status"]) for item in saved.get("subagents", [])],
                         [("call_running", "interrupted"), ("call_done", "finished")])

    async def test_resuming_an_older_session_clears_subagents_from_the_previous_one(self) -> None:
        record = self.save_restorable_record()
        self.owner.subagents = [{"id": "stale", "description": "x", "status": "finished", "started_at": 0.0,
                                 "duration": 1.0, "tokens": 0, "transcript": []}]

        with patch.object(app, "stream_chatgpt_response", side_effect=AssertionError("Resume must not run a provider")):
            await self.resume()

        self.assertEqual(self.owner.subagents, [])
        self.assertNotIn("subagents", present(storage.load_session(self.project, record["id"])))

    async def test_exact_title_and_full_id_resume_directly_without_opening_picker(self) -> None:
        record = self.save_restorable_record()
        self.picker.side_effect = AssertionError("An exact title/ID must bypass the picker.")
        for query in (record["name"], record["id"], f"  {record['name']}  "):
            with self.subTest(query=query):
                await self.resume(query)
                self.assertEqual(present(self.owner.session_record)["id"], record["id"])
                self.assertEqual(self.owner.messages, record["messages"])
                self.assertEqual(self.owner.saved_session_record, record)

    async def test_missing_title_and_partial_id_preserve_current_conversation(self) -> None:
        record = self.save_restorable_record()
        self.picker.side_effect = AssertionError("Unknown queries must not open the picker.")
        before = self.current_state()
        for query in ("Unknown title", "Saved", record["id"][:8]):
            with self.subTest(query=query):
                with self.assertRaisesRegex(RuntimeError, "No saved conversation matches"):
                    await self.resume(query)
                self.assertEqual(self.current_state(), before)

    async def test_unavailable_or_unrestorable_session_preserves_current_usage(self) -> None:
        record = self.save_restorable_record()
        before = self.current_state()

        with patch.object(conversations, "load_session", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no longer available"):
                await self.resume(record["id"])
        self.assertEqual(self.current_state(), before)

        with patch.object(conversations, "restore_session", side_effect=ValueError("Invalid saved context")):
            with self.assertRaisesRegex(ValueError, "Invalid saved context"):
                await self.resume(record["id"])
        self.assertEqual(self.current_state(), before)

    async def test_duplicate_title_reports_ambiguity_but_full_id_selects_the_right_session(self) -> None:
        first = self.save_restorable_record()
        second = new_session(self.project, first["name"])
        storage.save_session(self.project, second)
        self.picker.side_effect = AssertionError("Direct lookup must not open the picker.")
        before = self.current_state()
        with self.assertRaisesRegex(RuntimeError, "Multiple.*Use /resume"):
            await self.resume(first["name"])
        self.assertEqual(self.current_state(), before)

        await self.resume(first["id"])

        self.assertEqual(present(self.owner.session_record)["id"], first["id"])
        self.assertEqual(self.owner.messages, first["messages"])


class SessionCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", self.root / "sessions"))
        self.commands = WordCompleter(["/help", "/model", "/resume"], sentence=True)

    def completer(self) -> SessionCompleter:
        return SessionCompleter(self.project, self.commands)

    def suggestions(self, completer: Completer, text: str) -> list[Completion]:
        return list(completer.get_completions(Document(text), CompleteEvent(text_inserted=True)))

    def apply(self, text: str, completion: Completion) -> str:
        buffer = Buffer(document=Document(text))
        buffer.apply_completion(completion)
        return buffer.text

    def save(self, project: Pathish | None = None, *, updated: str = "2026-10-25T10:00:00+00:00") -> SessionRecord:
        project = self.project if project is None else project
        record = new_session(project, "Fix cancellation behavior")
        record.update(created_at="2026-10-24T10:00:00+00:00", updated_at=updated)
        storage.save_session(project, record)
        return record

    def test_title_queries_show_distinct_current_project_sessions_and_insert_full_ids(self) -> None:
        older = self.save()
        newer = self.save(updated="2026-10-25T12:00:00+00:00")
        other_project = self.root / "another-project"
        other_project.mkdir()
        self.save(other_project)
        completer = self.completer()

        for text, prefix in (("/resume ", "/resume "),
                             ("/resume CANCEL", "/resume "),
                             ("/resume Fix cancellation", "/resume "),
                             ("/resume\tCANCEL", "/resume\t")):
            with self.subTest(text=text):
                offered = self.suggestions(completer, text)
                self.assertEqual([item.text for item in offered], [newer["id"], older["id"]])
                for item, record in zip(offered, (newer, older)):
                    self.assertEqual(fragment_list_to_text(item.display), "Fix cancellation behavior")
                    meta = fragment_list_to_text(item.display_meta)
                    self.assertIn(record["updated_at"], meta)
                    self.assertIn(record["id"][:8], meta)
                    self.assertEqual(self.apply(text, item), prefix + record["id"])

    def test_id_query_and_newly_saved_sessions_are_available_without_recreating_completer(self) -> None:
        first = self.save()
        completer = self.completer()
        offered = self.suggestions(completer, f"/resume {first['id'][:8]}")
        self.assertEqual([item.text for item in offered], [first["id"]])
        self.assertEqual(self.suggestions(completer, "/resume No matching session"), [])

        second = self.save(updated="2026-10-25T12:00:00+00:00")
        offered = self.suggestions(completer, "/resume ")
        self.assertEqual([item.text for item in offered], [second["id"], first["id"]])


    def workspace(self) -> VinApp:
        owner = VinApp()
        owner.project_dir = self.project
        with create_app_session(input=DummyInput(), output=DummyOutput()):
            build_workspace(owner)
        return owner

    def test_workspace_composer_offers_current_project_session_ids_without_loading_them(self) -> None:
        record = self.save()
        path = storage.session_path(self.project, record["id"])
        saved_bytes = path.read_bytes()
        owner = self.workspace()
        document = owner.session.default_buffer.document

        assert owner.session.completer is not None
        offered = self.suggestions(owner.session.completer, "/resume cancellation")

        self.assertEqual([item.text for item in offered], [record["id"]])
        self.assertEqual(fragment_list_to_text(offered[0].display), record["name"])
        self.assertEqual(owner.session.default_buffer.document, document)
        self.assertEqual(owner.messages, [])
        self.assertIsNone(owner.session_record)
        self.assertIsNone(owner.pending_reply)
        self.assertEqual(path.read_bytes(), saved_bytes)


class SessionCompletionKeyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", Path(temporary.name) / "sessions"))
        self.records: list[SessionRecord] = []
        for updated in ("2026-10-25T12:00:00+00:00", "2026-10-25T10:00:00+00:00"):
            record = new_session(self.project, "Review cancellation behavior")
            record.update(created_at="2026-10-24T10:00:00+00:00", updated_at=updated)
            storage.save_session(self.project, record)
            self.records.append(record)
        self.enterContext(create_app_session(input=DummyInput(), output=DummyOutput()))
        self.owner = VinApp()
        self.owner.project_dir = self.project
        build_workspace(self.owner)
        self.enterContext(set_app(self.owner.app))
        self.buffer = self.owner.session.default_buffer
        self.buffer.accept_handler = self.owner.accept_input

    async def asyncTearDown(self) -> None:
        await self.owner.app.cancel_and_wait_for_background_tasks()

    async def offer(self, text: str, count: int) -> None:
        self.buffer.document = Document(text)
        self.buffer.start_completion(select_first=False)
        async with asyncio.timeout(2):
            while self.buffer.complete_state is None or len(self.buffer.complete_state.completions) != count:
                await asyncio.sleep(0)

    def press(self, key: Keys) -> None:
        self.owner.app.key_processor.feed(KeyPress(key))
        self.owner.app.key_processor.process_keys()

    async def test_tab_inserts_first_uuid_and_enter_alone_submits_the_command(self) -> None:
        await self.offer("/resume cancellation", 2)
        assert self.buffer.complete_state is not None
        self.assertIsNone(self.buffer.complete_state.current_completion)

        self.press(Keys.ControlI)

        command = f"/resume {self.records[0]['id']}"
        self.assertEqual(self.buffer.text, command)
        self.assertIsNone(self.buffer.complete_state)
        self.assertTrue(self.owner.submitted.empty())
        self.assertFalse(self.owner.queued_prompts)
        self.assertIsNone(self.owner.session_record)
        self.assertIsNone(self.owner.pending_reply)
        self.assertEqual(self.owner.messages, [])
        self.assertEqual(self.buffer.cursor_position, len(command))

        refreshed = asyncio.Event()
        def signal_refreshed(sender: Buffer) -> None:
            refreshed.set()

        self.buffer.on_completions_changed += signal_refreshed
        self.buffer.start_completion(select_first=False)
        await asyncio.wait_for(refreshed.wait(), timeout=2)
        self.assertIsNone(self.buffer.complete_state, "An exact UUID must not reopen the menu.")

        self.press(Keys.ControlM)

        self.assertEqual(self.owner.submitted.get_nowait(), command)
        self.assertTrue(self.owner.submitted.empty())
        self.assertFalse(self.owner.queued_prompts)

    async def test_tab_accepts_highlighted_uuid_and_closes_the_menu(self) -> None:
        await self.offer("/resume cancellation", 2)
        self.press(Keys.Down)
        self.press(Keys.Down)
        assert self.buffer.complete_state is not None
        assert self.buffer.complete_state.current_completion is not None
        self.assertEqual(self.buffer.complete_state.current_completion.text, self.records[1]["id"])

        self.press(Keys.ControlI)

        self.assertEqual(self.buffer.text, f"/resume {self.records[1]['id']}")
        self.assertIsNone(self.buffer.complete_state)
        self.assertTrue(self.owner.submitted.empty())
        self.assertIsNone(self.owner.session_record)

    async def test_regular_command_tab_completion_still_works(self) -> None:
        await self.offer("/mo", 1)

        self.press(Keys.ControlI)

        self.assertEqual(self.buffer.text, "/model")
        self.assertTrue(self.owner.submitted.empty())

    async def test_busy_compact_is_rejected_with_a_notice_and_leaves_the_queue(self) -> None:
        async def hold() -> None:
            await asyncio.Event().wait()

        for state in ("replying", "queued"):
            with self.subTest(state=state):
                self.owner.transcript.clear()
                self.owner.queued_prompts.clear()
                self.owner.queued_prompts.append("Next request")
                task = asyncio.create_task(hold()) if state == "replying" else None
                self.owner.pending_reply = task
                self.buffer.document = Document("/compact")
                try:
                    self.press(Keys.ControlM)
                    self.press(Keys.ControlM)
                finally:
                    self.owner.pending_reply = None
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

                self.assertTrue(self.owner.submitted.empty())
                self.assertEqual(list(self.owner.queued_prompts), ["Next request"])
                self.assertEqual(self.buffer.text, "/compact")
                self.assertEqual(
                    [row["content"] for row in self.owner.transcript],
                    ["Finish or cancel the current work before compacting."],
                )

    async def test_idle_compact_is_submitted_without_a_notice(self) -> None:
        self.buffer.document = Document("/compact")

        self.press(Keys.ControlM)

        self.assertEqual(self.owner.submitted.get_nowait(), "/compact")
        self.assertEqual(self.owner.transcript, [])
        self.assertEqual(self.buffer.text, "")


if __name__ == "__main__":
    unittest.main()
