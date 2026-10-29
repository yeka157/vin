"""Rename validation and persistence using temporary conversation storage."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer

from vin.sessions import storage
from vin.sessions.records import new_session, snapshot_session
from vin.tui import conversations
from vin.tui.app import VinApp
from vin.types import SessionRecord
from session_typing_helpers import present


class SessionRenameTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", self.root / "sessions"))
        self.owner = VinApp()
        self.owner.project_dir = self.project
        self.buffer = Buffer()
        self.buffer.text = "Unsent draft"

    def rename(self, name: object) -> str | None:
        return self.owner.rename_conversation(name)

    def test_invalid_name_preserves_current_conversation_and_saved_bytes(self) -> None:
        self.owner.messages = [{"role": "user", "content": "Original prompt"}]
        self.owner.transcript = [{"role": "user", "content": "Original prompt"}]
        record = snapshot_session(
            new_session(self.project, "Original prompt"), self.owner.messages,
            self.owner.transcript, active_turn=None,
        )
        path = storage.save_session(self.project, record)
        self.owner.session_record = deepcopy(record)
        self.owner.saved_session_record = deepcopy(record)
        previous = deepcopy((self.owner.session_record, self.owner.saved_session_record,
                             self.owner.messages, self.owner.transcript))
        previous_bytes = path.read_bytes()

        for name in (None, 42, "", " \t\n ", "x" * 81, "界" * 81):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.rename(name)
                self.assertEqual((self.owner.session_record, self.owner.saved_session_record,
                                  self.owner.messages, self.owner.transcript), previous)
                self.assertEqual(self.buffer.text, "Unsent draft")
                self.assertEqual(path.read_bytes(), previous_bytes)

    def prepare_saved_conversation(self) -> SessionRecord:
        self.owner.messages = [
            {"role": "user", "content": "Original prompt"},
            {"type": "reasoning", "encrypted_content": "fake-saved-reasoning"},
            {"role": "assistant", "content": "Original answer"},
        ]
        self.owner.transcript = [
            {"role": "user", "content": "Original prompt"},
            {"role": "assistant", "content": "Original answer"},
        ]
        record = snapshot_session(
            new_session(self.project, "Original prompt"), self.owner.messages,
            self.owner.transcript, active_turn=None,
        )
        storage.save_session(self.project, record)
        self.owner.session_record = deepcopy(record)
        self.owner.saved_session_record = deepcopy(record)
        self.owner.last_session_checkpoint = 10.0
        return record

    def test_rename_saves_normalized_name_immediately_without_changing_identity_or_history(self) -> None:
        record = self.prepare_saved_conversation()
        previous_messages = deepcopy(self.owner.messages)
        previous_transcript = deepcopy(self.owner.transcript)

        with patch.object(conversations, "monotonic", return_value=10.0):
            outcome = self.rename("  Fix\t cancellation\n behavior  ")

        self.assertEqual(outcome, "Renamed: Fix cancellation behavior")
        saved = present(storage.load_session(self.project, record["id"]))
        self.assertEqual(saved["name"], "Fix cancellation behavior")
        self.assertEqual(saved["id"], record["id"])
        self.assertEqual(saved["created_at"], record["created_at"])
        self.assertEqual(saved["project_dir"], record["project_dir"])
        self.assertEqual(saved["messages"], previous_messages)
        self.assertEqual(saved["transcript"], record["transcript"])
        self.assertEqual(self.owner.messages, previous_messages)
        self.assertEqual(self.owner.transcript, previous_transcript)
        self.assertEqual(self.buffer.text, "Unsent draft")
        self.assertEqual(self.owner.session_record, saved)
        self.assertEqual(self.owner.saved_session_record, saved)
        self.assertIsNot(self.owner.session_record, self.owner.saved_session_record)
        self.assertEqual([item["id"] for item in storage.list_sessions(self.project)], [record["id"]])
        self.assertEqual(self.rename("界" * 80), f"Renamed: {'界' * 80}")
        self.assertEqual(present(storage.load_session(self.project, record["id"]))["name"], "界" * 80)

    def test_failed_rename_keeps_previous_name_and_saved_bytes_then_can_retry(self) -> None:
        record = self.prepare_saved_conversation()
        path = storage.session_path(self.project, record["id"])
        previous_bytes = path.read_bytes()
        previous_transcript = deepcopy(self.owner.transcript)

        with patch.object(conversations, "monotonic", return_value=10.0):
            with patch.object(conversations, "save_session", side_effect=RuntimeError("Disk unavailable")):
                self.assertIsNone(self.rename("New title"))

            self.assertEqual(self.owner.session_record, record)
            self.assertEqual(self.owner.saved_session_record, record)
            self.assertEqual(path.read_bytes(), previous_bytes)
            self.assertEqual(self.owner.transcript[:-1], previous_transcript)
            notice = self.owner.transcript[-1]
            self.assertEqual(notice["role"], "notice")
            self.assertIn("Disk unavailable", str(notice["content"]))
            self.assertEqual(self.buffer.text, "Unsent draft")
            self.assertEqual(self.rename("New title"), "Renamed: New title")

        self.assertEqual(present(storage.load_session(self.project, record["id"]))["name"], "New title")
        self.assertIsNone(self.owner.session_save_error)
