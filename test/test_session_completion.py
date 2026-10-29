"""Session argument suggestions using temporary files and real completion objects."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.completion import CompleteEvent, WordCompleter, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from vin.sessions import storage
from vin.sessions.records import new_session
from vin.tui.app import VinApp
from vin.tui.layout import build_workspace
from vin.tui.completion import SessionCompleter
from vin.types import Pathish, SessionRecord


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
