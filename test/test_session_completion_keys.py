"""Real composer completion and keyboard handling without a provider or receiver."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.document import Document
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from vin.sessions import storage
from vin.sessions.records import new_session
from vin.tui.app import VinApp
from vin.tui.layout import build_workspace
from vin.types import SessionRecord


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
