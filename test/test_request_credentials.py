"""Renew chat credentials without blocking the UI or racing rotating tokens."""

import asyncio
from datetime import datetime, timedelta, timezone
from threading import Event, get_ident
from collections.abc import Callable
from typing import Literal, cast
from prompt_toolkit import PromptSession
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from vin.auth import chatgpt as auth
from vin.tui import app as tui
from vin.types import Credentials, History, JSONValue, ModelMetadata
from vin.provider.events import ReplyEvents

type ReplyTask = asyncio.Task[Literal["error", "rate_limit"] | None]
type PendingTask = asyncio.Task[list[ModelMetadata] | Literal["error", "rate_limit"] | None]


class RequestCredentialsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.owner = tui.VinApp()
        self.owner.credentials = {
            "issuer": "https://auth.openai.com", "subject": "fake-user",
            "client_id": "fake-client", "ext_agent_host_id": "fake-host",
            "email": "fake@example.test", "id_token": "fake-id",
            "access_token": "old-secret", "refresh_token": "old-refresh-secret",
            "token_type": "Bearer", "expires_in": 3600,
            "scopes": ["offline_access", "chatgpt.tokens.use.direct"],
            "saved_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        }
        self.original = self.owner.credentials
        self.owner.selected_model = "selected-model"
        self.owner.app = Mock()
        self.draft = Document("next draft", cursor_position=3)
        self.owner.session = cast(PromptSession[str], Mock(spec=PromptSession, app=self.owner.app, default_buffer=Buffer(document=self.draft)))
        self.tokens: dict[str, JSONValue] = {"access_token": "new-secret", "refresh_token": "new-refresh-secret",
                       "token_type": "Bearer", "expires_in": 3600}
        self.exchange = self.enterContext(patch.object(auth, "exchange_refresh_token", return_value=self.tokens))
        self.save = self.enterContext(patch.object(auth, "save_credential_record"))
        self.seen: list[Credentials | None] = []
        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
            self.seen.append(credentials)
            events.put(("text", "Done"))
            events.put(("complete", None))
        self.worker = self.enterContext(patch.object(tui, "stream_chatgpt_response", side_effect=worker))

    def start(self, text: str = "Inspect") -> ReplyTask:
        task = asyncio.create_task(self.owner.stream_reply(text))
        self.owner.pending_reply = task
        return task

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def test_expired_credentials_refresh_off_ui_thread_before_chat_and_are_saved(self) -> None:
        ui_thread = get_ident()
        def exchange(credentials: Credentials) -> dict[str, JSONValue]:
            self.assertNotEqual(get_ident(), ui_thread)
            self.assertIs(credentials, self.original)
            return self.tokens
        self.exchange.side_effect = exchange
        await asyncio.wait_for(self.start(), 2)
        self.exchange.assert_called_once()
        self.save.assert_called_once_with(self.owner.credentials)
        self.assertEqual(self.owner.credentials.get("access_token") if self.owner.credentials else None, "new-secret")
        self.assertEqual(self.seen[0].get("access_token") if self.seen[0] else None, "new-secret")
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


    async def test_transient_refresh_failure_blocks_chat_without_secrets_or_login_advice(self) -> None:
        self.exchange.side_effect = RuntimeError("old-secret new-refresh-secret")
        await asyncio.wait_for(self.start(), 2)
        self.worker.assert_not_called()
        self.assertEqual(self.owner.messages, [])
        self.assertEqual(len(self.owner.transcript), 2)
        notice = str(self.owner.transcript[0]["content"])
        self.assertIn("try again", notice.lower())
        self.assertNotIn("old-secret", str(self.owner.transcript))
        self.assertNotIn("new-refresh-secret", str(self.owner.transcript))
        self.assertNotIn("/login", notice)
        self.assertRegex(str(self.owner.transcript[1]["content"]), r"^Failed after \d+s$")
        self.assertIs(self.owner.credentials, self.original)
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)

    async def test_terminal_refresh_failure_advises_login_and_preserves_credentials(self) -> None:
        self.exchange.side_effect = auth.TokenExchangeError(400, "invalid_grant")
        await asyncio.wait_for(self.start(), 2)
        self.worker.assert_not_called()
        self.save.assert_not_called()
        self.assertIs(self.owner.credentials, self.original)
        self.assertEqual(len(self.owner.transcript), 2)
        self.assertIn("/login", str(self.owner.transcript[0]["content"]))
        self.assertRegex(str(self.owner.transcript[1]["content"]), r"^Failed after \d+s$")

    async def test_model_discovery_and_chat_share_the_same_refresh(self) -> None:
        entered, release = Event(), Event()
        def exchange(credentials: Credentials) -> dict[str, JSONValue]:
            entered.set()
            release.wait(2)
            return self.tokens
        self.exchange.side_effect = exchange
        tasks: list[PendingTask] = []
        with (
            patch.object(tui, "list_chatgpt_models", return_value=[]) as catalog,
            patch.object(tui, "save_cached_chatgpt_models") as cache,
        ):
            try:
                discovery = asyncio.create_task(self.owner.discover_chatgpt_models())
                tasks.append(discovery)
                await self.wait_until(entered.is_set)
                turn = self.start()
                tasks.append(turn)
                await asyncio.sleep(.02)
                self.exchange.assert_called_once()
                self.worker.assert_not_called()
                catalog.assert_not_called()
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
                self.exchange.assert_called_once()
                catalog.assert_called_once_with(self.owner.credentials)
                cache.assert_called_once_with(self.owner.credentials, [])
                self.assertEqual(self.seen[0].get("access_token") if self.seen[0] else None, "new-secret")
            finally:
                release.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_cancel_then_next_prompt_share_inflight_refresh(self) -> None:
        entered, release = Event(), Event()
        def exchange(credentials: Credentials) -> dict[str, JSONValue]:
            entered.set()
            release.wait(2)
            return self.tokens
        self.exchange.side_effect = exchange
        tasks: list[PendingTask] = []
        try:
            first = self.start("Cancelled")
            tasks.append(first)
            await self.wait_until(entered.is_set)
            self.owner.pending_reply = None
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(first, .5)
            second = self.start("Next")
            tasks.append(second)
            await asyncio.sleep(.02)
            self.exchange.assert_called_once()
            self.worker.assert_not_called()
            release.set()
            await asyncio.wait_for(second, 2)
            self.assertEqual(self.seen[0].get("access_token") if self.seen[0] else None, "new-secret")
            self.assertEqual([m["content"] for m in self.owner.messages], ["Next", "Done"])
            self.assertEqual(self.owner.session.default_buffer.document, self.draft)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_logout_waits_for_cancelled_turns_refresh_and_revokes_replacement(self) -> None:
        entered, release = Event(), Event()
        def exchange(credentials: Credentials) -> dict[str, JSONValue]:
            entered.set()
            release.wait(2)
            return self.tokens
        self.exchange.side_effect = exchange
        revoked: list[str | None] = []
        def logout(credentials: Credentials, **kwargs: object) -> Credentials:
            revoked.append(credentials.get("refresh_token"))
            result = credentials.copy()
            result.pop("access_token", None)
            result.pop("refresh_token", None)
            result.pop("id_token", None)
            return result
        tasks: list[PendingTask] = []
        with patch.object(tui, "logout_chatgpt", side_effect=logout) as sign_out:
            try:
                turn = self.start()
                tasks.append(turn)
                await self.wait_until(entered.is_set)
                turn.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await turn
                leaving = asyncio.create_task(self.owner.sign_out())
                tasks.append(leaving)
                await asyncio.sleep(.02)
                sign_out.assert_not_called()
                release.set()
                await asyncio.wait_for(leaving, 2)
                self.assertEqual(revoked, ["new-refresh-secret"])
                assert self.owner.credentials is not None
                self.assertNotIn("access_token", self.owner.credentials)
                self.worker.assert_not_called()
            finally:
                release.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_late_refresh_does_not_replace_a_different_active_session(self) -> None:
        entered, release = Event(), Event()
        def exchange(credentials: Credentials) -> dict[str, JSONValue]:
            entered.set()
            release.wait(2)
            return self.tokens
        self.exchange.side_effect = exchange
        task = self.start()
        try:
            await self.wait_until(entered.is_set)
            replacement: Credentials = {**self.original, "subject": "different-user", "access_token": "different-secret"}
            self.owner.credentials = replacement
            release.set()
            await asyncio.wait_for(task, 2)
            self.assertIs(self.owner.credentials, replacement)
            self.worker.assert_not_called()
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
