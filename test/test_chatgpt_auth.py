"""ChatGPT sign-in, saved credentials, token refresh and request credentials."""

import asyncio
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.message import Message
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
from typing import Literal, cast
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from vin.auth import chatgpt, chatgpt as auth
from vin.auth.credentials import load_credential_record, save_private_json
from vin.provider.events import ReplyEvents
from vin.tui import app as tui
from vin.tui.app import VinApp
from vin.types.auth import Credentials, Identity, Registration
from vin.types.data import JSONValue
from vin.types.history import History
from vin.types.models import ModelMetadata


class TokenExchangeErrorTests(unittest.TestCase):
    def test_terminal_refresh_codes_advise_login_without_raw_error_details(self) -> None:
        for code in chatgpt.TokenExchangeError.LOGIN_CODES:
            for detail in (f'"{code}"', f'{{"code":"{code}"}}'):
                with self.subTest(code=code, detail=detail):
                    body = BytesIO(
                        ('{"error":' + detail + ',"error_description":"secret-token"}').encode()
                    )
                    error = HTTPError("https://example.test", 400, "secret-token", Message(), body)
                    with patch.object(chatgpt, "urlopen", side_effect=error):
                        with self.assertRaises(chatgpt.TokenExchangeError) as caught:
                            chatgpt.request_tokens({"refresh_token": "secret-token"})
                    self.assertTrue(caught.exception.requires_login)
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.status_code, 400)
                    self.assertNotIn("secret-token", str(caught.exception))
                    self.assertTrue(body.closed)

    def test_unrecognized_and_unreadable_errors_do_not_advise_login_or_leak(self) -> None:
        for payload in (
            b'{"error":"server_error","message":"secret-token"}',
            b'{"error":{"code":"secret-token"}}',
            b'{"error":["secret-token"]}',
            b'secret-token', b'\xff', b'[]', b'{}',
        ):
            with self.subTest(payload=payload):
                body = BytesIO(payload)
                error = HTTPError("https://example.test", 503, "secret-token", Message(), body)
                with patch.object(chatgpt, "urlopen", side_effect=error):
                    with self.assertRaises(chatgpt.TokenExchangeError) as caught:
                        chatgpt.request_tokens({})
                self.assertFalse(caught.exception.requires_login)
                self.assertEqual(str(caught.exception), "Token exchange failed (HTTP 503)")
                self.assertTrue(body.closed)


class RefreshCredentialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials: Credentials = {
            "issuer": "https://auth.openai.com",
            "subject": "test-user",
            "email": "user@example.test",
            "client_id": "oaiapp_test",
            "ext_agent_host_id": "test-host",
            "id_token": "fake-id-token",
            "access_token": "fake-old-access",
            "refresh_token": "fake-old-refresh",
            "token_type": "Bearer",
            "scopes": ["offline_access", "chatgpt.tokens.use.direct"],
            "expires_in": 3600,
            "saved_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        }

    def test_unexpired_credentials_are_reused(self) -> None:
        self.credentials["saved_at"] = datetime.now(timezone.utc).isoformat()

        with (
            patch.object(chatgpt, "exchange_refresh_token") as exchange,
            patch.object(chatgpt, "save_credential_record") as save,
        ):
            result = chatgpt.refresh_credentials(self.credentials)

        self.assertIs(result, self.credentials)
        exchange.assert_not_called()
        save.assert_not_called()

    def test_expired_credentials_are_refreshed_and_saved(self) -> None:
        original = self.credentials.copy()
        tokens = {
            "access_token": "fake-new-access",
            "refresh_token": "fake-new-refresh",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with (
            patch.object(
                chatgpt,
                "exchange_refresh_token",
                return_value=tokens,
            ) as exchange,
            patch.object(chatgpt, "save_credential_record") as save,
        ):
            result = chatgpt.refresh_credentials(self.credentials)
        exchange.assert_called_once_with(self.credentials)
        save.assert_called_once_with(result)

        self.assertEqual(result.get("access_token"), "fake-new-access")
        self.assertEqual(result.get("refresh_token"), "fake-new-refresh")

        for field in (
            "issuer",
            "subject",
            "email",
            "client_id",
            "ext_agent_host_id",
            "id_token",
        ):
            self.assertEqual(result.get(field), original.get(field))

        self.assertCountEqual(result.get("scopes", []), original.get("scopes", []))
        self.assertFalse(chatgpt.is_access_token_expired(result))
        self.assertEqual(self.credentials, original)


    def test_invalid_refresh_does_not_save_or_change_credentials(self) -> None:
        original = self.credentials.copy()
        tokens = {
            "access_token": "fake-new-access",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with (
            patch.object(
                chatgpt,
                "exchange_refresh_token",
                return_value=tokens,
            ),
            patch.object(chatgpt, "save_credential_record") as save,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "replacement refresh token",
            ):
                chatgpt.refresh_credentials(self.credentials)
        save.assert_not_called()
        self.assertEqual(self.credentials, original)


class LogoutCredentialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials: Credentials = {
            "issuer": "https://auth.openai.com",
            "subject": "test-user",
            "email": "user@example.test",
            "client_id": "oaiapp_test",
            "ext_agent_host_id": "test-host",
            "id_token": "fake-id-token",
            "access_token": "fake-access-token",
            "refresh_token": "fake-refresh-token",
            "token_type": "Bearer",
            "scopes": ["offline_access", "chatgpt.tokens.use.direct"],
            "expires_in": 3600,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.registration = {
            "issuer": "https://auth.openai.com",
            "subject": "test-user",
            "email": "user@example.test",
            "client_id": "oaiapp_test",
            "ext_agent_host_id": "test-host",
        }

    def test_revocation_warning_uses_callback_without_printing_and_tokens_are_removed(self) -> None:
        notices: list[str] = []
        output = StringIO()
        with (
            patch.object(chatgpt, "revoke_refresh_token", return_value=False),
            patch.object(chatgpt, "save_credential_record") as save,
            redirect_stdout(output),
        ):
            result = chatgpt.logout_chatgpt(self.credentials, notify=notices.append)

        self.assertEqual(result, self.registration)
        self.assertEqual(self.credentials, self.registration)
        save.assert_called_once_with(self.registration)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(notices), 1)
        self.assertIn("Signed out locally", notices[0])
        self.assertIn("Remote revocation was not confirmed", notices[0])


class CredentialStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "account.json"

    def test_saved_credentials_are_private(self) -> None:
        data = {"access_token": "fake-token"}

        save_private_json(self.destination, data)

        self.assertEqual(
            json.loads(self.destination.read_text(encoding="utf-8")),
            data,
        )

        self.assertEqual(
            self.destination.stat().st_mode & 0o777,
            0o600,
        )

    def test_registration_preserves_unvalidated_non_token_metadata(self) -> None:
        registration: Registration = {
            "issuer": "https://auth.openai.com",
            "subject": "test-account",
            "client_id": "issued-client",
            "ext_agent_host_id": "test-host",
            "scopes": 123,
            "saved_at": [],
            "expires_in": "later",
            "token_type": False,
        }
        save_private_json(self.destination, registration)
        saved = self.destination.read_bytes()

        loaded = load_credential_record(self.destination)

        self.assertEqual(loaded, registration)
        self.assertEqual(self.destination.read_bytes(), saved)
        assert loaded is not None
        self.assertNotIn("access_token", loaded)
        self.assertNotIn("refresh_token", loaded)
        self.assertNotIn("id_token", loaded)
        self.assertEqual(loaded.get("scopes"), 123)
        self.assertEqual(loaded.get("saved_at"), [])
        self.assertEqual(loaded.get("expires_in"), "later")
        self.assertIs(loaded.get("token_type"), False)

        app = VinApp()
        app.credentials = loaded
        with patch("vin.tui.app.refresh_credentials") as refresh:
            self.assertIs(asyncio.run(app.prepare_request_credentials()), loaded)
        refresh.assert_not_called()
        self.assertIs(app.credentials, loaded)
        self.assertEqual(app.credentials, registration)

    def test_failed_save_preserves_existing_credentials(self) -> None:
        save_private_json(
            self.destination,
            {"access_token": "fake-old-token"},
        )

        original = self.destination.read_bytes()

        with patch(
            "vin.auth.credentials.os.replace",
            side_effect=OSError("Simulated write failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not save"):
                save_private_json(
                    self.destination,
                    {"access_token": "fake-new-token"},
                )

        self.assertEqual(self.destination.read_bytes(), original)
        self.assertEqual(
            list(self.destination.parent.iterdir()),
            [self.destination],
        )


class LoginLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancel = Event()
        self.notices: list[str] = []
        self.callback_wait = Mock(return_value=True)
        self.callback = Mock(wait=self.callback_wait)
        self.server = MagicMock()
        self.server_enter = cast(Mock, self.server.__enter__)
        self.server_enter.return_value = self.server
        self.server_exit = cast(Mock, self.server.__exit__)
        self.server_shutdown = Mock()
        self.server.shutdown = self.server_shutdown
        self.server.server_port = 12345
        self.listener_join = Mock()
        self.listener = Mock(join=self.listener_join)
        self.identity: Identity = {"iss": "https://auth.openai.com", "sub": "fake-account"}
        self.tokens: dict[str, JSONValue] = {
            "access_token": "fake-access-token",
            "id_token": "fake-id-token",
            "scope": "openid profile chatgpt.tokens.use.direct",
        }
        self.credentials = {"access_token": "fake-access-token", "subject": "fake-account"}

        def open_browser(url: str) -> bool:
            self.server.callback_result = {"code": "fake-code", "client_id": "fake-client"}
            return True

        self.browser = Mock(side_effect=open_browser)
        self.exchange = Mock(return_value=self.tokens)
        self.validate = Mock(return_value=self.identity)
        self.save = Mock(return_value=Path("fake-account.json"))
        self.pointer = Mock()
        self.server_factory = Mock(return_value=self.server)
        self.clock = Mock(return_value=0)
        replacements = patch.multiple(
            chatgpt,
            prepare_login_attempt=Mock(return_value={"state": "fake-state", "nonce": "fake-nonce"}),
            get_host_id=Mock(return_value="fake-host"),
            build_authorization_url=Mock(return_value="https://example.test/sign-in"),
            ThreadingHTTPServer=self.server_factory,
            Thread=Mock(return_value=self.listener),
            Event=Mock(return_value=self.callback),
            exchange_authorization_code=self.exchange,
            validate_id_token=self.validate,
            build_credential_record=Mock(return_value=self.credentials),
            save_credential_record=self.save,
            save_private_json=self.pointer,
        )
        replacements.start()
        self.addCleanup(replacements.stop)
        browser = patch.object(chatgpt.webbrowser, "open", self.browser)
        browser.start()
        self.addCleanup(browser.stop)
        clock = patch.object(chatgpt, "monotonic", self.clock, create=True)
        clock.start()
        self.addCleanup(clock.stop)

    def login(self) -> Credentials | None:
        try:
            return chatgpt.login_with_chatgpt(
                cancel_event=self.cancel, notify=self.notices.append
            )
        except TypeError as error:
            self.fail(f"Sign-in notification/cancellation contract is missing or invalid: {error}")

    def assert_closed(self) -> None:
        self.server_shutdown.assert_called_once()
        self.listener_join.assert_called_once()
        self.server_exit.assert_called_once()

    def test_success_sends_notice_without_printing(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            result = self.login()

        self.assertEqual(result, self.credentials)
        self.assertEqual(self.notices, ["Identity verified. ChatGPT plan permission granted."])
        self.assertEqual(output.getvalue(), "")
        self.save.assert_called_once_with(self.credentials)
        self.pointer.assert_called_once_with(
            chatgpt.ACTIVE_SESSION_PATH, {"account_file": "fake-account.json"}
        )
        self.assert_closed()

    def test_missing_plan_permission_uses_notice_callback(self) -> None:
        self.tokens["scope"] = "openid profile"

        self.assertEqual(self.login(), self.credentials)

        self.assertEqual(
            self.notices, ["Identity verified. ChatGPT plan permission was not granted."]
        )
        self.assert_closed()

    def test_pre_cancelled_sign_in_does_not_open_browser_or_save(self) -> None:
        self.cancel.set()

        self.assertIsNone(self.login())

        self.browser.assert_not_called()
        self.server_factory.assert_not_called()
        self.save.assert_not_called()
        self.pointer.assert_not_called()
        self.assertEqual(self.notices, ["Sign-in cancelled"])

    def test_cancellation_while_waiting_closes_listener_without_requesting_tokens(self) -> None:
        def cancel_while_waiting(timeout: float) -> bool:
            self.cancel.set()
            return False

        self.callback_wait.side_effect = cancel_while_waiting

        self.assertIsNone(self.login())

        self.callback_wait.assert_called_once_with(0.1)
        self.exchange.assert_not_called()
        self.save.assert_not_called()
        self.pointer.assert_not_called()
        self.assertEqual(self.notices, ["Sign-in cancelled"])
        self.assert_closed()


    def test_cancellation_during_token_request_does_not_validate_or_save(self) -> None:
        def exchange(*args: object) -> dict[str, JSONValue]:
            self.cancel.set()
            return self.tokens

        self.exchange.side_effect = exchange

        self.assertIsNone(self.login())

        self.validate.assert_not_called()
        self.save.assert_not_called()
        self.pointer.assert_not_called()
        self.assertEqual(self.notices, ["Sign-in cancelled"])
        self.assert_closed()

    def test_cancellation_during_validation_does_not_save_credentials(self) -> None:
        def validate(*args: object) -> Identity:
            self.cancel.set()
            return self.identity

        self.validate.side_effect = validate

        self.assertIsNone(self.login())

        self.save.assert_not_called()
        self.pointer.assert_not_called()
        self.assertEqual(self.notices, ["Sign-in cancelled"])
        self.assert_closed()


    def test_token_error_uses_notice_callback_and_closes_listener(self) -> None:
        self.exchange.side_effect = RuntimeError("Could not request tokens")

        self.assertIsNone(self.login())

        self.assertEqual(self.notices, ["Could not request tokens"])
        self.save.assert_not_called()
        self.assert_closed()


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
        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
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


    async def test_openrouter_turn_uses_the_saved_key_without_a_chatgpt_refresh(self) -> None:
        seen: list[object] = []
        def worker(account: object, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
            seen.append(account)
            events.put(("text", "Done"))
            events.put(("complete", None))
        self.owner.selected_provider = "openrouter"
        self.owner.openrouter = {"key": "sk-or-secret", "label": "test"}
        with patch.object(tui, "stream_openrouter_response", side_effect=worker):
            await asyncio.wait_for(self.start(), 2)
        self.assertEqual(seen, [{"key": "sk-or-secret", "label": "test"}])
        self.assertIsNot(seen[0], self.owner.openrouter)
        self.exchange.assert_not_called()
        self.worker.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
