"""OpenRouter sign-in: check and keep the pasted key, sign out per provider, start on the right one."""

import asyncio
import io
import json
import stat
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import to_filter
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.output import DummyOutput
from rich.text import Text

from vin.auth import openrouter
from vin.tui import app as app_module
from vin.tui.app import VinApp, get_chatgpt_model_preference
from vin.tui.bindings import create_bindings
from vin.tui.layout import build_workspace
from vin.types import ModelMetadata, OpenRouterAccount, TranscriptEntry
from tui_application_helpers import fixture_credentials, fixture_registration

KEY = "sk-or-v1-secret"
ACCOUNT: OpenRouterAccount = {"key": KEY, "label": "laptop"}
CHATGPT_MODEL: ModelMetadata = {"slug": "gpt", "display_name": "GPT", "provider": "chatgpt"}
OPENROUTER_MODEL: ModelMetadata = {"slug": "acme/coder", "display_name": "Coder", "provider": "openrouter"}


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def http_error(code: int) -> HTTPError:
    return HTTPError(openrouter.KEY_URL, code, "error", Message(), io.BytesIO(b"{}"))


class CheckKeyTests(unittest.TestCase):
    def test_valid_key_is_sent_as_a_bearer_token_and_returns_the_label(self) -> None:
        body = json.dumps({"data": {"label": "laptop", "usage": 0}}).encode()
        with patch.object(openrouter, "urlopen", return_value=Response(body)) as urlopen:
            self.assertEqual(openrouter.check_openrouter_key(f"  {KEY}\n"), ACCOUNT)

        request = cast(Request, urlopen.call_args.args[0])
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/key")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")

    def test_missing_label_becomes_empty(self) -> None:
        for body in (b"{}", b'{"data": {}}', b'{"data": {"label": 3}}', b"[]"):
            with self.subTest(body=body), patch.object(openrouter, "urlopen", return_value=Response(body)):
                self.assertEqual(openrouter.check_openrouter_key(KEY), {"key": KEY, "label": ""})

    def test_empty_key_is_refused_without_a_request(self) -> None:
        with patch.object(openrouter, "urlopen") as urlopen:
            for key in ("", "   \n"):
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "^Paste an OpenRouter API key.$"):
                    openrouter.check_openrouter_key(key)
        urlopen.assert_not_called()

    def test_failures_have_clear_messages_without_the_key(self) -> None:
        cases: list[tuple[BaseException, str]] = [
            (http_error(401), "OpenRouter key was rejected."),
            (http_error(403), "OpenRouter key was rejected."),
            (http_error(500), "Could not check the OpenRouter key (HTTP 500)."),
            (URLError("offline"), "Could not connect to OpenRouter."),
            (TimeoutError(), "Could not connect to OpenRouter."),
        ]
        for error, message in cases:
            with self.subTest(message=message), patch.object(openrouter, "urlopen", side_effect=error), \
                    self.assertRaises(RuntimeError) as raised:
                openrouter.check_openrouter_key(KEY)
            self.assertEqual(str(raised.exception), message)
            self.assertIsNone(raised.exception.__cause__)

    def test_invalid_json_is_reported(self) -> None:
        with patch.object(openrouter, "urlopen", return_value=Response(b"not json")), \
                self.assertRaisesRegex(RuntimeError, "^OpenRouter returned an invalid response.$"):
            openrouter.check_openrouter_key(KEY)


class AccountFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "vin" / "openrouter.json"
        self.enterContext(patch.object(openrouter, "ACCOUNT_PATH", self.path))

    def test_saved_key_loads_back_from_a_private_file(self) -> None:
        openrouter.save_openrouter_account(ACCOUNT)
        self.assertEqual(openrouter.load_openrouter_account(), ACCOUNT)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_missing_file_means_signed_out(self) -> None:
        self.assertIsNone(openrouter.load_openrouter_account())

    def test_invalid_file_is_rejected(self) -> None:
        self.path.parent.mkdir(parents=True)
        for body in ("not JSON", "[]", "{}", '{"key": "k"}', '{"label": ""}', '{"key": "", "label": ""}',
                     '{"key": 1, "label": ""}', '{"key": "k", "label": null}'):
            with self.subTest(body=body):
                self.path.write_text(body, encoding="utf-8")
                with self.assertRaises(RuntimeError) as raised:
                    openrouter.load_openrouter_account()
                self.assertIn(str(raised.exception),
                              ("Saved OpenRouter key is invalid.", "Could not read the saved OpenRouter key."))

    def test_failed_save_keeps_the_previous_key(self) -> None:
        openrouter.save_openrouter_account(ACCOUNT)
        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "^Could not save the OpenRouter key.$"):
            openrouter.save_openrouter_account({"key": "sk-other", "label": ""})
        self.assertEqual(openrouter.load_openrouter_account(), ACCOUNT)

    def test_delete_removes_the_key_and_tolerates_a_missing_file(self) -> None:
        openrouter.save_openrouter_account(ACCOUNT)
        openrouter.delete_openrouter_account()
        self.assertFalse(self.path.exists())
        openrouter.delete_openrouter_account()

    def test_failed_delete_is_reported(self) -> None:
        with patch.object(Path, "unlink", side_effect=PermissionError()), \
                self.assertRaisesRegex(RuntimeError, "^Could not remove the saved OpenRouter key.$"):
            openrouter.delete_openrouter_account()


def notices(transcript: list[TranscriptEntry]) -> list[str]:
    return [str(entry["content"]) for entry in transcript if entry["role"] == "notice"]


class SecretInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.vin = VinApp()
        self.vin.app = Mock()
        self.buffer = Buffer(document=Document("half-typed draft", cursor_position=4))
        self.vin.session = cast(PromptSession[str], Mock(spec=PromptSession, default_buffer=self.buffer))
        bindings = create_bindings(self.vin)
        self.handlers = {binding.handler.__name__: binding.handler for binding in bindings.bindings}

    def press(self, name: str) -> None:
        event = cast(KeyPressEvent, Mock(current_buffer=self.buffer, app=Mock()))
        _ = self.handlers[name](event)

    async def wait_for_prompt(self) -> None:
        async with asyncio.timeout(1):
            while self.vin.secret_future is None:
                await asyncio.sleep(0)

    async def test_enter_returns_the_typed_key_and_restores_the_draft(self) -> None:
        task = asyncio.create_task(self.vin.ask_secret("Paste key"))
        await self.wait_for_prompt()
        self.assertEqual((self.buffer.text, self.vin.auth_status), ("", "Paste key"))

        self.buffer.text = KEY
        self.press("submit")

        self.assertEqual(await task, KEY)
        self.assertEqual((self.buffer.text, self.buffer.cursor_position), ("half-typed draft", 4))
        self.assertIsNone(self.vin.secret_future)
        self.assertIsNone(self.vin.auth_status)

    async def test_escape_cancels_and_restores_the_draft(self) -> None:
        task = asyncio.create_task(self.vin.ask_secret("Paste key"))
        await self.wait_for_prompt()
        self.buffer.text = "sk-partial"
        self.press("cancel_or_exit")

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.buffer.text, "half-typed draft")
        self.assertIsNone(self.vin.secret_future)

    async def test_a_second_prompt_is_refused_while_one_is_open(self) -> None:
        task = asyncio.create_task(self.vin.ask_secret("Paste key"))
        await self.wait_for_prompt()
        with self.assertRaisesRegex(RuntimeError, "A choice is already open"):
            await self.vin.ask_secret("Again")
        task.cancel()
        _ = await asyncio.gather(task, return_exceptions=True)

    def test_input_is_masked_only_while_a_key_is_asked(self) -> None:
        vin = VinApp()
        with create_app_session(input=DummyInput(), output=DummyOutput()):
            build_workspace(vin)
        masked = to_filter(vin.session.is_password)
        self.assertFalse(masked())
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        vin.secret_future = loop.create_future()
        self.assertTrue(masked())


class SignInTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.vin = VinApp()
        self.vin.app = Mock()
        self.vin.models = [CHATGPT_MODEL]
        self.ask = self.enterContext(patch.object(self.vin, "ask_secret", AsyncMock(return_value=KEY)))
        self.check = self.enterContext(patch.object(app_module, "check_openrouter_key", return_value=ACCOUNT))
        self.save = self.enterContext(patch.object(app_module, "save_openrouter_account"))

    async def test_valid_key_is_saved_and_openrouter_becomes_the_provider(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.selected_provider, self.vin.selected_model = "chatgpt", "gpt"

        await self.vin.sign_in_openrouter()

        self.check.assert_called_once_with(KEY)
        self.save.assert_called_once_with(ACCOUNT)
        self.assertEqual(self.vin.openrouter, ACCOUNT)
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", None))
        self.assertEqual(self.vin.models, [CHATGPT_MODEL])
        self.assertEqual(self.vin.credentials, fixture_credentials())
        self.assertEqual(notices(self.vin.transcript), ["Signed in to OpenRouter · laptop."])

    async def test_key_without_label_signs_in_without_one(self) -> None:
        self.check.return_value = {"key": KEY, "label": ""}
        await self.vin.sign_in_openrouter()
        self.assertEqual(notices(self.vin.transcript), ["Signed in to OpenRouter."])

    async def test_rejected_key_saves_nothing_and_changes_nothing(self) -> None:
        self.check.side_effect = RuntimeError("OpenRouter key was rejected.")
        self.vin.selected_provider, self.vin.selected_model = "chatgpt", "gpt"

        with self.assertRaisesRegex(RuntimeError, "OpenRouter key was rejected."):
            await self.vin.sign_in_openrouter()

        self.save.assert_not_called()
        self.assertIsNone(self.vin.openrouter)
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("chatgpt", "gpt"))

    async def test_escape_signs_in_to_nothing(self) -> None:
        self.ask.side_effect = asyncio.CancelledError()
        await self.vin.sign_in_openrouter()
        self.check.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(self.vin.transcript, [])

    async def test_login_picker_offers_openrouter_and_keeps_chatgpt_signed_in(self) -> None:
        self.vin.credentials = fixture_credentials()
        inline = self.enterContext(patch.object(self.vin, "choose_inline", AsyncMock(return_value="openrouter")))
        sign_out = self.enterContext(patch.object(self.vin, "sign_out", AsyncMock()))
        login = self.enterContext(patch.object(app_module, "login_with_chatgpt"))

        await self.vin.initialize_session(sign_in=True)

        inline.assert_awaited_once_with(
            "Sign in to Vin",
            [("chatgpt", "Sign in with ChatGPT"), ("openrouter", "Use an OpenRouter API key")])
        sign_out.assert_not_awaited()
        login.assert_not_called()
        self.assertEqual(self.vin.selected_provider, "openrouter")
        self.assertEqual(self.vin.credentials, fixture_credentials())


class SignOutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.vin = VinApp()
        self.vin.app = Mock()
        self.vin.openrouter = ACCOUNT
        self.vin.models = [CHATGPT_MODEL, OPENROUTER_MODEL]
        self.delete = self.enterContext(patch.object(app_module, "delete_openrouter_account"))
        self.load_initial = self.enterContext(patch.object(self.vin, "load_initial_model", AsyncMock()))

    async def test_openrouter_logout_switches_to_a_signed_in_chatgpt(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"

        await self.vin.run_auth_command("/logout")

        self.delete.assert_called_once_with()
        self.assertIsNone(self.vin.openrouter)
        self.assertEqual(self.vin.models, [CHATGPT_MODEL])
        self.assertEqual(self.vin.selected_provider, "chatgpt")
        self.load_initial.assert_awaited_once_with()
        self.assertEqual(self.vin.credentials, fixture_credentials())
        self.assertEqual(notices(self.vin.transcript), ["Signed out of OpenRouter."])

    async def test_openrouter_logout_alone_leaves_nothing_selected(self) -> None:
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"
        self.vin.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        await self.vin.run_auth_command("/logout")

        self.assertEqual((self.vin.selected_provider, self.vin.selected_model, self.vin.last_usage), (None, None, None))
        self.load_initial.assert_not_awaited()

    async def test_failed_key_removal_keeps_openrouter_signed_in(self) -> None:
        self.delete.side_effect = RuntimeError("Could not remove the saved OpenRouter key.")
        self.vin.selected_provider = "openrouter"

        with self.assertRaisesRegex(RuntimeError, "Could not remove"):
            await self.vin.sign_out_openrouter()

        self.assertEqual(self.vin.openrouter, ACCOUNT)
        self.assertEqual(self.vin.selected_provider, "openrouter")

    async def test_chatgpt_logout_switches_to_openrouter_and_keeps_its_models(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.selected_provider, self.vin.selected_model = "chatgpt", "gpt"

        with patch.object(app_module, "logout_chatgpt", return_value=fixture_registration()):
            await self.vin.run_auth_command("/logout")

        self.delete.assert_not_called()
        self.assertEqual(self.vin.openrouter, ACCOUNT)
        self.assertEqual(self.vin.models, [OPENROUTER_MODEL])
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", None))

    async def test_chatgpt_relogin_from_openrouter_keeps_the_openrouter_choice(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"

        with patch.object(app_module, "logout_chatgpt", return_value=fixture_registration()):
            await self.vin.sign_out(announce=False)

        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", "acme/coder"))
        self.assertEqual(self.vin.models, [OPENROUTER_MODEL])


class StartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.vin = VinApp()
        self.vin.app = Mock()
        self.account = Mock(return_value=ACCOUNT)
        self.preference = Mock(return_value={"provider": "openrouter", "model": "acme/coder"})
        self.session = Mock(return_value=None)
        self.inline = self.enterContext(patch.object(self.vin, "choose_inline", AsyncMock(return_value="openrouter")))
        self.enterContext(patch.object(self.vin, "sign_in_openrouter", AsyncMock()))
        self.enterContext(patch.multiple(
            app_module, load_openrouter_account=self.account, load_model_preference=self.preference,
            load_active_session=self.session, load_cached_chatgpt_models=Mock(return_value=[CHATGPT_MODEL]),
            load_subagent_settings=Mock(return_value={"enabled": True, "budget": 8})))

    async def test_saved_key_and_openrouter_choice_start_on_openrouter(self) -> None:
        await self.vin.initialize_session()
        self.assertEqual(self.vin.openrouter, ACCOUNT)
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", "acme/coder"))
        self.inline.assert_not_awaited()

    async def test_saved_key_without_a_choice_starts_on_openrouter_with_no_model(self) -> None:
        self.preference.return_value = None
        await self.vin.initialize_session()
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", None))
        self.inline.assert_not_awaited()

    async def test_chatgpt_choice_wins_when_chatgpt_is_signed_in(self) -> None:
        self.preference.return_value = {"provider": "chatgpt", "model": "gpt"}
        self.session.return_value = fixture_credentials()
        await self.vin.initialize_session()
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("chatgpt", "gpt"))
        self.assertEqual(self.vin.models, [CHATGPT_MODEL])
        self.assertEqual(self.vin.openrouter, ACCOUNT)

    async def test_chatgpt_choice_falls_back_to_openrouter_when_chatgpt_is_signed_out(self) -> None:
        self.preference.return_value = {"provider": "chatgpt", "model": "gpt"}
        await self.vin.initialize_session()
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), ("openrouter", None))
        self.inline.assert_not_awaited()

    async def test_openrouter_choice_without_a_key_does_not_select_openrouter(self) -> None:
        self.account.return_value = None
        self.session.return_value = fixture_credentials()
        load_initial = self.enterContext(patch.object(self.vin, "load_initial_model", AsyncMock()))
        await self.vin.initialize_session()
        self.assertIsNone(self.vin.selected_provider)
        load_initial.assert_awaited_once_with()

    async def test_bad_key_file_shows_a_red_notice_and_offers_sign_in(self) -> None:
        self.account.side_effect = RuntimeError("Saved OpenRouter key is invalid.")
        await self.vin.initialize_session()
        entry = self.vin.transcript[0]
        assert isinstance(entry["content"], Text)
        self.assertEqual((str(entry["content"]), entry["content"].style), ("Saved OpenRouter key is invalid.", "red"))
        self.assertIsNone(self.vin.openrouter)
        self.inline.assert_awaited_once()


class AccountDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vin = VinApp()

    def user(self) -> str:
        notice = self.vin.handle_command("/user")
        assert isinstance(notice, Text)
        return notice.plain

    def test_user_lists_every_signed_in_account(self) -> None:
        self.vin.credentials = {**fixture_credentials(), "email": "me@example.test"}
        self.vin.openrouter = ACCOUNT
        self.assertEqual(self.user(), "ChatGPT: me@example.test\nOpenRouter: laptop")

    def test_user_shows_openrouter_alone_and_never_the_key(self) -> None:
        self.vin.openrouter = {"key": KEY, "label": ""}
        self.assertEqual(self.user(), "OpenRouter: API key")

    def test_user_when_signed_out_of_both(self) -> None:
        self.assertEqual(self.user(), "Signed out. Use /login to sign in")

    def test_indicator_shows_openrouter_before_a_model_is_picked(self) -> None:
        self.vin.selected_provider = "openrouter"
        self.assertEqual(self.vin.model_indicator_text(), " OpenRouter · No model selected")

    def test_set_models_replaces_only_that_providers_models(self) -> None:
        self.vin.models = [CHATGPT_MODEL, OPENROUTER_MODEL]
        newer: ModelMetadata = {"slug": "gpt-2", "display_name": "GPT 2", "provider": "chatgpt"}
        self.vin.set_models("chatgpt", [newer])
        self.assertEqual(self.vin.models, [OPENROUTER_MODEL, newer])

    def test_saved_openrouter_choice_is_not_a_missing_chatgpt_model(self) -> None:
        transcript: list[TranscriptEntry] = []
        with patch.object(app_module, "load_model_preference",
                          return_value={"provider": "openrouter", "model": "acme/coder"}):
            self.assertIsNone(get_chatgpt_model_preference(transcript, [CHATGPT_MODEL]))
        self.assertEqual(transcript, [])


if __name__ == "__main__":
    unittest.main()
