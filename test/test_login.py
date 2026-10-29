import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock, patch
from typing import cast
from vin.types import Credentials, Identity, JSONValue

from vin.auth import chatgpt


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
