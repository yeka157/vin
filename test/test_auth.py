import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from urllib.error import HTTPError
from unittest.mock import patch

from vin.auth import chatgpt


class TokenExchangeErrorTests(unittest.TestCase):
    def test_terminal_refresh_codes_advise_login_without_raw_error_details(self):
        for code in chatgpt.TokenExchangeError.LOGIN_CODES:
            for detail in (f'"{code}"', f'{{"code":"{code}"}}'):
                with self.subTest(code=code, detail=detail):
                    body = BytesIO(
                        ('{"error":' + detail + ',"error_description":"secret-token"}').encode()
                    )
                    error = HTTPError("https://example.test", 400, "secret-token", {}, body)
                    with patch.object(chatgpt, "urlopen", side_effect=error):
                        with self.assertRaises(chatgpt.TokenExchangeError) as caught:
                            chatgpt.request_tokens({"refresh_token": "secret-token"})
                    self.assertTrue(caught.exception.requires_login)
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.status_code, 400)
                    self.assertNotIn("secret-token", str(caught.exception))
                    self.assertTrue(body.closed)

    def test_unrecognized_and_unreadable_errors_do_not_advise_login_or_leak(self):
        for payload in (
            b'{"error":"server_error","message":"secret-token"}',
            b'{"error":{"code":"secret-token"}}',
            b'{"error":["secret-token"]}',
            b'secret-token', b'\xff', b'[]', b'{}',
        ):
            with self.subTest(payload=payload):
                body = BytesIO(payload)
                error = HTTPError("https://example.test", 503, "secret-token", {}, body)
                with patch.object(chatgpt, "urlopen", side_effect=error):
                    with self.assertRaises(chatgpt.TokenExchangeError) as caught:
                        chatgpt.request_tokens({})
                self.assertFalse(caught.exception.requires_login)
                self.assertEqual(str(caught.exception), "Token exchange failed (HTTP 503)")
                self.assertTrue(body.closed)


class RefreshCredentialsTests(unittest.TestCase):
    def setUp(self):
        self.credentials = {
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

    def test_unexpired_credentials_are_reused(self):
        self.credentials["saved_at"] = datetime.now(timezone.utc).isoformat()

        with (
            patch.object(chatgpt, "exchange_refresh_token") as exchange,
            patch.object(chatgpt, "save_credential_record") as save,
        ):
            result = chatgpt.refresh_credentials(self.credentials)

        self.assertIs(result, self.credentials)
        exchange.assert_not_called()
        save.assert_not_called()

    def test_expired_credentials_are_refreshed_and_saved(self):
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

        self.assertEqual(result["access_token"], "fake-new-access")
        self.assertEqual(result["refresh_token"], "fake-new-refresh")

        for field in (
            "issuer",
            "subject",
            "email",
            "client_id",
            "ext_agent_host_id",
            "id_token",
        ):
            self.assertEqual(result[field], original[field])

        self.assertCountEqual(result["scopes"], original["scopes"])
        self.assertFalse(chatgpt.is_access_token_expired(result))
        self.assertEqual(self.credentials, original)


    def test_invalid_refresh_does_not_save_or_change_credentials(self):
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
    def setUp(self):
        self.credentials = {
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

    def test_revocation_warning_uses_callback_without_printing_and_tokens_are_removed(self):
        notices = []
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
