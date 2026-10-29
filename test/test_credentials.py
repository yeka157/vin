import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vin.auth.credentials import load_credential_record, save_private_json
from vin.types import Registration
from vin.tui.app import VinApp


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
