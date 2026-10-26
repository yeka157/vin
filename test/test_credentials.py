import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vin.auth.credentials import save_private_json


class CredentialStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "account.json"

    def test_saved_credentials_are_private(self):
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

    def test_failed_save_preserves_existing_credentials(self):
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
