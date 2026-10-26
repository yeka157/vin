import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vin.provider import preferences


class ModelPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "preferences.json"

        location = patch.object(preferences, "PREFERENCES_PATH", self.destination)

        location.start()
        self.addCleanup(location.stop)


    def test_saved_preference_can_be_loaded(self):
        preferences.save_model_preference("chatgpt", "test-model")

        self.assertEqual(
            preferences.load_model_preference(),
            {"provider": "chatgpt", "model": "test-model"},
        )

    def test_invalid_saved_data_is_rejected(self):
        bodies = [
            "not JSON",
            "[]",
            "{}",
            '{"provider":"unsupported", "model": "test-model"}',
            '{"provider":"chatgpt", "model": ""}',
            '{"provider":"chatgpt", "model": " "}',
            '{"provider":"chatgpt", "model": 123}',
        ]

        for body in bodies:
            with self.subTest(body=body):
                self.destination.write_text(body, encoding="utf-8")

                with self.assertRaises(RuntimeError):
                    preferences.load_model_preference()


    def test_failed_save_preserves_previous_preference(self):
        preferences.save_model_preference("chatgpt", "old-model")
        original = self.destination.read_bytes()

        with patch(
            "vin.auth.credentials.os.replace",
            side_effect=OSError("Simulated write faillure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Could not save model preference"
            ):
                preferences.save_model_preference("chatgpt", "new-model")

        self.assertEqual(self.destination.read_bytes(), original)
        self.assertEqual(
            list(self.destination.parent.iterdir()),
            [self.destination],
        )
