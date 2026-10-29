import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from vin.provider import model_cache
from vin.types import Credentials, ModelMetadata


class ChatGPTModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.cache = model_cache
        except ModuleNotFoundError as error:
            if error.name != "vin.provider.model_cache":
                raise
            self.fail("Implement src/vin/provider/model_cache.py before running these tests.")

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        directory_patch = patch.object(self.cache, "MODEL_CACHE_DIR", self.root)
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        self.credentials: Credentials = {
            "ext_agent_host_id": "fake-host",
            "issuer": "https://auth.openai.com",
            "client_id": "fake-issued-client",
            "subject": "fake-account",
            "access_token": "fake-access-token",
            "refresh_token": "fake-refresh-token",
            "id_token": "fake-id-token",
        }
        self.models: list[ModelMetadata] = [
            {"slug": "old-model", "display_name": "Old Model"},
            {"slug": "new-model", "display_name": "New Model"},
        ]

    def cache_file(self) -> Path:
        files = list(self.root.rglob("*.json"))
        self.assertEqual(len(files), 1)
        return files[0]

    def test_round_trip_stores_only_model_ids_and_display_names(self) -> None:
        self.cache.save_cached_chatgpt_models(
            self.credentials,
            [dict(self.models[0], extra="not-needed-in-cache"), self.models[1]],
        )

        self.assertEqual(
            self.cache.load_cached_chatgpt_models(self.credentials), self.models
        )
        payload = self.cache_file().read_text(encoding="utf-8")
        self.assertEqual(json.loads(payload), self.models)
        for value in ("fake-access-token", "fake-refresh-token", "fake-id-token",
                      "not-needed-in-cache"):
            self.assertNotIn(value, payload)

    def test_cache_is_scoped_to_issuer_client_and_account(self) -> None:
        self.cache.save_cached_chatgpt_models(self.credentials, self.models)

        for field, value in (
            ("issuer", "https://other.example.test"),
            ("client_id", "another-issued-client"),
            ("subject", "another-account"),
        ):
            with self.subTest(field=field):
                other_account = self.credentials.copy()
                other_account[field] = value
                self.assertEqual(
                    self.cache.load_cached_chatgpt_models(other_account), []
                )
                other_models = [{"slug": "other-model", "display_name": "Other Model"}]
                self.cache.save_cached_chatgpt_models(other_account, other_models)
                self.assertEqual(
                    self.cache.load_cached_chatgpt_models(other_account), other_models
                )
                self.assertEqual(
                    self.cache.load_cached_chatgpt_models(self.credentials), self.models
                )


    def test_corrupt_json_and_invalid_encoding_are_ignored(self) -> None:
        self.cache.save_cached_chatgpt_models(self.credentials, self.models)
        path = self.cache_file()
        for content in (b"{unfinished", b"\xff"):
            with self.subTest(content=content):
                path.write_bytes(content)
                self.assertEqual(
                    self.cache.load_cached_chatgpt_models(self.credentials), []
                )


    def test_cache_write_failure_preserves_the_previous_cache(self) -> None:
        self.cache.save_cached_chatgpt_models(self.credentials, self.models)
        replacement = [{"slug": "replacement", "display_name": "Replacement"}]
        with patch.object(
            self.cache, "save_private_json", side_effect=RuntimeError("Disk unavailable")
        ):
            self.cache.save_cached_chatgpt_models(self.credentials, replacement)

        self.assertEqual(self.cache.load_cached_chatgpt_models(self.credentials), self.models)


    def test_missing_account_identity_does_not_read_or_write_a_shared_cache(self) -> None:
        for field in ("issuer", "client_id", "subject"):
            with self.subTest(field=field):
                incomplete: dict[str, object] = dict(self.credentials)
                del incomplete[field]
                self.cache.save_cached_chatgpt_models(incomplete, self.models)
                self.assertEqual(self.cache.load_cached_chatgpt_models(incomplete), [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cache_file_has_owner_only_permissions(self) -> None:
        self.cache.save_cached_chatgpt_models(self.credentials, self.models)
        self.assertEqual(stat.S_IMODE(self.cache_file().stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
