"""Saved model choice and subagent settings, /settings and /tasks."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from rich.text import Text

from provider_typing_helpers import credentials as fixture_credentials
from vin.provider import preferences
from vin.tui import app as app_module
from vin.tui.app import VinApp
from vin.types import Subagent, SubagentSettings


class ModelPreferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "preferences.json"

        location = patch.object(preferences, "PREFERENCES_PATH", self.destination)

        location.start()
        self.addCleanup(location.stop)


    def test_saved_preference_can_be_loaded(self) -> None:
        preferences.save_model_preference("chatgpt", "test-model")

        self.assertEqual(
            preferences.load_model_preference(),
            {"provider": "chatgpt", "model": "test-model"},
        )

    def test_openrouter_preference_can_be_saved_and_loaded(self) -> None:
        preferences.save_model_preference("openrouter", "acme/coder")

        self.assertEqual(
            preferences.load_model_preference(),
            {"provider": "openrouter", "model": "acme/coder"},
        )

    def test_invalid_saved_data_is_rejected(self) -> None:
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


    def test_failed_save_preserves_previous_preference(self) -> None:
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


def subagent(task_id: str, description: str, status: Literal["finished", "failed"] = "finished") -> Subagent:
    return {"id": task_id, "description": description, "status": status,
            "started_at": 0.0, "duration": 1.0, "tokens": 0, "transcript": []}


class SubagentSettingsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.destination = Path(directory.name) / "settings.json"
        self.enterContext(patch.object(preferences, "SETTINGS_PATH", self.destination))

    def test_missing_file_gives_a_fresh_copy_of_the_defaults(self) -> None:
        settings = preferences.load_subagent_settings()
        self.assertEqual(settings, {"enabled": True, "budget": 8})
        settings["budget"] = 1
        self.assertEqual(preferences.DEFAULT_SUBAGENT_SETTINGS["budget"], 8)

    def test_saved_settings_load_back_including_the_limits(self) -> None:
        for budget in (1, 32):
            with self.subTest(budget=budget):
                preferences.save_subagent_settings({"enabled": False, "budget": budget})
                self.assertEqual(preferences.load_subagent_settings(), {"enabled": False, "budget": budget})

    def test_invalid_saved_settings_are_rejected(self) -> None:
        for body in ("not JSON", "[]", "{}", '{"enabled":true}', '{"budget":8}', '{"enabled":"yes","budget":8}',
                     '{"enabled":true,"budget":0}', '{"enabled":true,"budget":33}',
                     '{"enabled":true,"budget":true}', '{"enabled":true,"budget":8.0}'):
            with self.subTest(body=body):
                self.destination.write_text(body, encoding="utf-8")
                with self.assertRaises(RuntimeError) as raised:
                    preferences.load_subagent_settings()
                self.assertIn(str(raised.exception), ("Saved settings are invalid.", "Could not read saved settings."))

    def test_invalid_settings_are_not_saved(self) -> None:
        for budget in (0, 33):
            with self.subTest(budget=budget), self.assertRaisesRegex(RuntimeError, "Invalid settings."):
                preferences.save_subagent_settings({"enabled": True, "budget": budget})
        self.assertFalse(self.destination.exists())

    def test_failed_save_keeps_the_previous_file(self) -> None:
        preferences.save_subagent_settings({"enabled": True, "budget": 8})
        original = self.destination.read_bytes()

        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            preferences.save_subagent_settings({"enabled": False, "budget": 2})

        self.assertEqual(self.destination.read_bytes(), original)


class SettingsCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = VinApp()
        self.inline = AsyncMock()
        self.save = Mock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.enterContext(patch.object(app_module, "save_subagent_settings", self.save))

    def test_menu_shows_current_values_and_toggles_subagents(self) -> None:
        self.inline.return_value = "enabled"

        self.assertEqual(asyncio.run(self.app.change_settings()), "Subagents off · request limit 8")

        self.inline.assert_awaited_once_with(
            "Settings", [("enabled", "Subagents: on"), ("budget", "Subagent request limit: 8")])
        off: SubagentSettings = {"enabled": False, "budget": 8}
        self.save.assert_called_once_with(off)
        self.assertEqual(self.app.subagent_settings, off)

    def test_request_limit_is_picked_from_one_to_thirty_two(self) -> None:
        self.inline.side_effect = ["budget", "3"]

        self.assertEqual(asyncio.run(self.app.change_settings()), "Subagents on · request limit 3")

        self.assertEqual(self.inline.await_args_list[1], call(
            "Subagent request limit", [(str(number), str(number)) for number in range(1, 33)], default="8"))
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 3})

    def test_escape_in_either_picker_changes_nothing(self) -> None:
        for side_effect in ([asyncio.CancelledError()], ["budget", asyncio.CancelledError()]):
            with self.subTest(side_effect=side_effect):
                self.inline.reset_mock()
                self.inline.side_effect = side_effect
                self.assertIsNone(asyncio.run(self.app.change_settings()))
                self.save.assert_not_called()
                self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8})

    def test_failed_save_keeps_the_current_settings(self) -> None:
        self.inline.return_value = "enabled"
        self.save.side_effect = RuntimeError("Could not save settings.")

        with self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            asyncio.run(self.app.change_settings())
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8})

    def test_bad_settings_file_shows_a_notice_and_keeps_defaults(self) -> None:
        self.app.credentials = fixture_credentials("token")
        with patch.multiple(app_module, create=True,
                            load_subagent_settings=Mock(side_effect=RuntimeError("Saved settings are invalid.")),
                            load_model_preference=Mock(return_value={"provider": "chatgpt", "model": "m"}),
                            load_cached_chatgpt_models=Mock(return_value=[])):
            asyncio.run(self.app.initialize_session())

        entry, = self.app.transcript
        self.assertEqual(str(entry["content"]), "Saved settings are invalid.")
        assert isinstance(entry["content"], Text)
        self.assertEqual(entry["content"].style, "red")
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8})
        self.assertEqual(self.app.selected_model, "m")


class TasksCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = VinApp()
        self.inline = AsyncMock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.scroll = Mock()
        self.enterContext(patch.object(self.app.transcript_control, "scroll_to_bottom", self.scroll))

    def test_no_subagents_says_so_without_a_picker(self) -> None:
        self.assertEqual(asyncio.run(self.app.open_task()), "No subagents in this conversation.")
        self.inline.assert_not_awaited()

    def test_lists_every_subagent_newest_first_and_opens_the_chosen_one(self) -> None:
        self.app.subagents = [subagent("call_a", "Old"), subagent("call_b", "New", "failed")]
        self.app.panel_start = 2
        self.app.panel_index = 0
        self.inline.return_value = "call_a"

        self.assertIsNone(asyncio.run(self.app.open_task()))

        self.inline.assert_awaited_once_with(
            "Open a subagent", [("call_b", "New · failed"), ("call_a", "Old · finished")])
        self.assertEqual((self.app.viewed, self.app.panel_index), ("call_a", None))
        self.scroll.assert_called_once_with()

    def test_escape_keeps_the_current_view(self) -> None:
        self.app.subagents = [subagent("call_a", "Old")]
        self.inline.side_effect = asyncio.CancelledError()

        self.assertIsNone(asyncio.run(self.app.open_task()))
        self.assertIsNone(self.app.viewed)


if __name__ == "__main__":
    unittest.main()
