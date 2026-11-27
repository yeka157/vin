"""Saved model choice and subagent settings, /settings and /tasks."""

import asyncio
from email.message import Message
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from rich.text import Text

from provider_typing_helpers import credentials as fixture_credentials
from vin.provider import preferences
from vin.tui import app as app_module
from vin.tui.app import VinApp
from vin.auth import search
from vin.types import SearchSettings, Subagent, SubagentSettings


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def http_error(code: int) -> HTTPError:
    return HTTPError(search.TAVILY_SEARCH_URL, code, "error", Message(), io.BytesIO(b"{}"))


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
        self.assertEqual(settings, {"enabled": True, "budget": 8, "fetch": True})
        settings["budget"] = 1
        self.assertEqual(preferences.DEFAULT_SUBAGENT_SETTINGS["budget"], 8)

    def test_saved_settings_load_back_including_the_limits(self) -> None:
        for budget in (1, 32):
            with self.subTest(budget=budget):
                preferences.save_subagent_settings({"enabled": False, "budget": budget, "fetch": True})
                self.assertEqual(preferences.load_subagent_settings(), {"enabled": False, "budget": budget, "fetch": True})

    def test_invalid_saved_settings_are_rejected(self) -> None:
        for body in ("not JSON", "[]", "{}", '{"enabled":true}', '{"budget":8}', '{"enabled":"yes","budget":8}',
                     '{"enabled":true,"budget":0}', '{"enabled":true,"budget":33}',
                     '{"enabled":true,"budget":true}', '{"enabled":true,"budget":8.0}',
                     '{"enabled":true,"budget":8,"fetch":"no"}', '{"enabled":true,"budget":8,"fetch":null}'):
            with self.subTest(body=body):
                self.destination.write_text(body, encoding="utf-8")
                with self.assertRaises(RuntimeError) as raised:
                    preferences.load_subagent_settings()
                self.assertIn(str(raised.exception), ("Saved settings are invalid.", "Could not read saved settings."))

    def test_file_from_before_web_fetch_loads_with_web_fetch_on(self) -> None:
        self.destination.write_text('{"enabled": false, "budget": 3}', encoding="utf-8")
        self.assertEqual(preferences.load_subagent_settings(), {"enabled": False, "budget": 3, "fetch": True})

    def test_web_fetch_off_is_saved_and_loads_back(self) -> None:
        preferences.save_subagent_settings({"enabled": True, "budget": 8, "fetch": False})
        self.assertEqual(preferences.load_subagent_settings(), {"enabled": True, "budget": 8, "fetch": False})

    def test_invalid_settings_are_not_saved(self) -> None:
        for budget in (0, 33):
            with self.subTest(budget=budget), self.assertRaisesRegex(RuntimeError, "Invalid settings."):
                preferences.save_subagent_settings({"enabled": True, "budget": budget, "fetch": True})
        self.assertFalse(self.destination.exists())

    def test_failed_save_keeps_the_previous_file(self) -> None:
        preferences.save_subagent_settings({"enabled": True, "budget": 8, "fetch": True})
        original = self.destination.read_bytes()

        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            preferences.save_subagent_settings({"enabled": False, "budget": 2, "fetch": True})

        self.assertEqual(self.destination.read_bytes(), original)


class SettingsCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = VinApp()
        self.inline = AsyncMock()
        self.save = Mock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.enterContext(patch.object(app_module, "save_subagent_settings", self.save))

    def test_settings_opens_a_menu_per_topic(self) -> None:
        self.inline.side_effect = ["subagents", asyncio.CancelledError()]

        self.assertIsNone(asyncio.run(self.app.change_settings()))

        self.assertEqual(self.inline.await_args_list, [
            call("Settings", [("subagents", "Subagents           ›"), ("search", "Web search          ›"),
                              ("fetch", "Web fetch           on")]),
            call("Subagents", [("enabled", "Subagents           on"), ("budget", "Request limit       8 ›")]),
        ])
        self.save.assert_not_called()

    def test_menu_shows_current_values_and_toggles_subagents(self) -> None:
        self.inline.side_effect = ["subagents", "enabled"]

        self.assertEqual(asyncio.run(self.app.change_settings()), "Subagents off · request limit 8")

        off: SubagentSettings = {"enabled": False, "budget": 8, "fetch": True}
        self.save.assert_called_once_with(off)
        self.assertEqual(self.app.subagent_settings, off)

    def test_web_fetch_row_toggles_and_saves_without_touching_subagents(self) -> None:
        self.inline.side_effect = ["fetch"]
        self.assertEqual(asyncio.run(self.app.change_settings()), "Web fetch off")
        off: SubagentSettings = {"enabled": True, "budget": 8, "fetch": False}
        self.save.assert_called_once_with(off)
        self.assertEqual(self.app.subagent_settings, off)

        self.inline.reset_mock()
        self.inline.side_effect = ["fetch"]
        self.assertEqual(asyncio.run(self.app.change_settings()), "Web fetch on")
        self.assertEqual(self.inline.await_args_list[0].args[1][2], ("fetch", "Web fetch           off"))
        self.assertEqual(self.app.subagent_settings["fetch"], True)

    def test_changing_subagents_keeps_web_fetch_off(self) -> None:
        self.app.subagent_settings = {"enabled": True, "budget": 8, "fetch": False}
        for side_effect, expected in ((["subagents", "enabled"], {"enabled": False, "budget": 8, "fetch": False}),
                                      (["subagents", "budget", "5"], {"enabled": True, "budget": 5, "fetch": False})):
            with self.subTest(side_effect=side_effect):
                self.app.subagent_settings = {"enabled": True, "budget": 8, "fetch": False}
                self.inline.side_effect = side_effect
                _ = asyncio.run(self.app.change_settings())
                self.assertEqual(self.app.subagent_settings, expected)

    def test_failed_web_fetch_save_keeps_it_on(self) -> None:
        self.inline.side_effect = ["fetch"]
        self.save.side_effect = RuntimeError("Could not save settings.")
        with self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            asyncio.run(self.app.change_settings())
        self.assertTrue(self.app.subagent_settings["fetch"])

    def test_request_limit_is_picked_from_one_to_thirty_two(self) -> None:
        self.inline.side_effect = ["subagents", "budget", "3"]

        self.assertEqual(asyncio.run(self.app.change_settings()), "Subagents on · request limit 3")

        self.assertEqual(self.inline.await_args_list[2], call(
            "Subagent request limit", [(str(number), str(number)) for number in range(1, 33)], default="8"))
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 3, "fetch": True})

    def test_escape_in_either_picker_changes_nothing(self) -> None:
        for side_effect in ([asyncio.CancelledError()], ["subagents", asyncio.CancelledError()],
                            ["subagents", "budget", asyncio.CancelledError()]):
            with self.subTest(side_effect=side_effect):
                self.inline.reset_mock()
                self.inline.side_effect = side_effect
                self.assertIsNone(asyncio.run(self.app.change_settings()))
                self.save.assert_not_called()
                self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8, "fetch": True})

    def test_failed_save_keeps_the_current_settings(self) -> None:
        self.inline.side_effect = ["subagents", "enabled"]
        self.save.side_effect = RuntimeError("Could not save settings.")

        with self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            asyncio.run(self.app.change_settings())
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8, "fetch": True})

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
        self.assertEqual(self.app.subagent_settings, {"enabled": True, "budget": 8, "fetch": True})
        self.assertEqual(self.app.selected_model, "m")


KEY = "tvly-dev-secret-key"
ON: SearchSettings = {"key": KEY, "enabled": True}
OFF: SearchSettings = {"key": KEY, "enabled": False}


class SearchSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "search.json"
        self.enterContext(patch.object(search, "SEARCH_KEY_PATH", self.path))

    def test_saved_settings_load_back_and_delete(self) -> None:
        self.assertIsNone(search.load_search_settings())
        _ = search.save_search_settings(ON)
        self.assertEqual(search.load_search_settings(), ON)
        self.assertEqual(search.load_search_key(), KEY)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

        search.delete_search_settings()
        self.assertFalse(self.path.exists())
        search.delete_search_settings()

    def test_enabled_must_be_a_boolean(self) -> None:
        for enabled in (None, 1, "true"):
            with self.subTest(enabled=enabled):
                _ = self.path.write_text(json.dumps({"key": KEY, "enabled": enabled}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Saved web search key is invalid"):
                    _ = search.load_search_settings()

    def test_failed_save_and_delete_say_so(self) -> None:
        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "Could not save the web search key"):
                _ = search.save_search_settings(ON)
        with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "Could not remove the saved web search key"):
                search.delete_search_settings()


class CheckSearchKeyTests(unittest.TestCase):
    def test_key_is_trimmed_and_checked_with_one_search(self) -> None:
        with patch.object(search, "urlopen", return_value=Response(b"{}")) as urlopen:
            self.assertEqual(search.check_search_key(f"  {KEY}\n"), KEY)

        request = cast(Request, urlopen.call_args.args[0])
        self.assertEqual(request.full_url, search.TAVILY_SEARCH_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(cast(bytes, request.data))["max_results"], 1)

    def test_empty_key_is_rejected_without_a_request(self) -> None:
        with patch.object(search, "urlopen") as urlopen:
            with self.assertRaisesRegex(RuntimeError, "Paste a Tavily API key"):
                _ = search.check_search_key("   ")
        urlopen.assert_not_called()

    def test_failures_have_their_own_messages_without_the_key(self) -> None:
        cases: list[tuple[BaseException, str]] = [
            (http_error(401), "Tavily key was rejected."),
            (http_error(403), "Tavily key was rejected."),
            (http_error(432), "Could not check the Tavily key (HTTP 432)."),
            (http_error(500), "Could not check the Tavily key (HTTP 500)."),
            (URLError("offline"), "Could not connect to Tavily."),
            (TimeoutError("timed out"), "Could not connect to Tavily."),
        ]
        for error, message in cases:
            with self.subTest(message=message), patch.object(search, "urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as raised:
                    _ = search.check_search_key(KEY)
                self.assertEqual(str(raised.exception), message)
                self.assertIsNone(raised.exception.__cause__)


class SearchSettingsCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = VinApp()
        self.inline = AsyncMock()
        self.secret = AsyncMock(return_value=f" {KEY} ")
        def check(key: str) -> str:
            return key.strip()

        self.check = Mock(side_effect=check)
        self.save = Mock()
        self.delete = Mock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.enterContext(patch.object(self.app, "ask_secret", self.secret))
        self.enterContext(patch.object(self.app, "app", Mock(), create=True))
        self.enterContext(patch.multiple(app_module, check_search_key=self.check,
                                         save_search_settings=self.save,
                                         delete_search_settings=self.delete,
                                         save_subagent_settings=Mock()))

    def settings(self) -> str | Text | None:
        return asyncio.run(self.app.change_settings())

    def test_menu_shows_the_key_row_only_when_a_key_is_saved(self) -> None:
        self.inline.side_effect = ["search", asyncio.CancelledError()]
        self.assertIsNone(self.settings())
        self.assertEqual(self.inline.await_args_list[-1], call("Web search", [("enabled", "Web search          off")]))

        self.inline.side_effect = ["search", asyncio.CancelledError()]
        self.app.search_settings = ON
        self.assertIsNone(self.settings())
        rows = cast(list[tuple[str, str]], self.inline.await_args_list[-1].args[1])
        self.assertEqual(rows, [("enabled", "Web search          on"), ("key", "Key                 tvly-dev-•••-key ›")])

    def test_first_turn_on_asks_for_the_key_checks_it_and_saves_it_on(self) -> None:
        self.inline.side_effect = ["search", "enabled"]

        self.assertEqual(self.settings(), "Web search on")

        self.secret.assert_awaited_once_with("Paste your Tavily API key")
        self.check.assert_called_once_with(f" {KEY} ")
        self.save.assert_called_once_with(ON)
        self.assertEqual(self.app.search_settings, ON)

    def test_spinner_says_checking_while_the_key_is_checked_and_clears_after(self) -> None:
        seen: list[tuple[str | None, bool]] = []

        def check(key: str) -> str:
            seen.append((self.app.auth_status, self.app.loading_model))
            return key.strip()

        for loading in (False, True):
            with self.subTest(loading=loading):
                self.app.search_settings = None
                self.app.loading_model = loading
                self.check.side_effect = check
                self.inline.side_effect = ["search", "enabled"]
                self.assertEqual(self.settings(), "Web search on")
                self.assertEqual(seen[-1], ("Checking Tavily key…", True))
                self.assertEqual((self.app.auth_status, self.app.loading_model), (None, loading))

    def test_spinner_clears_when_the_key_is_rejected(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.check.side_effect = RuntimeError("Tavily key was rejected.")

        with self.assertRaisesRegex(RuntimeError, "rejected"):
            _ = self.settings()
        self.assertEqual((self.app.auth_status, self.app.loading_model), (None, False))

    def test_bad_key_saves_nothing_and_stays_off(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.check.side_effect = RuntimeError("Tavily key was rejected.")

        with self.assertRaisesRegex(RuntimeError, "Tavily key was rejected"):
            _ = self.settings()
        self.save.assert_not_called()
        self.assertIsNone(self.app.search_settings)

    def test_escape_at_the_key_prompt_saves_nothing(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.secret.side_effect = asyncio.CancelledError()

        self.assertIsNone(self.settings())
        self.check.assert_not_called()
        self.save.assert_not_called()
        self.assertIsNone(self.app.search_settings)

    def test_with_a_saved_key_the_switch_flips_without_asking(self) -> None:
        self.inline.side_effect = ["search", "enabled", "search", "enabled"]
        self.app.search_settings = ON
        self.assertEqual(self.settings(), "Web search off")
        self.assertEqual(self.app.search_settings, OFF)
        self.assertEqual(self.settings(), "Web search on")
        self.assertEqual(self.app.search_settings, ON)
        self.secret.assert_not_awaited()
        self.assertEqual(self.save.call_args_list, [call(OFF), call(ON)])

    def test_failed_save_keeps_the_current_setting(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.app.search_settings = OFF
        self.save.side_effect = RuntimeError("Could not save the web search key.")

        with self.assertRaisesRegex(RuntimeError, "Could not save the web search key"):
            _ = self.settings()
        self.assertEqual(self.app.search_settings, OFF)

    def test_replacing_the_key_keeps_the_switch(self) -> None:
        self.inline.side_effect = ["search", "key", "replace"]
        self.app.search_settings = OFF
        self.secret.return_value = "tvly-dev-another-key"

        self.assertEqual(self.settings(), "Web search key replaced · tvly-dev-•••-key")

        self.assertEqual(self.inline.await_args_list[2],
                         call("Web search key", [("replace", "Replace key"), ("remove", "Remove key")]))
        replaced: SearchSettings = {"key": "tvly-dev-another-key", "enabled": False}
        self.save.assert_called_once_with(replaced)
        self.assertEqual(self.app.search_settings, replaced)

    def test_removing_the_key_turns_web_search_off(self) -> None:
        self.inline.side_effect = ["search", "key", "remove"]
        self.app.search_settings = ON

        self.assertEqual(self.settings(), "Web search key removed · web search off")

        self.delete.assert_called_once_with()
        self.save.assert_not_called()
        self.assertIsNone(self.app.search_settings)

    def test_escape_in_the_key_menu_or_prompt_changes_nothing(self) -> None:
        for side_effect, secret in (
            (["search", asyncio.CancelledError()], None),
            (["search", "key", asyncio.CancelledError()], None),
            (["search", "key", "replace"], asyncio.CancelledError()),
        ):
            with self.subTest(secret=secret):
                self.inline.side_effect = side_effect
                self.secret.side_effect = secret
                self.app.search_settings = ON
                self.assertIsNone(self.settings())
                self.save.assert_not_called()
                self.delete.assert_not_called()
                self.assertEqual(self.app.search_settings, ON)

    def test_saved_settings_load_at_startup_and_a_bad_file_shows_a_notice(self) -> None:
        self.app.credentials = fixture_credentials("token")
        common = {
            "load_subagent_settings": Mock(return_value={"enabled": True, "budget": 8, "fetch": True}),
            "load_model_preference": Mock(return_value={"provider": "chatgpt", "model": "m"}),
            "load_cached_chatgpt_models": Mock(return_value=[]),
            "load_openrouter_account": Mock(return_value=None),
        }
        with patch.multiple(app_module, create=True, load_search_settings=Mock(return_value=ON), **common):
            asyncio.run(self.app.initialize_session())
        self.assertEqual(self.app.search_settings, ON)
        self.assertEqual(self.app.transcript, [])

        app = VinApp()
        app.credentials = fixture_credentials("token")
        bad = Mock(side_effect=RuntimeError("Saved web search key is invalid."))
        with patch.multiple(app_module, create=True, load_search_settings=bad, **common):
            asyncio.run(app.initialize_session())
        entry, = app.transcript
        self.assertEqual(str(entry["content"]), "Saved web search key is invalid.")
        self.assertIsNone(app.search_settings)


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
