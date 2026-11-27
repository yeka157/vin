"""Saved model choice, settings and web search key, /settings and /tasks."""

import asyncio
from email.message import Message
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Unpack, cast
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from rich.text import Text

from tui_application_helpers import patch_app
from provider_typing_helpers import credentials as fixture_credentials
from vin import settings as settings_module
from vin.provider import preferences
from vin.tui import settings_menu as settings_menu_module
from vin.tui.app import VinApp
from vin.auth import search
from vin.types.settings import Settings, SettingsChange
from vin.types.transcript import Subagent


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


DEFAULTS: Settings = {"subagents": True, "subagent_budget": 8, "search": False, "fetch": True}


def changed(**changes: Unpack[SettingsChange]) -> Settings:
    return {**DEFAULTS, **changes}


class SettingsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.destination = Path(directory.name) / "settings.json"
        self.search_file = Path(directory.name) / "search.json"
        self.enterContext(patch.object(settings_module, "SETTINGS_PATH", self.destination))
        self.enterContext(patch.object(search, "SEARCH_KEY_PATH", self.search_file))

    def write(self, value: object, path: Path | None = None) -> None:
        _ = (path or self.destination).write_text(json.dumps(value), encoding="utf-8")

    def test_missing_file_gives_a_fresh_copy_of_the_defaults(self) -> None:
        settings = settings_module.load_settings()
        self.assertEqual(settings, DEFAULTS)
        settings["subagent_budget"] = 1
        self.assertEqual(settings_module.DEFAULT_SETTINGS["subagent_budget"], 8)

    def test_saved_settings_load_back_including_the_limits(self) -> None:
        for saved in (changed(subagent_budget=1, subagents=False), changed(subagent_budget=32, search=True, fetch=False)):
            with self.subTest(saved=saved):
                _ = settings_module.save_settings(saved)
                self.assertEqual(settings_module.load_settings(), saved)
                self.assertEqual(self.destination.stat().st_mode & 0o777, 0o600)

    def test_missing_names_take_their_defaults(self) -> None:
        self.write({"subagents": False})
        self.assertEqual(settings_module.load_settings(), changed(subagents=False))

    def test_file_from_an_older_vin_loads_under_the_new_names(self) -> None:
        self.write({"enabled": False, "budget": 3})
        self.assertEqual(settings_module.load_settings(), changed(subagents=False, subagent_budget=3))

    def test_web_search_switch_saved_next_to_an_older_key_is_kept(self) -> None:
        for enabled, expected in ((True, True), (False, False), ("yes", False)):
            with self.subTest(enabled=enabled):
                self.write({"key": "tvly-key", "enabled": enabled}, self.search_file)
                self.assertIs(settings_module.load_settings()["search"], expected)

        self.write({"search": False})
        self.write({"key": "tvly-key", "enabled": True}, self.search_file)
        self.assertIs(settings_module.load_settings()["search"], False)

        _ = self.search_file.write_text("{", encoding="utf-8")
        self.destination.unlink()
        self.assertIs(settings_module.load_settings()["search"], False)

    def test_invalid_saved_settings_are_rejected(self) -> None:
        for body in ("not JSON", "[]", '{"subagents":"yes"}', '{"subagent_budget":0}', '{"subagent_budget":33}',
                     '{"subagent_budget":true}', '{"subagent_budget":8.0}', '{"budget":0}', '{"fetch":"no"}',
                     '{"fetch":null}', '{"search":1}', '{"unknown":true}'):
            with self.subTest(body=body):
                _ = self.destination.write_text(body, encoding="utf-8")
                with self.assertRaises(RuntimeError) as raised:
                    _ = settings_module.load_settings()
                self.assertIn(str(raised.exception), ("Saved settings are invalid.", "Could not read saved settings."))

    def test_invalid_settings_are_not_saved(self) -> None:
        for budget in (0, 33):
            with self.subTest(budget=budget), self.assertRaisesRegex(RuntimeError, "Invalid settings."):
                _ = settings_module.save_settings(changed(subagent_budget=budget))
        self.assertFalse(self.destination.exists())

    def test_failed_save_keeps_the_previous_file(self) -> None:
        _ = settings_module.save_settings(DEFAULTS)
        original = self.destination.read_bytes()

        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            _ = settings_module.save_settings(changed(subagents=False))

        self.assertEqual(self.destination.read_bytes(), original)


ESC = asyncio.CancelledError()
TOP = [("subagents", "Subagents           ›"), ("search", "Web search          ›")]


def top(fetch: str = "on") -> list[tuple[str, str]]:
    return [*TOP, ("fetch", f"Web fetch           {fetch}")]


class SettingsCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = VinApp()
        self.inline = AsyncMock()
        self.save = Mock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.enterContext(patch.object(settings_menu_module, "save_settings", self.save))

    def run_menu(self, *steps: object) -> None:
        self.inline.side_effect = list(steps)
        self.assertIsNone(asyncio.run(self.app.change_settings()))
        self.assertEqual(self.inline.await_count, len(steps), "every step should be a menu the user saw")

    def test_settings_opens_a_menu_per_topic_and_esc_steps_back(self) -> None:
        self.run_menu("subagents", ESC, ESC)

        self.assertEqual(self.inline.await_args_list, [
            call("Settings", top(), default="subagents"),
            call("Subagents", [("enabled", "Subagents           on"), ("budget", "Request limit       8 ›")],
                 default="enabled"),
            call("Settings", top(), default="subagents"),
        ])
        self.save.assert_not_called()

    def test_a_change_stays_in_the_menu_showing_the_new_value(self) -> None:
        self.run_menu("subagents", "enabled", "enabled", ESC, ESC)

        menus = [args.args[1] for args in self.inline.await_args_list]
        self.assertEqual(menus[2][0], ("enabled", "Subagents           off"))
        self.assertEqual(menus[3][0], ("enabled", "Subagents           on"))
        self.assertEqual(self.save.call_args_list, [call(changed(subagents=False)), call(DEFAULTS)])
        self.assertEqual(self.app.settings, DEFAULTS)

    def test_web_fetch_row_toggles_in_place_without_touching_subagents(self) -> None:
        self.run_menu("fetch", "fetch", ESC)

        self.assertEqual([args.args[1] for args in self.inline.await_args_list], [top(), top("off"), top()])
        self.assertEqual([args.kwargs["default"] for args in self.inline.await_args_list],
                         ["subagents", "fetch", "fetch"])
        self.assertEqual(self.save.call_args_list, [call(changed(fetch=False)), call(DEFAULTS)])

    def test_changing_subagents_keeps_the_other_settings(self) -> None:
        start = changed(fetch=False, search=True)
        for steps, expected in ((["subagents", "enabled", ESC, ESC], {**start, "subagents": False}),
                                (["subagents", "budget", "5", ESC, ESC], {**start, "subagent_budget": 5})):
            with self.subTest(steps=steps):
                self.app.settings = start
                self.inline.reset_mock()
                self.run_menu(*steps)
                self.assertEqual(self.app.settings, expected)

    def test_request_limit_is_picked_from_one_to_thirty_two(self) -> None:
        self.run_menu("subagents", "budget", "3", ESC, ESC)

        self.assertEqual(self.inline.await_args_list[2], call(
            "Subagent request limit", [(str(number), str(number)) for number in range(1, 33)], default="8"))
        self.assertEqual(self.inline.await_args_list[3].args[1][1], ("budget", "Request limit       3 ›"))
        self.assertEqual(self.inline.await_args_list[3].kwargs["default"], "budget")
        self.assertEqual(self.app.settings, changed(subagent_budget=3))

    def test_escape_anywhere_changes_nothing(self) -> None:
        for steps in ([ESC], ["subagents", ESC, ESC], ["subagents", "budget", ESC, ESC, ESC]):
            with self.subTest(steps=steps):
                self.inline.reset_mock()
                self.run_menu(*steps)
                self.save.assert_not_called()
                self.assertEqual(self.app.settings, DEFAULTS)

    def test_ctrl_c_while_a_menu_is_open_is_not_swallowed(self) -> None:
        async def main() -> None:
            async def wait(*_args: object, **_kwargs: object) -> str:
                await asyncio.sleep(10)
                return "fetch"

            self.inline.side_effect = wait
            task = asyncio.create_task(self.app.change_settings())
            await asyncio.sleep(0)
            _ = task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        self.save.assert_not_called()

    def test_failed_save_keeps_the_current_settings_and_leaves_the_menu(self) -> None:
        self.inline.side_effect = ["subagents", "enabled"]
        self.save.side_effect = RuntimeError("Could not save settings.")

        with self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            asyncio.run(self.app.change_settings())
        self.assertEqual(self.app.settings, DEFAULTS)

    def test_bad_settings_file_shows_a_notice_and_keeps_defaults(self) -> None:
        self.app.credentials = fixture_credentials("token")
        with patch_app(create=True,
                       load_settings=Mock(side_effect=RuntimeError("Saved settings are invalid.")),
                       load_model_preference=Mock(return_value={"provider": "chatgpt", "model": "m"}),
                       load_cached_chatgpt_models=Mock(return_value=[])):
            asyncio.run(self.app.initialize_session())

        entry, = self.app.transcript
        self.assertEqual(str(entry["content"]), "Saved settings are invalid.")
        assert isinstance(entry["content"], Text)
        self.assertEqual(entry["content"].style, "red")
        self.assertEqual(self.app.settings, DEFAULTS)
        self.assertEqual(self.app.selected_model, "m")


KEY = "tvly-dev-secret-key"
ON = changed(search=True)


class SearchKeyFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "search.json"
        self.enterContext(patch.object(search, "SEARCH_KEY_PATH", self.path))

    def test_saved_key_loads_back_and_deletes(self) -> None:
        self.assertIsNone(search.load_search_key())
        _ = search.save_search_key(KEY)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"key": KEY})
        self.assertEqual(search.load_search_key(), KEY)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

        search.delete_search_key()
        self.assertFalse(self.path.exists())
        search.delete_search_key()

    def test_key_saved_with_the_older_switch_still_loads(self) -> None:
        _ = self.path.write_text(json.dumps({"key": KEY, "enabled": False}), encoding="utf-8")
        self.assertEqual(search.load_search_key(), KEY)

    def test_failed_save_and_delete_say_so(self) -> None:
        with patch("vin.auth.credentials.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "Could not save the web search key"):
                _ = search.save_search_key(KEY)
        with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "Could not remove the saved web search key"):
                search.delete_search_key()


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
        self.save_key = Mock()
        self.delete_key = Mock()
        self.save = Mock()
        self.enterContext(patch.object(self.app, "choose_inline", self.inline))
        self.enterContext(patch.object(self.app, "ask_secret", self.secret))
        self.enterContext(patch.object(self.app, "app", Mock(), create=True))
        self.enterContext(patch.multiple(settings_menu_module, check_search_key=self.check,
                                         save_search_key=self.save_key,
                                         delete_search_key=self.delete_key,
                                         save_settings=self.save))

    def run_menu(self, *steps: object) -> None:
        self.inline.side_effect = list(steps)
        self.assertIsNone(asyncio.run(self.app.change_settings()))

    def search_menus(self) -> list[list[tuple[str, str]]]:
        return [args.args[1] for args in self.inline.await_args_list if args.args[0] == "Web search"]

    def test_menu_shows_the_key_row_only_when_a_key_is_saved(self) -> None:
        self.run_menu("search", ESC, ESC)
        self.assertEqual(self.search_menus(), [[("enabled", "Web search          off")]])

        self.inline.reset_mock()
        self.app.search_key = KEY
        self.app.settings = ON
        self.run_menu("search", ESC, ESC)
        self.assertEqual(self.search_menus(),
                         [[("enabled", "Web search          on"), ("key", "Key                 tvly-dev-•••-key ›")]])

    def test_switch_saved_on_without_a_key_shows_off(self) -> None:
        self.app.settings = ON
        self.run_menu("search", ESC, ESC)
        self.assertEqual(self.search_menus(), [[("enabled", "Web search          off")]])

    def test_first_turn_on_asks_for_the_key_checks_it_and_shows_it_on(self) -> None:
        self.run_menu("search", "enabled", ESC, ESC)

        self.secret.assert_awaited_once_with("Paste your Tavily API key")
        self.check.assert_called_once_with(f" {KEY} ")
        self.save_key.assert_called_once_with(KEY)
        self.save.assert_called_once_with(ON)
        self.assertEqual((self.app.search_key, self.app.settings), (KEY, ON))
        self.assertEqual(self.search_menus()[1][0], ("enabled", "Web search          on"))

    def test_spinner_says_checking_while_the_key_is_checked_and_clears_after(self) -> None:
        seen: list[tuple[str | None, bool]] = []

        def check(key: str) -> str:
            seen.append((self.app.auth_status, self.app.loading_model))
            return key.strip()

        for loading in (False, True):
            with self.subTest(loading=loading):
                self.app.search_key = None
                self.app.loading_model = loading
                self.check.side_effect = check
                self.run_menu("search", "enabled", ESC, ESC)
                self.assertEqual(seen[-1], ("Checking Tavily key…", True))
                self.assertEqual((self.app.auth_status, self.app.loading_model), (None, loading))

    def test_rejected_key_saves_nothing_stays_off_and_clears_the_spinner(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.check.side_effect = RuntimeError("Tavily key was rejected.")

        with self.assertRaisesRegex(RuntimeError, "Tavily key was rejected"):
            asyncio.run(self.app.change_settings())
        self.save_key.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual((self.app.search_key, self.app.settings), (None, DEFAULTS))
        self.assertEqual((self.app.auth_status, self.app.loading_model), (None, False))

    def test_escape_at_the_key_prompt_returns_to_the_menu_and_saves_nothing(self) -> None:
        self.secret.side_effect = ESC
        self.run_menu("search", "enabled", ESC, ESC)

        self.check.assert_not_called()
        self.save_key.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(len(self.search_menus()), 2)
        self.assertEqual((self.app.search_key, self.app.settings), (None, DEFAULTS))

    def test_with_a_saved_key_the_switch_flips_without_asking(self) -> None:
        self.app.search_key = KEY
        self.app.settings = ON
        self.run_menu("search", "enabled", "enabled", ESC, ESC)

        self.assertEqual(self.app.settings, ON)
        self.secret.assert_not_awaited()
        self.save_key.assert_not_called()
        self.assertEqual(self.save.call_args_list, [call(DEFAULTS), call(ON)])
        self.assertEqual([menu[0][1] for menu in self.search_menus()],
                         ["Web search          on", "Web search          off", "Web search          on"])

    def test_failed_save_keeps_the_current_setting(self) -> None:
        self.inline.side_effect = ["search", "enabled"]
        self.app.search_key = KEY
        self.save.side_effect = RuntimeError("Could not save settings.")

        with self.assertRaisesRegex(RuntimeError, "Could not save settings."):
            asyncio.run(self.app.change_settings())
        self.assertEqual(self.app.settings, DEFAULTS)

    def test_replacing_the_key_keeps_the_switch_and_shows_the_new_key(self) -> None:
        self.app.search_key = KEY
        self.secret.return_value = "tvly-dev-another-key"
        self.run_menu("search", "key", "replace", ESC, ESC)

        self.assertEqual(self.inline.await_args_list[2],
                         call("Web search key", [("replace", "Replace key"), ("remove", "Remove key")], default=None))
        self.save_key.assert_called_once_with("tvly-dev-another-key")
        self.save.assert_not_called()
        self.assertEqual((self.app.search_key, self.app.settings), ("tvly-dev-another-key", DEFAULTS))
        self.assertEqual(self.search_menus()[1][1], ("key", "Key                 tvly-dev-•••-key ›"))

    def test_removing_the_key_turns_web_search_off_and_hides_the_key_row(self) -> None:
        self.app.search_key = KEY
        self.app.settings = ON
        self.run_menu("search", "key", "remove", ESC, ESC)

        self.delete_key.assert_called_once_with()
        self.save.assert_called_once_with(DEFAULTS)
        self.assertEqual((self.app.search_key, self.app.settings), (None, DEFAULTS))
        self.assertEqual(self.search_menus()[1], [("enabled", "Web search          off")])

    def test_escape_in_the_key_menu_or_prompt_changes_nothing(self) -> None:
        for steps, secret in (
            (["search", ESC, ESC], None),
            (["search", "key", ESC, ESC, ESC], None),
            (["search", "key", "replace", ESC, ESC], ESC),
        ):
            with self.subTest(steps=steps):
                self.inline.reset_mock()
                self.secret.side_effect = secret
                self.app.search_key = KEY
                self.app.settings = ON
                self.run_menu(*steps)
                self.save_key.assert_not_called()
                self.save.assert_not_called()
                self.delete_key.assert_not_called()
                self.assertEqual((self.app.search_key, self.app.settings), (KEY, ON))

    def test_saved_key_loads_at_startup_and_a_bad_file_shows_a_notice(self) -> None:
        self.app.credentials = fixture_credentials("token")
        common = {
            "load_settings": Mock(return_value=ON),
            "load_model_preference": Mock(return_value={"provider": "chatgpt", "model": "m"}),
            "load_cached_chatgpt_models": Mock(return_value=[]),
            "load_openrouter_account": Mock(return_value=None),
        }
        with patch_app(create=True, load_search_key=Mock(return_value=KEY), **common):
            asyncio.run(self.app.initialize_session())
        self.assertEqual((self.app.search_key, self.app.settings), (KEY, ON))
        self.assertEqual(self.app.transcript, [])

        app = VinApp()
        app.credentials = fixture_credentials("token")
        bad = Mock(side_effect=RuntimeError("Saved web search key is invalid."))
        with patch_app(create=True, load_search_key=bad, **common):
            asyncio.run(app.initialize_session())
        entry, = app.transcript
        self.assertEqual(str(entry["content"]), "Saved web search key is invalid.")
        self.assertIsNone(app.search_key)


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
