"""OpenRouter model list: catalog, saved whitelist, /model add, /model remove and the combined picker."""

import asyncio
import io
import json
import stat
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.error import HTTPError, URLError

from prompt_toolkit.completion import CompleteEvent, WordCompleter
from prompt_toolkit.document import Document

from rich.text import Text

from vin.auth.chatgpt import TokenExchangeError
from vin.provider import openrouter
from vin.tui import app as app_module
from vin.tui import model_menu as model_menu_module
from vin.tui.app import VinApp
from vin.tui.completion import SessionCompleter
from vin.tui.constants import is_command
from vin.types.auth import OpenRouterAccount
from vin.types.models import ModelMetadata
from tui_application_helpers import fixture_credentials

ACCOUNT: OpenRouterAccount = {"key": "sk-or-v1-secret", "label": "laptop"}
GPT: ModelMetadata = {"slug": "gpt", "display_name": "GPT", "provider": "chatgpt"}
CODER: ModelMetadata = {"slug": "acme/coder", "display_name": "Acme: Coder", "provider": "openrouter", "context_window": 1000}
OPUS: ModelMetadata = {"slug": "anthropic/claude-opus", "display_name": "Anthropic: Claude Opus", "provider": "openrouter"}
GEMINI: ModelMetadata = {"slug": "google/gemini", "display_name": "Google: Gemini", "provider": "openrouter"}


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def catalog_body(*entries: object) -> Response:
    return Response(json.dumps({"data": list(entries)}).encode())


class CatalogTests(unittest.TestCase):
    def test_only_models_with_tools_are_listed_with_name_and_context(self) -> None:
        body = catalog_body(
            {"id": "acme/coder", "name": "Acme: Coder", "context_length": 1000, "supported_parameters": ["tools"]},
            {"id": "acme/chat", "name": "Acme: Chat", "supported_parameters": ["temperature"]},
            {"id": "acme/plain", "name": "Plain"},
            {"id": "acme/unnamed", "name": " ", "context_length": 0, "supported_parameters": ["tools"]},
            {"id": "", "supported_parameters": ["tools"]},
            {"name": "No id", "supported_parameters": ["tools"]},
            "not a record",
        )
        with patch.object(openrouter, "urlopen", return_value=body) as urlopen:
            models = openrouter.list_openrouter_models()
        self.assertEqual(urlopen.call_args.args[0], "https://openrouter.ai/api/v1/models")
        self.assertEqual(models, [
            {**CODER, "features": []},
            {"slug": "acme/unnamed", "display_name": "acme/unnamed", "provider": "openrouter", "features": []}])

    def test_prices_are_per_million_and_features_skip_text(self) -> None:
        body = catalog_body(
            {"id": "a/paid", "name": "Paid", "supported_parameters": ["tools", "reasoning"],
             "pricing": {"prompt": "0.000003", "completion": "0.000015"},
             "architecture": {"input_modalities": ["text", "image", "file"]}},
            {"id": "a/free", "name": "Free", "supported_parameters": ["tools"],
             "pricing": {"prompt": "0", "completion": 0}},
            {"id": "a/router", "name": "Router", "supported_parameters": ["tools"],
             "pricing": {"prompt": "-1", "completion": "-1"}, "architecture": {"input_modalities": "text"}},
            {"id": "a/bad", "name": "Bad", "supported_parameters": ["tools"],
             "pricing": {"prompt": "abc", "completion": True}, "architecture": []},
            {"id": "a/half", "name": "Half", "supported_parameters": ["tools"], "pricing": {"prompt": "0.000001"}},
        )
        with patch.object(openrouter, "urlopen", return_value=body):
            models = {model["slug"]: model for model in openrouter.list_openrouter_models()}
        paid = models["a/paid"]
        self.assertEqual((paid.get("input_price"), paid.get("output_price")), (3.0, 15.0))
        self.assertEqual(paid.get("features"), ["image", "file", "reasoning"])
        self.assertEqual((models["a/free"].get("input_price"), models["a/free"].get("output_price")), (0.0, 0.0))
        for slug in ("a/router", "a/bad", "a/half"):
            with self.subTest(slug=slug):
                self.assertNotIn("input_price", models[slug])
                self.assertNotIn("output_price", models[slug])
                self.assertEqual(models[slug].get("features"), [])

    def test_models_are_sorted_newest_first(self) -> None:
        body = catalog_body(
            {"id": "a/old", "created": 100, "supported_parameters": ["tools"]},
            {"id": "a/undated", "supported_parameters": ["tools"]},
            {"id": "a/new", "created": 300, "supported_parameters": ["tools"]},
            {"id": "a/mid", "created": 200, "supported_parameters": ["tools"]},
            {"id": "a/mid-twin", "created": 200, "supported_parameters": ["tools"]},
        )
        with patch.object(openrouter, "urlopen", return_value=body):
            slugs = [model["slug"] for model in openrouter.list_openrouter_models()]
        self.assertEqual(slugs, ["a/new", "a/mid", "a/mid-twin", "a/old", "a/undated"])

    def test_failures_have_clear_messages(self) -> None:
        cases: list[tuple[object, str]] = [
            (HTTPError(openrouter.CATALOG_URL, 500, "error", Message(), io.BytesIO(b"{}")),
             "Could not load OpenRouter models (HTTP 500)."),
            (URLError("offline"), "Could not connect to OpenRouter."),
            (TimeoutError(), "Could not connect to OpenRouter."),
            (Response(b"not json"), "OpenRouter returned an invalid model list."),
            (Response(b'{"data": {}}'), "OpenRouter returned an invalid model list."),
            (Response(b"[]"), "OpenRouter returned an invalid model list."),
        ]
        for outcome, message in cases:
            mock = Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=outcome)
            with self.subTest(message=message, outcome=type(outcome).__name__), patch.object(openrouter, "urlopen", mock):
                with self.assertRaises(RuntimeError) as raised:
                    openrouter.list_openrouter_models()
                self.assertEqual(str(raised.exception), message)


class WhitelistFileTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "vin" / "openrouter_models.json"
        self.enterContext(patch.object(openrouter, "WHITELIST_PATH", self.path))

    def test_missing_file_is_an_empty_list(self) -> None:
        self.assertEqual(openrouter.load_openrouter_whitelist(), [])

    def test_saved_list_round_trips_privately(self) -> None:
        openrouter.save_openrouter_whitelist([CODER, OPUS])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(openrouter.load_openrouter_whitelist(), [CODER, OPUS])

    def test_prices_and_features_round_trip_and_bad_values_are_dropped(self) -> None:
        priced: ModelMetadata = {**OPUS, "input_price": 5.0, "output_price": 0, "features": ["image"]}
        openrouter.save_openrouter_whitelist([priced])
        self.assertEqual(openrouter.load_openrouter_whitelist(), [priced])
        self.path.write_text(json.dumps([{**OPUS, "input_price": True, "output_price": -1, "features": [1]},
                                         {**CODER, "input_price": "5"}]))
        self.assertEqual(openrouter.load_openrouter_whitelist(), [OPUS, CODER])

    def test_loaded_models_are_always_openrouter_models(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps([{"slug": "acme/coder", "display_name": "Acme: Coder", "provider": "chatgpt"}]))
        self.assertEqual(openrouter.load_openrouter_whitelist()[0]["provider"], "openrouter")

    def test_unreadable_or_invalid_files_raise_clear_errors(self) -> None:
        self.path.parent.mkdir(parents=True)
        for content, message in (("not json", "Could not read saved OpenRouter models."),
                                 ('{"slug": "x"}', "Saved OpenRouter models are invalid."),
                                 ('[{"slug": "x"}]', "Saved OpenRouter models are invalid.")):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(RuntimeError) as raised:
                    openrouter.load_openrouter_whitelist()
                self.assertEqual(str(raised.exception), message)

    def test_save_failure_has_a_clear_message(self) -> None:
        with patch.object(openrouter, "save_private_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, r"^Could not save OpenRouter models\.$"):
                openrouter.save_openrouter_whitelist([CODER])


class ModelCommandMethodTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.vin = VinApp()
        self.vin.app = Mock()
        self.vin.openrouter = ACCOUNT
        self.vin.models = [GPT, CODER]
        self.vin.selected_provider, self.vin.selected_model = "chatgpt", "gpt"
        self.vin.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        self.pick = self.enterContext(patch.object(self.vin, "pick_models", AsyncMock()))
        self.catalog = self.enterContext(patch.object(model_menu_module, "list_openrouter_models", return_value=[CODER, OPUS, GEMINI]))
        self.save_list = self.enterContext(patch.object(model_menu_module, "save_openrouter_whitelist"))
        self.save_choice = self.enterContext(patch.object(model_menu_module, "save_model_preference"))

    def assert_selection(self, provider: str | None, model: str | None) -> None:
        self.assertEqual((self.vin.selected_provider, self.vin.selected_model), (provider, model))

    # /model

    async def test_picker_lists_every_provider_and_switches_provider(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.pick.return_value = [CODER]
        self.assertEqual(await self.vin.change_model(), "Model: OpenRouter · Acme: Coder")
        self.pick.assert_awaited_once_with(
            "Choose a model", [GPT, CODER], multi=False, action="choose", current=("chatgpt", "gpt"))
        self.save_choice.assert_called_once_with("openrouter", "acme/coder")
        self.assert_selection("openrouter", "acme/coder")
        self.assertIsNone(self.vin.last_usage)

    async def test_picking_the_current_model_changes_nothing(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.pick.return_value = [GPT]
        self.assertIsNone(await self.vin.change_model())
        self.save_choice.assert_not_called()
        self.assertIsNotNone(self.vin.last_usage)

    async def test_openrouter_only_lists_its_models_without_chatgpt_discovery(self) -> None:
        self.vin.models = [CODER]
        self.vin.selected_provider, self.vin.selected_model = None, None
        self.pick.return_value = [CODER]
        with patch.object(self.vin, "discover_chatgpt_models", AsyncMock()) as discover:
            await self.vin.change_model()
        discover.assert_not_awaited()
        self.assertEqual(self.pick.await_args_list[0].args[1], [CODER])
        self.assertEqual(self.pick.await_args_list[0].kwargs["current"], ("", ""))

    async def test_failed_chatgpt_sign_in_still_lists_the_openrouter_models(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.models = [CODER]
        self.vin.selected_provider, self.vin.selected_model = None, None
        failure = TokenExchangeError(401, "invalid_grant")
        for picked, expected in (
            ([CODER], f"ChatGPT models are unavailable: {failure}\nModel: OpenRouter · Acme: Coder"),
            (None, f"ChatGPT models are unavailable: {failure}"),
        ):
            with self.subTest(picked=picked):
                self.pick.reset_mock()
                self.pick.return_value = picked
                with patch.object(self.vin, "discover_chatgpt_models", AsyncMock(side_effect=failure)):
                    outcome = await self.vin.change_model()
                assert isinstance(outcome, Text)
                self.assertEqual(str(outcome), expected)
                self.assertIn("red", [outcome.style, *(span.style for span in outcome.spans)])
                self.assertEqual(self.pick.await_args_list[0].args[1], [CODER])
                self.assertFalse(self.vin.loading_model)
        self.assert_selection("openrouter", "acme/coder")

    async def test_failed_chatgpt_discovery_with_no_other_models_is_an_error(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.vin.openrouter = None
        self.vin.models = []
        with patch.object(self.vin, "discover_chatgpt_models", AsyncMock(side_effect=RuntimeError("offline"))):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await self.vin.change_model()
        self.pick.assert_not_awaited()
        self.assertFalse(self.vin.loading_model)

    async def test_openrouter_with_an_empty_list_points_to_add(self) -> None:
        self.vin.models = []
        self.assertEqual(await self.vin.change_model(), "No models yet. Use /model add to add an OpenRouter model.")
        self.pick.assert_not_awaited()

    async def test_signed_out_of_both_cannot_choose(self) -> None:
        self.vin.openrouter = None
        self.assertEqual(await self.vin.change_model(), "Sign in with /login before choosing a model.")

    async def test_cancelled_picker_changes_nothing(self) -> None:
        self.vin.credentials = fixture_credentials()
        self.pick.return_value = None
        self.assertIsNone(await self.vin.change_model())
        self.save_choice.assert_not_called()
        self.assert_selection("chatgpt", "gpt")

    # /model add

    async def test_add_saves_the_models_without_selecting_them(self) -> None:
        self.pick.return_value = [OPUS, GEMINI]
        self.assertEqual(await self.vin.add_openrouter_model(""),
                         "Added OpenRouter · Anthropic: Claude Opus, Google: Gemini. Use /model to switch.")
        self.pick.assert_awaited_once_with(
            "Add OpenRouter models", [OPUS, GEMINI], multi=True, action="add", search="")
        self.save_list.assert_called_once_with([CODER, OPUS, GEMINI])
        self.assertEqual(self.vin.models, [GPT, CODER, OPUS, GEMINI])
        self.assert_selection("chatgpt", "gpt")
        self.save_choice.assert_not_called()
        self.assertIsNotNone(self.vin.last_usage)
        self.assertFalse(self.vin.loading_model)
        self.assertIsNone(self.vin.pending_model_load)

    async def test_the_only_model_is_selected_after_adding(self) -> None:
        self.vin.models = []
        self.vin.selected_provider, self.vin.selected_model = None, None
        self.pick.return_value = [GEMINI]
        self.assertEqual(await self.vin.add_openrouter_model(""),
                         "Added OpenRouter · Google: Gemini. Model: OpenRouter · Google: Gemini")
        self.save_choice.assert_called_once_with("openrouter", "google/gemini")
        self.assert_selection("openrouter", "google/gemini")

    async def test_adding_several_models_with_nothing_selected_selects_none(self) -> None:
        self.vin.models = []
        self.vin.selected_provider, self.vin.selected_model = None, None
        self.pick.return_value = [OPUS, GEMINI]
        self.assertEqual(await self.vin.add_openrouter_model(""),
                         "Added OpenRouter · Anthropic: Claude Opus, Google: Gemini. Use /model to switch.")
        self.save_choice.assert_not_called()
        self.assert_selection(None, None)

    async def test_add_refreshes_saved_models_from_the_catalog_even_when_cancelled(self) -> None:
        priced: ModelMetadata = {**CODER, "input_price": 1.0, "output_price": 2.0, "features": ["image"]}
        gone: ModelMetadata = {"slug": "old/gone", "display_name": "Gone", "provider": "openrouter"}
        self.vin.models = [GPT, CODER, gone]
        self.catalog.return_value = [priced, OPUS]
        self.pick.return_value = None
        self.assertIsNone(await self.vin.add_openrouter_model(""))
        self.save_list.assert_called_once_with([priced, gone])
        self.assertEqual(self.vin.models, [GPT, priced, gone])
        self.assertEqual(self.pick.await_args_list[0].args[1], [OPUS])

    async def test_add_keeps_the_refreshed_entries_when_adding(self) -> None:
        priced: ModelMetadata = {**CODER, "input_price": 1.0, "output_price": 2.0}
        self.catalog.return_value = [priced, OPUS]
        self.pick.return_value = [OPUS]
        await self.vin.add_openrouter_model("")
        self.assertEqual(self.save_list.call_args_list[-1].args[0], [priced, OPUS])

    async def test_an_up_to_date_list_is_not_saved_again(self) -> None:
        self.pick.return_value = None
        await self.vin.add_openrouter_model("")
        self.save_list.assert_not_called()

    async def test_search_text_opens_the_picker_already_filtered(self) -> None:
        self.pick.return_value = [OPUS]
        await self.vin.add_openrouter_model("claude")
        self.assertEqual(self.pick.await_args_list[0].kwargs["search"], "claude")

    async def test_nothing_to_add_explains_why(self) -> None:
        self.vin.models = [GPT, CODER, OPUS, GEMINI]
        self.assertEqual(await self.vin.add_openrouter_model(""),
                         "Every OpenRouter model with tool support is already added.")
        self.pick.assert_not_awaited()
        self.save_list.assert_not_called()

    async def test_add_needs_an_openrouter_sign_in(self) -> None:
        self.vin.openrouter = None
        self.assertEqual(await self.vin.add_openrouter_model(""),
                         "Sign in to OpenRouter with /login before adding a model.")
        self.catalog.assert_not_called()

    async def test_cancelled_add_saves_nothing(self) -> None:
        self.pick.return_value = None
        self.assertIsNone(await self.vin.add_openrouter_model(""))
        self.save_list.assert_not_called()
        self.assertEqual(self.vin.models, [GPT, CODER])

    async def test_catalog_failure_is_raised_and_clears_loading(self) -> None:
        self.catalog.side_effect = RuntimeError("Could not connect to OpenRouter.")
        with self.assertRaisesRegex(RuntimeError, "Could not connect to OpenRouter."):
            await self.vin.add_openrouter_model("")
        self.assertFalse(self.vin.loading_model)
        self.assertIsNone(self.vin.pending_model_load)

    async def test_escape_while_the_catalog_loads_cancels_quietly(self) -> None:
        started = asyncio.Event()

        async def slow(*_args: object) -> list[ModelMetadata]:
            started.set()
            await asyncio.sleep(10)
            return []
        with patch.object(app_module.asyncio, "to_thread", slow):
            adding = asyncio.create_task(self.vin.add_openrouter_model(""))
            await started.wait()
            assert self.vin.pending_model_load is not None
            self.vin.pending_model_load.cancel()
            self.assertIsNone(await adding)
        self.assertFalse(self.vin.loading_model)
        self.pick.assert_not_awaited()

    async def test_failed_list_save_keeps_the_list(self) -> None:
        self.pick.return_value = [GEMINI]
        self.save_list.side_effect = RuntimeError("Could not save OpenRouter models.")
        with self.assertRaisesRegex(RuntimeError, "Could not save OpenRouter models."):
            await self.vin.add_openrouter_model("")
        self.assertEqual(self.vin.models, [GPT, CODER])

    # /model remove

    async def test_removing_other_models_keeps_the_selection(self) -> None:
        self.vin.models = [GPT, CODER, OPUS, GEMINI]
        self.pick.return_value = [CODER, GEMINI]
        self.assertEqual(await self.vin.remove_openrouter_model(), "Removed OpenRouter · Acme: Coder, Google: Gemini.")
        self.pick.assert_awaited_once_with(
            "Remove OpenRouter models", [CODER, OPUS, GEMINI], multi=True, action="remove")
        self.save_list.assert_called_once_with([OPUS])
        self.assertEqual(self.vin.models, [GPT, OPUS])
        self.assert_selection("chatgpt", "gpt")

    async def test_removing_the_selected_model_asks_for_another(self) -> None:
        self.vin.models = [GPT, CODER, OPUS]
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"
        self.pick.return_value = [CODER]
        with patch.object(self.vin, "change_model", AsyncMock(return_value="Model: ChatGPT · GPT")) as change:
            self.assertEqual(await self.vin.remove_openrouter_model(),
                             "Removed OpenRouter · Acme: Coder. Model: ChatGPT · GPT")
        change.assert_awaited_once_with()
        self.assertIsNone(self.vin.last_usage)

    async def test_removing_the_selected_model_with_one_left_selects_it(self) -> None:
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"
        self.vin.models = [CODER, OPUS]
        self.pick.return_value = [CODER]
        with patch.object(self.vin, "change_model", AsyncMock()) as change:
            self.assertEqual(await self.vin.remove_openrouter_model(),
                             "Removed OpenRouter · Acme: Coder. Model: OpenRouter · Anthropic: Claude Opus")
        change.assert_not_awaited()
        self.assert_selection("openrouter", "anthropic/claude-opus")

    async def test_removing_the_selected_model_and_cancelling_leaves_none(self) -> None:
        self.vin.models = [GPT, CODER, OPUS]
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"
        self.pick.return_value = [CODER]
        with patch.object(self.vin, "change_model", AsyncMock(return_value=None)):
            self.assertEqual(await self.vin.remove_openrouter_model(), "Removed OpenRouter · Acme: Coder.")
        self.assert_selection("openrouter", None)

    async def test_removing_the_last_model_leaves_no_model_and_no_picker(self) -> None:
        self.vin.models = [CODER]
        self.vin.selected_provider, self.vin.selected_model = "openrouter", "acme/coder"
        self.pick.return_value = [CODER]
        with patch.object(self.vin, "change_model", AsyncMock()) as change:
            self.assertEqual(await self.vin.remove_openrouter_model(), "Removed OpenRouter · Acme: Coder.")
        change.assert_not_awaited()
        self.assertEqual(self.vin.models, [])
        self.assert_selection("openrouter", None)

    async def test_nothing_to_remove(self) -> None:
        self.vin.models = [GPT]
        self.assertEqual(await self.vin.remove_openrouter_model(), "No OpenRouter models to remove.")
        self.pick.assert_not_awaited()

    async def test_cancelled_remove_changes_nothing(self) -> None:
        self.pick.return_value = None
        self.assertIsNone(await self.vin.remove_openrouter_model())
        self.save_list.assert_not_called()
        self.assertEqual(self.vin.models, [GPT, CODER])


class ModelSubcommandCompletionTests(unittest.TestCase):
    def suggestions(self, text: str) -> list[str]:
        completer = SessionCompleter(".", WordCompleter(["/model", "/help"], sentence=True))
        return [item.text for item in completer.get_completions(Document(text), CompleteEvent(text_inserted=True))]

    def test_model_offers_add_and_remove(self) -> None:
        self.assertEqual(self.suggestions("/model "), ["add", "remove"])
        self.assertEqual(self.suggestions("/model a"), ["add"])
        self.assertEqual(self.suggestions("/model re"), ["remove"])
        self.assertEqual(self.suggestions("/model add "), [])
        self.assertEqual(self.suggestions("/model x"), [])

    def test_bare_command_still_completes_command_names(self) -> None:
        self.assertEqual(self.suggestions("/mo"), ["/model"])
        self.assertEqual(self.suggestions("/model"), ["/model"])

    def test_model_with_arguments_is_a_command(self) -> None:
        for text in ("/model", "/model add", "/model add claude", "/model remove"):
            with self.subTest(text=text):
                self.assertTrue(is_command(text))


if __name__ == "__main__":
    unittest.main()
