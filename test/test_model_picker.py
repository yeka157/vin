"""Searchable model picker: search, move, tick, render, and the pick_models popup."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import BufferControl
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.layout import Window, walk
from prompt_toolkit.widgets.base import Border
from prompt_toolkit.output import DummyOutput

from vin.tui.app import VinApp
from vin.tui.bindings import create_bindings
from vin.tui.model_picker import ModelPicker, context_label, price_labels
from vin.types.models import ModelMetadata

OPUS: ModelMetadata = {"slug": "anthropic/claude-opus", "display_name": "Anthropic: Claude Opus",
                       "provider": "openrouter", "context_window": 1_048_576,
                       "input_price": 5.0, "output_price": 25.0, "features": ["image", "reasoning"]}
SONNET: ModelMetadata = {"slug": "anthropic/claude-sonnet", "display_name": "Anthropic: Claude Sonnet",
                         "provider": "openrouter", "context_window": 200_000}
FREE: ModelMetadata = {"slug": "acme/free", "display_name": "Acme: Free", "provider": "openrouter",
                       "input_price": 0, "output_price": 0}
GPT: ModelMetadata = {"slug": "gpt", "display_name": "GPT", "provider": "chatgpt"}


@contextmanager
def screen(rows: int = 30, columns: int = 100) -> Generator[None]:
    def size(_self: DummyOutput) -> Size:
        return Size(rows=rows, columns=columns)

    with patch.object(DummyOutput, "get_size", size), \
            set_app(Application(output=DummyOutput())):
        yield


def text(picker: ModelPicker) -> str:
    return "".join(fragment[1] for fragment in picker.render())


class LabelTests(unittest.TestCase):
    def test_context_is_short(self) -> None:
        cases: list[tuple[ModelMetadata, str]] = [
            (OPUS, "1M"), (SONNET, "200K"), (FREE, ""),
            ({**FREE, "context_window": 2_500_000}, "2.5M"), ({**FREE, "context_window": 131_072}, "131K")]
        for model, label in cases:
            with self.subTest(label=label):
                self.assertEqual(context_label(model), label)

    def test_price_is_per_million_in_and_out(self) -> None:
        self.assertEqual(price_labels(OPUS), ("5.00", "25.00"))
        self.assertEqual(price_labels(FREE), ("free", "free"))
        self.assertEqual(price_labels(SONNET), ("", ""))
        self.assertEqual(price_labels({**FREE, "input_price": 1.0}), ("1.00", "0.00"))


class PickerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(screen())

    def press(self, picker: ModelPicker, key: str | Keys, data: str = "") -> None:
        control = picker.search_window.content
        assert isinstance(control, BufferControl) and control.key_bindings is not None
        handlers = [b for b in control.key_bindings.get_bindings_for_keys((key,)) if b.filter()]
        assert handlers, f"No active binding for {key!r}"
        event = cast(KeyPressEvent, Mock(data=data or str(key), app=Mock()))
        _ = handlers[-1].handler(event)

    def test_search_matches_every_word_in_name_or_slug_ignoring_case(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add")
        for search, expected in (("", [OPUS, SONNET, FREE]), ("CLAUDE opus", [OPUS]),
                                 ("anthropic/", [OPUS, SONNET]), ("acme: free", [FREE]), ("nope", [])):
            with self.subTest(search=search):
                picker.search.text = search
                self.assertEqual(picker.matches(), expected)

    def test_initial_search_text_is_kept(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add", search="sonnet")
        self.assertEqual(picker.matches(), [SONNET])
        self.assertEqual(picker.search.cursor_position, len("sonnet"))

    def test_enter_takes_the_first_match_from_search_or_the_highlighted_row(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add")
        self.assertIsNone(picker.row)
        self.assertEqual(picker.chosen(), [OPUS])
        self.press(picker, "down")
        self.press(picker, "down")
        self.assertEqual(picker.chosen(), [SONNET])
        picker.search.text = "nope"
        self.assertEqual(picker.chosen(), [])

    def test_arrows_move_between_search_and_list_without_leaving_it(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET], multi=True, action="add")
        self.press(picker, "up")
        self.assertIsNone(picker.row)
        for expected in (0, 1, 1):
            self.press(picker, "down")
            self.assertEqual(picker.row, expected)
        self.press(picker, "up")
        self.press(picker, "up")
        self.assertIsNone(picker.row)
        picker.search.text = "nope"
        self.press(picker, "down")
        self.assertIsNone(picker.row)

    def test_space_ticks_in_the_list_and_types_in_search(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add")
        control = picker.search_window.content
        assert isinstance(control, BufferControl) and control.key_bindings is not None
        space = [b for b in control.key_bindings.get_bindings_for_keys((" ",)) if b.filter()]
        self.assertFalse(any(b.keys == (" ",) for b in space), "Space must type while searching")
        self.press(picker, "down")
        self.press(picker, " ")
        self.press(picker, "down")
        self.press(picker, "down")
        self.press(picker, " ")
        self.assertEqual(picker.chosen(), [OPUS, FREE])
        self.assertEqual(picker.header(), " Add 2 selected")
        self.press(picker, "up")
        self.press(picker, "up")
        self.press(picker, " ")
        self.assertEqual(picker.chosen(), [FREE])

    def test_ticks_survive_a_new_search(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add")
        self.press(picker, "down")
        self.press(picker, " ")
        picker.search.text = "free"
        self.assertEqual(picker.chosen(), [OPUS])

    def test_typing_in_the_list_goes_back_to_search(self) -> None:
        picker = ModelPicker("Add", [OPUS, SONNET, FREE], multi=True, action="add")
        self.press(picker, "down")
        self.press(picker, Keys.Any, "f")
        self.assertIsNone(picker.row)
        self.assertEqual(picker.search.text, "f")
        self.press(picker, "down")
        self.press(picker, Keys.Any, "\x7f")
        self.assertIsNone(picker.row)
        self.assertEqual(picker.search.text, "f")

    def test_single_picker_has_no_ticks_and_starts_on_the_current_model(self) -> None:
        picker = ModelPicker("Choose", [GPT, OPUS], multi=False, action="choose", current=("openrouter", OPUS["slug"]))
        self.assertEqual(picker.row, 1)
        self.assertEqual(picker.chosen(), [OPUS])
        self.press(picker, Keys.Any, " ")
        self.assertEqual(picker.search.text, " ")
        self.assertNotIn("Space", picker.footer())
        self.assertEqual(ModelPicker("Choose", [GPT], multi=False, action="choose", current=("", "")).row, None)

    def test_rows_show_name_context_price_and_input(self) -> None:
        picker = ModelPicker("Add", [OPUS, FREE], multi=True, action="add")
        heading, top_rule, opus, free, bottom_rule = text(picker).splitlines()
        self.assertEqual(heading.split(), ["Model", "Context", "In", "$/M", "Out", "$/M", "Also", "accepts"])
        self.assertEqual(set(top_rule.strip()), {"─"})
        self.assertEqual(top_rule, bottom_rule)
        self.assertEqual(opus.split(), ["○", "Anthropic:", "Claude", "Opus", "1M", "5.00", "25.00",
                                            "image,", "reasoning"])
        self.assertTrue(opus.rstrip().endswith("  image, reasoning"))
        self.assertEqual(free.split(), ["○", "Acme:", "Free", "free", "free"])
        self.assertEqual(heading.index("Model"), opus.index("Anthropic"))
        self.assertEqual(heading.index("Context") + len("Context"), opus.index("1M") + len("1M"))
        self.assertEqual(heading.index("In $/M") + len("In $/M"), opus.index("5.00") + len("5.00"))
        self.assertEqual(heading.index("Out $/M") + len("Out $/M"), opus.index("25.00") + len("25.00"))
        self.assertEqual(heading.index("Also accepts"), opus.index("image"))
        self.assertEqual(picker.footer(), "   2 of 2 · ↑↓ move · Space select · Enter add")

    def test_highlighted_row_has_the_pointer_and_style(self) -> None:
        picker = ModelPicker("Add", [OPUS, FREE], multi=True, action="add")
        self.press(picker, "down")
        self.press(picker, " ")
        self.press(picker, "down")
        rows = picker.render()[2:-1]
        self.assertTrue(rows[0][1].startswith("  ◉ "))
        self.assertEqual(rows[0][0], "")
        self.assertTrue(rows[1][1].startswith("❯ ○ "))
        self.assertIn("bg:", rows[1][0])

    def test_mixed_providers_show_the_provider_and_the_current_mark(self) -> None:
        picker = ModelPicker("Choose", [GPT, OPUS], multi=False, action="choose", current=("chatgpt", "gpt"))
        lines = text(picker).splitlines()
        self.assertIn("✓ GPT · ChatGPT", lines[2])
        self.assertIn("Anthropic: Claude Opus · OpenRouter", lines[3])
        self.assertNotIn("·", ModelPicker("Add", [OPUS], multi=True, action="add").header())

    def test_long_names_and_lines_are_cut_to_the_screen(self) -> None:
        long: ModelMetadata = {**OPUS, "display_name": "X" * 200}
        with screen(columns=80):
            picker = ModelPicker("Add", [long], multi=True, action="add")
            line = text(picker).splitlines()[2]
            self.assertLessEqual(len(line), 79)
            self.assertIn("X…", line)

    def test_search_box_has_a_frame_and_a_hint_only_while_empty(self) -> None:
        picker = ModelPicker("Add", [OPUS], multi=True, action="add")
        framed = list(walk(picker.container.children[1]))
        self.assertIn(picker.search_window, framed)
        self.assertTrue(any(isinstance(w, Window) and w.char == Border.VERTICAL for w in framed))
        control = picker.search_window.content
        assert isinstance(control, BufferControl)
        hint = next(p for p in control.input_processors or [] if isinstance(p, ConditionalProcessor))
        assert isinstance(hint.processor, AfterInput)
        self.assertEqual(hint.processor.text, "Type a name or id, e.g. claude opus")
        self.assertTrue(hint.filter())
        picker.search.text = "c"
        self.assertFalse(hint.filter())

    def test_no_matches_says_so(self) -> None:
        picker = ModelPicker("Add", [OPUS], multi=True, action="add", search="nope")
        self.assertIn("No models match.", text(picker))
        self.assertTrue(picker.footer().startswith("   0 of 1 "))

    def test_list_scrolls_to_keep_the_highlighted_row_visible(self) -> None:
        models: list[ModelMetadata] = [{"slug": f"m/{i}", "display_name": f"Model {i}", "provider": "openrouter"}
                                       for i in range(50)]
        with screen(rows=24):
            picker = ModelPicker("Add", models, multi=True, action="add")
            self.assertEqual(picker.visible(), 6)
            for _ in range(10):
                self.press(picker, "down")
            shown = text(picker).splitlines()[2:-1]
            self.assertEqual(len(shown), 6)
            self.assertIn("❯ ○ Model 9", shown[-1])
            for _ in range(9):
                self.press(picker, "up")
            self.assertIn("❯ ○ Model 0", text(picker).splitlines()[2])
            picker.search.text = "Model 4"
            self.assertEqual(picker.top, 0)

    def test_tiny_screens_still_show_one_row(self) -> None:
        with screen(rows=9):
            self.assertEqual(ModelPicker("Add", [OPUS, FREE], multi=True, action="add").visible(), 1)


class PickModelsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(screen())
        self.vin = VinApp()
        self.vin.app = Mock()
        self.focus = Mock()
        self.vin.session = cast(PromptSession[str], Mock(layout=Mock(focus=self.focus)))
        self.vin.input_window = Mock()
        bindings = create_bindings(self.vin)
        self.handlers = {binding.handler.__name__: binding.handler for binding in bindings.bindings}

    def press(self, name: str) -> None:
        _ = self.handlers[name](cast(KeyPressEvent, Mock(current_buffer=Mock(), app=Mock())))

    async def open(self, models: list[ModelMetadata], **options: object) -> "asyncio.Task[list[ModelMetadata] | None]":
        task = asyncio.create_task(self.vin.pick_models("Add OpenRouter models", models, multi=True, action="add",
                                                        **options))  # pyright: ignore[reportArgumentType]
        async with asyncio.timeout(1):
            while self.vin.picker is None:
                await asyncio.sleep(0)
        return task

    def assert_closed(self) -> None:
        self.assertIsNone(self.vin.picker)
        self.assertIsNone(self.vin.choice_menu)
        self.assertIsNone(self.vin.choice_future)
        self.focus.assert_called_with(self.vin.input_window)

    async def test_enter_returns_the_ticked_models_and_closes(self) -> None:
        task = await self.open([OPUS, SONNET, FREE])
        picker = self.vin.picker
        assert picker is not None
        self.assertIs(self.vin.choice_menu, picker.container)
        self.focus.assert_called_with(picker.search_window)
        picker.row = 2
        picker.checked = {OPUS["slug"], FREE["slug"]}
        self.press("submit")
        self.assertEqual(await task, [OPUS, FREE])
        self.assert_closed()

    async def test_escape_returns_nothing_and_closes(self) -> None:
        task = await self.open([OPUS])
        self.press("cancel_or_exit")
        self.assertIsNone(await task)
        self.assert_closed()

    async def test_enter_with_no_match_keeps_the_picker_open(self) -> None:
        task = await self.open([OPUS], search="nope")
        self.press("submit")
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        assert self.vin.picker is not None
        self.vin.picker.search.text = ""
        self.press("submit")
        self.assertEqual(await task, [OPUS])

    async def test_a_second_picker_is_refused_while_one_is_open(self) -> None:
        task = await self.open([OPUS])
        with self.assertRaisesRegex(RuntimeError, "A choice is already open"):
            await self.vin.pick_models("Again", [OPUS], multi=False, action="choose")
        task.cancel()
        _ = await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
