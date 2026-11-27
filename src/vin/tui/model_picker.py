"""Searchable model list for adding, removing, and choosing models."""

from collections.abc import Callable, Sequence

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import (
    AfterInput,
    BeforeInput,
    ConditionalProcessor,
)
from prompt_toolkit.widgets import Frame

from vin.tui.constants import PROVIDER_LABELS
from vin.types.models import ModelMetadata


def context_label(model: ModelMetadata) -> str:
    n = model.get("context_window")
    if n is None:
        return ""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}".removesuffix(".0") + "M"
    return f"{round(n/1000)}K"


def price_labels(model: ModelMetadata) -> tuple[str, str]:
    input_price = model.get("input_price")
    output_price = model.get("output_price")
    if input_price is None or output_price is None:
        return "", ""
    if input_price == 0 and output_price == 0:
        return "free", "free"
    return f"{input_price:.2f}", f"{output_price:.2f}"


class ModelPicker:
    def __init__(
        self,
        title: str,
        models: list[ModelMetadata],
        *,
        multi: bool,
        action: str,
        current: tuple[str, str] | None = None,
        search: str = "",
        tabs: Sequence[str] = (),
        tab: int = 0,
        on_switch: Callable[[], None] | None = None,
    ) -> None:
        self.title = title
        self.tabs = tabs
        self.tab = tab
        # Set by Tab / Shift+Tab: the tab the caller should open next.
        self.switch_to: int | None = None
        self.models = models
        self.multi = multi
        self.action = action
        self.current = current
        self.checked: set[str] = set()
        self.top = 0
        self.show_provider = len({m["provider"] for m in models}) > 1
        self.search = Buffer(
            multiline=False,
            document=Document(search, len(search)),
            on_text_changed=self.reset,
        )
        self.row: int | None = next(
            (i for i, m in enumerate(models) if (m["provider"], m["slug"]) == current),
            None,
        )
        kb = KeyBindings()

        @kb.add("up")
        def _up(event: KeyPressEvent) -> None:
            if self.row == 0:
                self.row = None
            elif self.row is not None:
                self.row -= 1
            event.app.invalidate()

        @kb.add("down")
        def _down(event: KeyPressEvent) -> None:
            count = len(self.matches())
            if count:
                self.row = 0 if self.row is None else min(self.row + 1, count - 1)
            event.app.invalidate()

        @kb.add(" ", filter=Condition(lambda: self.multi and self.row is not None))
        def _toogle(event: KeyPressEvent) -> None:
            assert self.row is not None
            self.checked ^= {self.matches()[self.row]["slug"]}
            event.app.invalidate()

        @kb.add(Keys.Any, filter=Condition(lambda: self.row is not None))
        def _type(event: KeyPressEvent) -> None:
            self.row = None
            if event.data.isprintable():
                self.search.insert_text(event.data)
            event.app.invalidate()

        def switch(step: int) -> None:
            self.switch_to = (self.tab + step) % len(self.tabs)
            if on_switch is not None:
                on_switch()

        has_tabs = Condition(lambda: len(self.tabs) > 1)

        @kb.add("tab", filter=has_tabs, eager=True)
        def _next_tab(_event: KeyPressEvent) -> None:
            switch(1)

        @kb.add("s-tab", filter=has_tabs, eager=True)
        def _previous_tab(_event: KeyPressEvent) -> None:
            switch(-1)

        self.search_window = Window(
            BufferControl(
                self.search,
                input_processors=[
                    BeforeInput("Search  ", style="bold"),
                    ConditionalProcessor(
                        AfterInput(
                            "Type a name or id, e.g. claude opus",
                            style="class:placeholder",
                        ),
                        filter=Condition(lambda: not self.search.text),
                    ),
                ],
                key_bindings=kb,
            ),
            height=1,
            always_hide_cursor=Condition(lambda: self.row is not None),
        )
        self.container = HSplit(
            [
                Window(FormattedTextControl(self.title_line), height=1),
                Frame(self.search_window),
                Window(
                    FormattedTextControl(self.render),
                    height=lambda: Dimension.exact(self.visible() + 3),
                ),
                Window(
                    FormattedTextControl(self.footer),
                    height=1,
                    style="class:placeholder",
                ),
            ]
        )

    def reset(self, _buffer: Buffer) -> None:
        self.row = None
        self.top = 0

    def matches(self) -> list[ModelMetadata]:
        words = self.search.text.casefold().split()
        return [
            m
            for m in self.models
            if all(
                word in f"{m['slug']} {m['display_name']}".casefold() for word in words
            )
        ]

    def chosen(self) -> list[ModelMetadata]:
        if self.multi and len(self.checked) > 0:
            return [model for model in self.models if model["slug"] in self.checked]
        matches = self.matches()
        return [matches[self.row or 0]] if matches else []

    def visible(self) -> int:
        return max(1, min(len(self.models), get_app().output.get_size().rows - 18))

    def render(self) -> StyleAndTextTuples:
        width = get_app().output.get_size().columns
        name_width = max(20, width - 56)
        matches = self.matches()
        count = self.visible()
        rule = ("class:placeholder", f"  {'─' * (width - 4)}\n")
        heading = (
            f"    {'Model': <{name_width}}  {'Context':>7}  {'In $/M':>7}  "
            f"{'Out $/M':>8}  Also accepts"
        )
        lines: StyleAndTextTuples = [("bold", heading[: width - 1] + "\n"), rule]
        if self.row is not None:
            if self.row < self.top:
                self.top = self.row
            elif self.row >= self.top + count:
                self.top = self.row - count + 1

        if not matches:
            lines.append(("class:placeholder", "    No models match.\n"))

        for index, model in enumerate(matches[self.top : self.top + count], self.top):
            pointer = "❯ " if index == self.row else "  "
            if self.multi:
                mark = "◉" if model["slug"] in self.checked else "○"
            else:
                mark = (
                    "✓" if (model["provider"], model["slug"]) == self.current else " "
                )
            name = model["display_name"]
            if self.show_provider:
                name += f" · {PROVIDER_LABELS[model['provider']]}"
            if len(name) > name_width:
                name = name[: name_width - 1] + "…"
            input_price, output_price = price_labels(model)
            line = (
                f"{pointer}{mark} {name:<{name_width}}  {context_label(model):>7}  "
                f"{input_price:>7}  {output_price:>8}  {', '.join(model.get('features', []))}"
            )
            style = "fg:#ffffff bg:#303b59 bold" if index == self.row else ""
            lines.append((style, line[: width - 1] + "\n"))
        lines.append(rule)
        return lines

    def title_line(self) -> StyleAndTextTuples:
        if len(self.tabs) < 2:
            return [("bold", self.header())]
        line: StyleAndTextTuples = [("", " ")]
        for index, name in enumerate(self.tabs):
            style = "fg:#ffffff bg:#303b59 bold" if index == self.tab else "class:placeholder"
            line += [(style, f" {name} "), ("", " ")]
        if self.checked:
            line.append(("bold", f" {len(self.checked)} selected"))
        return line

    def header(self) -> str:
        return f" {self.title}" + (
            f" {len(self.checked)} selected" if self.checked else ""
        )

    def footer(self) -> str:
        text = f"   {len(self.matches())} of {len(self.models)} · ↑↓ move"
        if self.multi:
            text += " · Space select"
        text += f" · Enter {self.action}"
        if len(self.tabs) > 1:
            text += " · Tab switch"
        return text
