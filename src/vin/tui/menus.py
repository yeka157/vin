"""Inline menus, secret prompts and panels shown in place of the input box."""

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import BufferControl, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import (
    BeforeInput,
    Processor,
    Transformation,
    TransformationInput,
)
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.widgets import RadioList
from rich.console import RenderableType
from rich.padding import Padding

from vin.auth.openrouter import mask_key
from vin.tui.transcript import render_transcript
from vin.types.models import ModelMetadata

if TYPE_CHECKING:
    from vin.tui.app import VinApp


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and bool(task.cancelling())


class MaskKey(Processor):
    def apply_transformation(
        self, transformation_input: TransformationInput
    ) -> Transformation:
        text = fragment_list_to_text(transformation_input.fragments)
        return Transformation([("", mask_key(text, max(len(text) - 13, 0)))])


def create_choice_menu(
    message: str,
    options: Sequence[tuple[str, str]],
    default: str | None = None,
) -> tuple[RadioList[str], HSplit]:
    if not options:
        raise RuntimeError("No choices are available.")

    choices: RadioList[str] = RadioList(
        values=options,
        default=default,
        select_on_focus=True,
        show_numbers=False,
        open_character="",
        select_character="❯",
        close_character="",
        show_cursor=False,
        show_scrollbar=False,
        container_style="class:completion-menu.completion",
        default_style="",
        selected_style="fg:#8fa8ff bold",
        checked_style="class:completion-menu.completion.current",
        number_style="",
    )

    container = HSplit(
        [
            Window(FormattedTextControl(f" {message}"), height=1),
            choices,
        ],
        height=Dimension(max=6),
    )

    return choices, container


def setting_row(label: str, value: str) -> str:
    return f"{label:<20}{value}"


def choose_login_method() -> str:
    bindings = KeyBindings()

    @bindings.add("c-d")
    def exit_menu(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    return choice(
        message="Sign in to Vin",
        options=[("chatgpt", "Sign in with ChatGPT")],
        symbol="❯",
        key_bindings=bindings,
    )


def choose_chatgpt_model(models: Sequence[ModelMetadata]) -> str:
    if not models:
        raise RuntimeError("No ChatGPT models are available for this account.")

    bindings = KeyBindings()

    @bindings.add("c-d")
    def exit_menu(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    return choice(
        message="Choose a ChatGPT model",
        options=[(model["slug"], model["display_name"]) for model in models],
        symbol="❯",
        key_bindings=bindings,
    )


async def ask_secret(owner: "VinApp", message: str) -> str:
    if owner.choice_future is not None or owner.secret_future is not None:
        raise RuntimeError("A choice is already open")
    buffer = Buffer(multiline=False)
    field = Window(
        BufferControl(buffer, input_processors=[MaskKey(), BeforeInput(" › ")]),
        height=1,
    )
    owner.choice_menu = HSplit(
        [
            Window(FormattedTextControl(f" {message}"), height=1, style="bold"),
            field,
            Window(
                FormattedTextControl(" Enter to confirm"),
                height=1,
                style="class:placeholder",
            ),
        ]
    )
    owner.secret_future = asyncio.get_running_loop().create_future()
    try:
        owner.session.layout.focus(field)
        owner.app.invalidate()
        return await owner.secret_future
    finally:
        owner.secret_future = None
        owner.choice_menu = None
        owner.session.layout.focus(owner.input_window)
        owner.app.invalidate()


async def choose_inline(owner: "VinApp",
    message: str,
    options: Sequence[tuple[str, str]],
    default: str | None = None,
) -> str:
    if owner.choice_future is not None:
        raise RuntimeError("A choice is already open")

    owner.choices, owner.choice_menu = create_choice_menu(message, options, default)
    owner.choice_future = asyncio.get_running_loop().create_future()

    try:
        owner.session.layout.focus(owner.choices)
        owner.session.app.invalidate()
        selected = await owner.choice_future
        assert selected is not None
        return selected
    finally:
        owner.choices = None
        owner.choice_menu = None
        owner.choice_future = None
        owner.session.layout.focus(owner.input_window)
        owner.session.app.invalidate()


async def show_panel(owner: "VinApp", content: RenderableType) -> None:
    if owner.choice_future is not None:
        raise RuntimeError("A panel is already open")

    owner.choices = None
    owner.choice_menu = Window(
        FormattedTextControl(
            lambda: render_transcript(
                [
                    {
                        "role": "notice",
                        "content": Padding(
                            content,
                            (
                                0,
                                0,
                                0,
                                2,
                            ),
                        ),
                    }
                ],
                owner.app.output.get_size().columns,
                owner.app.output.get_size().rows,
            ),
            focusable=True,
            show_cursor=False,
        ),
        wrap_lines=True,
        dont_extend_height=True,
    )
    owner.choice_future = asyncio.get_running_loop().create_future()

    try:
        owner.session.layout.focus(owner.choice_menu)
        owner.app.invalidate()
        await owner.choice_future
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
    finally:
        owner.choice_menu = None
        owner.choice_future = None
        owner.session.layout.focus(owner.input_window)
        owner.app.invalidate()


async def choose_or_back(
    owner: "VinApp", title: str, rows: list[tuple[str, str]], focus: str | None = None
) -> str | None:
    """One menu step; None when the user presses Esc to go back."""
    try:
        return await owner.choose_inline(title, rows, default=focus)
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
