"""Build the editable composer, panels, loading indicator, and application."""

from time import monotonic
from typing import TYPE_CHECKING

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Always, Condition
from prompt_toolkit.layout import Container, ConditionalContainer, DynamicContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from vin.tui.bindings import create_bindings
from vin.tui.completion import SessionCompleter

if TYPE_CHECKING:
    from vin.tui.app import VinApp


def build_workspace(app: "VinApp") -> None:
    input_style = Style.from_dict(
        {
            "": "fg:#ffffff",
            "placeholder": "fg:#888888",
            "input-border": "fg:#6b6b6b",
            "completion-menu": "bg:default",
            "completion-menu.completion": "bg:default fg:#888888",
            "completion-menu.completion.current": "bg:default fg:#a5b4fc bold noreverse",
            "completion-menu.meta.completion": "bg:default fg:#888888",
            "completion-menu.meta.completion.current": "bg:default fg:#a5b4fc noreverse",
        }
    )

    bindings = create_bindings(app)

    app.session = PromptSession(
        style=input_style,
        color_depth=ColorDepth.TRUE_COLOR,
        multiline=True,
        key_bindings=bindings,
        erase_when_done=True,
        refresh_interval=0.1,
        completer=SessionCompleter(
            app.project_dir,
            WordCompleter(
                [
                    "/help",
                    "/user",
                    "/login",
                    "/logout",
                    "/exit",
                    "/clear",
                    "/model",
                    "/resume",
                    "/rename",
                ],
                meta_dict={
                    "/help": "Show commands",
                    "/clear": "Start new session with empty context",
                    "/model": "Set the AI model",
                    "/resume": "Resume a saved conversation",
                    "/rename": "Rename the current conversation",
                    "/user": "Show current account",
                    "/login": "Sign in with ChatGPT Subscriptions",
                    "/logout": "Sign out",
                    "/exit": "Exit Vin",
                },
                sentence=True,
            ),
        ),
        complete_while_typing=True,
    )
    app.input_window = app.session.layout.current_window
    app.input_window.height = Dimension(min=1)
    app.input_window.dont_extend_height = Always()

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    loading = ConditionalContainer(
        HSplit(
            [
                Window(
                    FormattedTextControl(
                        lambda: frames[int(monotonic() * 10) % len(frames)]
                        + " "
                        + (
                            app.auth_status
                            or (
                                (
                                    " Loading models…"
                                    if app.credentials
                                    and app.credentials.get("access_token")
                                    else " Waiting for sign-in…"
                                )
                                if app.loading_model
                                else "Thinking…"
                            )
                        )
                    ),
                    height=1,
                    style="class:placeholder",
                ),
                Window(height=1),
            ]
        ),
        filter=Condition(lambda: (app.loading_model and app.choice_future is None)),
    )

    completion_menu = CompletionsMenu(max_height=5)
    assert isinstance(completion_menu.content, Window)
    completion_menu.content.right_margins = []

    menu_area = ConditionalContainer(
        completion_menu,
        filter=Condition(lambda: app.choice_future is None),
    )

    composer = HSplit(
        [
            ConditionalContainer(
                Window(
                    FormattedTextControl(
                        lambda: (
                            (
                                f" Paused ({len(app.queued_prompts)}) · Enter to resume: "
                                if app.queue_paused
                                else f" Pending ({len(app.queued_prompts)}): "
                            )
                            + " · ".join(
                                " ".join(text.split()) for text in app.queued_prompts
                            )
                        ),
                        show_cursor=False,
                    ),
                    height=1,
                    wrap_lines=False,
                    style="fg:#a5b4fc",
                ),
                filter=Condition(lambda: bool(app.queued_prompts)),
            ),
            Window(height=1, char="─", style="class:input-border"),
            app.input_window,
            Window(height=1, char="─", style="class:input-border"),
            Window(
                FormattedTextControl(lambda: app.model_indicator_text()),
                height=1,
                style="class:placeholder",
            ),
            Window(height=1),
        ]
    )

    def current_panel() -> Container:
        assert app.choice_menu is not None
        return app.choice_menu

    panel = HSplit(
        [
            Window(height=1, char="─", style="fg:#a5b4fc"),
            Window(height=1),
            DynamicContainer(current_panel),
            Window(
                FormattedTextControl("  Esc to return"),
                height=1,
                style="class:placeholder",
            ),
            Window(height=1),
        ]
    )

    app.session.layout.container = HSplit(
        [
            Window(
                app.transcript_control,
                height=Dimension(min=0),
                wrap_lines=True,
            ),
            loading,
            menu_area,
            DynamicContainer(
                lambda: panel if app.choice_menu is not None else composer
            ),
        ]
    )

    app.app = Application(
        layout=app.session.layout,
        key_bindings=app.session.app.key_bindings,
        style=input_style,
        color_depth=ColorDepth.TRUE_COLOR,
        full_screen=True,
        erase_when_done=True,
        enable_page_navigation_bindings=False,
        refresh_interval=0.1,
        mouse_support=True,
    )

    app.session.app = app.app
