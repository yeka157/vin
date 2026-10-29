"""Typed construction and inspection for receiver-loop UI tests."""

import asyncio
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from typing import Literal, TypedDict, Unpack, cast
from types import ModuleType
from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.key_binding import KeyBindingsBase, KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import BufferControl, ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.output import ColorDepth
from rich.console import Console
from typing import TextIO

from vin.tui import bindings, layout, transcript as transcript_ui
from vin.tui.app import VinApp
from vin.types import Credentials, TranscriptEntry

REAL_SLEEP = asyncio.sleep


class SessionOptions(TypedDict):
    style: BaseStyle
    color_depth: ColorDepth
    multiline: FilterOrBool
    key_bindings: KeyBindingsBase
    erase_when_done: bool
    refresh_interval: float
    completer: Completer
    complete_while_typing: FilterOrBool


class ApplicationOptions(TypedDict):
    layout: Layout
    key_bindings: KeyBindingsBase | None
    style: BaseStyle
    color_depth: ColorDepth
    full_screen: bool
    erase_when_done: bool
    enable_page_navigation_bindings: FilterOrBool
    refresh_interval: float
    mouse_support: FilterOrBool


class ConsoleArguments(TypedDict, total=False):
    file: TextIO
    width: int
    height: int
    force_terminal: bool
    color_system: Literal["auto", "standard", "256", "truecolor", "windows"] | None
    no_color: bool


@contextmanager
def patch_tui(module: ModuleType, *, create: bool = False, **replacements: object) -> Generator[None]:
    owners = {"PromptSession": layout, "Application": layout, "Console": transcript_ui}
    with ExitStack() as stack:
        for name, value in replacements.items():
            stack.enter_context(patch.object(owners.get(name, module), name, value, create=create))
        yield


@contextmanager
def patch_ui_clock(*, return_value: float | None = None, side_effect: list[float] | None = None) -> Generator[Mock]:
    clock = Mock(return_value=return_value, side_effect=side_effect)
    with ExitStack() as stack:
        for module in (layout, bindings):
            stack.enter_context(patch.object(module, "monotonic", clock))
        from vin.tui import app
        stack.enter_context(patch.object(app, "monotonic", clock))
        yield clock


def configure_test_session(session: PromptSession[str], **options: Unpack[SessionOptions]) -> PromptSession[str]:
    for name, value in options.items():
        setattr(session, name, value)
    return session


def console_factory(console: Console | Mock, **options: Unpack[ConsoleArguments]) -> Console | Mock:
    return Console(**options) if options else console


def mock_prompt(session: PromptSession[str]) -> AsyncMock:
    prompt = session.prompt_async
    assert isinstance(prompt, AsyncMock)
    return prompt


def mock_run(application: Application[str]) -> AsyncMock:
    runner = application.run_async
    assert isinstance(runner, AsyncMock)
    return runner


def transcript_entries(session: PromptSession[str]) -> list[TranscriptEntry]:
    root = session.layout.container
    assert isinstance(root, HSplit)
    window = root.children[0]
    assert isinstance(window, Window)
    control = window.content
    assert isinstance(control, transcript_ui.TranscriptControl)
    return list(control.get_entries())


def loading_container(session: PromptSession[str]) -> ConditionalContainer:
    root = session.layout.container
    assert isinstance(root, HSplit)
    loading = root.children[1]
    assert isinstance(loading, ConditionalContainer)
    return loading


def loading_control(session: PromptSession[str]) -> FormattedTextControl:
    loading = loading_container(session)
    body = loading.content
    assert isinstance(body, HSplit)
    window = body.children[0]
    assert isinstance(window, Window)
    assert isinstance(window.content, FormattedTextControl)
    return window.content


def session_bindings(session: PromptSession[str]) -> KeyBindingsBase:
    assert session.key_bindings is not None
    return session.key_bindings


def fixture_credentials(token: str = "fake-token", subject: str = "fake-account") -> Credentials:
    return {"issuer": "https://test.invalid", "subject": subject, "client_id": "test-client", "ext_agent_host_id": "test-host", "access_token": token}


def create_test_application(session: PromptSession[str], **options: Unpack[ApplicationOptions]) -> Application[str]:
    app = Application[str](**options)
    native_run = app.run_async

    async def feed_submissions() -> None:
        run_future = app.future
        assert run_future is not None
        handler = session.default_buffer.accept_handler
        owner = cast(object, getattr(handler, "__self__", None))
        assert isinstance(owner, VinApp)

        async def wait_for_setup() -> None:
            while owner.loading_model:
                if run_future.done():
                    return
                if not isinstance(app.layout.current_control, BufferControl):
                    app.key_processor.feed(KeyPress(Keys.ControlM))
                    app.key_processor.process_keys()
                await REAL_SLEEP(0)

        await wait_for_setup()
        if run_future.done():
            return
        while True:
            text = await session.prompt_async("❯ ")
            session.default_buffer.text = text
            app.key_processor.feed(KeyPress(Keys.ControlM))
            app.key_processor.process_keys()
            if text in ("/exit", "/quit"):
                await run_future
                return
            await REAL_SLEEP(0)
            await wait_for_setup()
            if run_future.done():
                return
            if not isinstance(app.layout.current_control, BufferControl):
                app.key_processor.feed(KeyPress(Keys.ControlM))
                app.key_processor.process_keys()
                await REAL_SLEEP(0)

    async def run_async(*, pre_run: Callable[[], None] | None = None, set_exception_handler: bool = True, handle_sigint: bool = True, slow_callback_duration: float = 0.5) -> str:
        if runner.await_count > 1:
            raise AssertionError("Vin restarted the Application after it exited.")

        def start() -> None:
            if pre_run is not None:
                pre_run()
            app.create_background_task(feed_submissions())

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()

        def fail_on_background_error(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
            error = context.get("exception")
            if app.future is not None and not app.future.done():
                app.exit(exception=error if isinstance(error, BaseException) else RuntimeError(context["message"]))
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(fail_on_background_error)
        try:
            return await asyncio.wait_for(native_run(pre_run=start, set_exception_handler=False, handle_sigint=handle_sigint, slow_callback_duration=slow_callback_duration), timeout=5)
        except TimeoutError as error:
            raise AssertionError("The UI did not finish within 5 seconds.") from error
        finally:
            loop.set_exception_handler(previous_handler)

    runner = AsyncMock(wraps=run_async)
    app.run_async = runner
    return app


def preserve_credentials(saved: Credentials) -> Credentials:
    return saved


def fixture_registration(subject: str = "fake-account") -> Credentials:
    return {"issuer": "https://test.invalid", "subject": subject, "client_id": "test-client", "ext_agent_host_id": "test-host"}


def sign_out_registration(saved: Credentials, *, notify: Callable[[str], None] = print) -> Credentials:
    return fixture_registration(saved["subject"])
