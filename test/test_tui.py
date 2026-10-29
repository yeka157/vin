import asyncio
import vin.tui.app as vin
from vin import main as entrypoint
from vin.tui import transcript as transcript_ui
from contextlib import nullcontext, redirect_stdout
from threading import Event, get_ident
from unittest.mock import Mock, patch, AsyncMock, call
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import BufferControl, Layout, Window
from io import StringIO
from rich.console import Console
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.input import DummyInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit import PromptSession, Application
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.buffer import Buffer

from functools import partial
from collections.abc import Callable, Coroutine, Sequence
from typing import Literal, TypedDict, cast
from rich.console import RenderableType
from vin.types import Credentials, History, ModelMetadata, TranscriptEntry
from vin.sessions.transcript import decode_transcript
from vin.provider.events import ReplyEvents
from prompt_toolkit.layout.controls import UIContent
from session_typing_helpers import present, text_value, validated_transcript
from tui_typing_helpers import key_event
from tui_application_helpers import (
    configure_test_session, console_factory, create_test_application,
    fixture_credentials, loading_container, loading_control, mock_prompt, mock_run,
    patch_tui, patch_ui_clock, session_bindings, transcript_entries,
    fixture_registration, preserve_credentials, sign_out_registration,
)


REAL_SLEEP = asyncio.sleep


def setUpModule() -> None:
    # Receiver-loop tests may checkpoint conversations; never use user storage.
    temporary = TemporaryDirectory()
    unittest.addModuleCleanup(temporary.cleanup)
    location = patch("vin.sessions.storage.SESSIONS_DIR", Path(temporary.name) / "sessions")
    location.start()
    unittest.addModuleCleanup(location.stop)




class SignInContext(TypedDict, total=False):
    loop: asyncio.AbstractEventLoop
    cancel_event: Event
    notify: Callable[[str], None]


class RenderingTests(unittest.TestCase):


    def test_intermediate_update_and_final_answer_keep_readable_markdown(self) -> None:
        entries: list[TranscriptEntry] = [
            {"role": "assistant", "content": "**Checking** the file.", "intermediate": True, "streaming": False},
            {"role": "assistant", "content": "**Finished** the change.", "streaming": False},
        ]
        fragments = to_formatted_text(transcript_ui.render_transcript(entries, 60, 20))
        text = "".join(fragment[1] for fragment in fragments)
        dim_text = "".join(fragment[1] for fragment in fragments if "dim" in fragment[0].split())
        bold_text = "".join(fragment[1] for fragment in fragments if "bold" in fragment[0].split())
        self.assertIn("Checking the file.", text)
        self.assertIn("Finished the change.", text)
        self.assertNotIn("**", text)
        self.assertNotIn("Checking", dim_text)
        self.assertNotIn("Finished", dim_text)
        self.assertIn("Checking", bold_text)
        self.assertIn("Finished", bold_text)


class TranscriptControlTests(unittest.TestCase):

    def test_viewport_holds_earlier_output_during_streaming_and_can_follow_latest(self) -> None:
        entries: list[TranscriptEntry] = [{"role": "notice", "content": f"message {i:02}"} for i in range(20)]
        entries.append({"role": "assistant", "content": "stream partial", "streaming": True})
        control = transcript_ui.TranscriptControl(lambda: entries)

        def visible(content: UIContent) -> str:
            return "\n".join(
                "".join(part[1] for part in content.get_line(index))
                for index in range(content.line_count)
            )

        latest = control.create_content(40, 5)
        self.assertEqual(latest.line_count, 5)
        self.assertIn("stream partial", visible(latest))
        self.assertNotIn("message 00", visible(latest))

        control.scroll(-10_000)
        earlier = control.create_content(40, 5)
        before = visible(earlier)
        self.assertIn("message 00", before)
        self.assertNotIn("stream partial", before)

        assert entries[-1]["role"] == "assistant"
        entries[-1]["content"] += "\n" + "new streamed line\n" * 12 + "stream latest"
        growing = control.create_content(40, 5)
        self.assertEqual(visible(growing), before)
        self.assertEqual(control.top_line, 0)

        control.scroll_to_bottom()
        latest = control.create_content(40, 5)
        self.assertIn("stream latest", visible(latest))
        self.assertIsNone(control.top_line)
        self.assertFalse(latest.show_cursor)

        control.scroll(-3)
        resized = control.create_content(40, 12)
        self.assertLessEqual(resized.line_count, 12)
        self.assertIsNone(control.top_line)
        self.assertIn("stream latest", visible(resized))

        control.scroll(-10_000)
        entries = []
        empty = control.create_content(20, 3)
        self.assertEqual(visible(empty), "")
        self.assertIsNone(control.top_line)
        self.assertEqual(empty.cursor_position.y, 0)


class TranscriptLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_transcript_keeps_latest_message_and_draft_visible(self) -> None:
        from prompt_toolkit import Application
        from prompt_toolkit.application.current import create_app_session, set_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.layout import HSplit, Layout
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.mouse_handlers import MouseHandlers
        from prompt_toolkit.layout.screen import Screen, WritePosition

        entries: list[TranscriptEntry] = [
            {"role": "notice", "content": f"message {i}"} for i in range(50)
        ]
        transcript = Window(
            transcript_ui.TranscriptControl(lambda: entries),
            height=Dimension(min=0),
            wrap_lines=True,
        )
        input_control = BufferControl(Buffer(document=Document("next draft")))
        input_window = Window(input_control, height=1)
        root = HSplit([transcript, Window(height=1, char="─"), input_window])

        with create_app_session(input=DummyInput(), output=DummyOutput()):
            app = Application[str](layout=Layout(root, focused_element=input_window))
            try:
                with set_app(app):
                    screen = Screen()
                    root.write_to_screen(
                        screen,
                        MouseHandlers(),
                        WritePosition(xpos=0, ypos=0, width=40, height=10),
                        "",
                        True,
                        None,
                    )
                    text = "\n".join(
                        "".join(screen.data_buffer[y][x].char for x in range(40))
                        for y in range(10)
                    )
                    self.assertIn("message 49", text)
                    self.assertIn("next draft", text)
                    self.assertIs(app.layout.current_control, input_control)
                    bounds = screen.visible_windows_to_write_positions[input_window]
                    self.assertEqual(bounds.height, 1)
                    self.assertLessEqual(bounds.ypos + bounds.height, 10)
            finally:
                await app.cancel_and_wait_for_background_tasks()


class StreamingReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(vin, "refresh_credentials", side_effect=preserve_credentials))
        self.owner = vin.VinApp()
        self.owner.credentials = fixture_credentials("fake-access-token")
        self.owner.selected_model = "selected-model"
        self.owner.messages = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        self.previous = [message.copy() for message in self.owner.messages]
        self.draft = Document("next draft\nkeep typing", cursor_position=4)
        self.owner.session = PromptSession[str](input=DummyInput(), output=DummyOutput())
        self.owner.session.default_buffer = Buffer(document=self.draft)
        self.owner.session.app = cast(Application[str], Mock(spec=Application))

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await REAL_SLEEP(0.001)

    def start_reply(self, text: str) -> asyncio.Task[Literal["error", "rate_limit"] | None]:
        self.assertTrue(
            callable(self.owner.stream_reply),
            "VinApp needs the async stream_reply method before connecting requests.",
        )
        task = asyncio.create_task(self.owner.stream_reply(text))
        self.owner.pending_reply = task
        return task

    def assert_completion_footer(self, entry: TranscriptEntry) -> None:
        self.assertEqual(entry["role"], "notice")
        self.assertIsInstance(entry["content"], vin.Text)
        self.assertEqual(text_value(entry["content"]).style, "dim")
        self.assertRegex(
            text_value(entry["content"]).plain,
            r"^[*✻] Worked for (?:\d+h )?(?:\d+m )?\d+s · done \d{1,2}:\d{2} [AP]M$",
        )

    async def test_worker_streams_one_entry_and_commits_context_only_after_completion(self) -> None:
        release = Event()
        seen: list[tuple[int, Credentials | None, str | None, History, Event]] = []
        ui_thread = get_ident()

        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
            seen.append((get_ident(), credentials, model, messages, cancel_event))
            events.put(("text", "**Hel"))
            release.wait(2)
            events.put(("text", "lo**"))
            events.put(("complete", None))

        with patch.object(vin, "stream_chatgpt_response", side_effect=worker, create=True):
            task = self.start_reply("Next question")
            try:
                await self.wait_until(lambda: bool(self.owner.transcript))
                self.assertEqual(self.owner.transcript, [{
                    "role": "assistant", "content": "**Hel", "streaming": True,
                }])
                entry = self.owner.transcript[0]
                self.assertIs(self.owner.reply_entry, entry)
                self.assertEqual(self.owner.messages, self.previous)
                self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                self.assertNotEqual(seen[0][0], ui_thread)
                self.assertEqual(seen[0][1], fixture_credentials("fake-access-token"))
                self.assertEqual(seen[0][2], "selected-model")
                self.assertEqual(seen[0][3], self.previous + [
                    {"role": "user", "content": "Next question"}
                ])
                self.assertIsNot(seen[0][1], self.owner.credentials)
                self.assertIsNot(seen[0][3], self.owner.messages)
                self.assertIsNot(seen[0][3][0], self.owner.messages[0])
                release.set()
                await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(len(self.owner.transcript), 2)
        self.assertEqual(self.owner.transcript[0], {
            "role": "assistant", "content": "**Hello**", "streaming": False,
        })
        self.assert_completion_footer(self.owner.transcript[1])
        self.assertIs(self.owner.transcript[0], entry)
        self.assertEqual(self.owner.messages, self.previous + [
            {"role": "user", "content": "Next question"},
            {"role": "assistant", "content": "**Hello**"},
        ])
        self.assertIsNone(self.owner.pending_reply)
        self.assertIsNone(self.owner.reply_entry)
        self.assertTrue(seen[0][4].is_set())
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


    async def test_worker_error_keeps_partial_and_draft_without_committing_context(self) -> None:
        cases: list[tuple[str | None, Literal["error", "rate_limit"], str]] = [
            (None, "error", "ChatGPT stream ended before completion."),
            ("**partial", "error", "ChatGPT stream ended before completion."),
            ("**partial", "rate_limit", "ChatGPT request failed (HTTP 429)."),
        ]
        for partial, kind, message in cases:
            with self.subTest(partial=partial, kind=kind):
                self.owner.transcript.clear()
                cancel_events: list[Event] = []
                assert kind == "error" or kind == "rate_limit"

                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, event_kind: Literal["error", "rate_limit"] = kind) -> None:
                    cancel_events.append(cancel_event)
                    if partial:
                        events.put(("text", partial))
                    events.put((event_kind, message))

                with (
                    patch.object(vin, "stream_chatgpt_response", side_effect=worker, create=True),
                    patch_ui_clock(side_effect=[100.0, 103.75]),
                ):
                    outcome = await asyncio.wait_for(self.start_reply("Next question"), timeout=2)

                self.assertEqual(outcome, kind)
                self.assertEqual(self.owner.messages, self.previous)
                assistants = [entry for entry in self.owner.transcript if entry["role"] == "assistant"]
                self.assertEqual(assistants, [{
                    "role": "assistant", "content": partial, "streaming": True,
                }] if partial else [])
                self.assertEqual(len(self.owner.transcript), 3 if partial else 2)
                notice = self.owner.transcript[-2]
                self.assertEqual(notice["role"], "notice")
                self.assertEqual(text_value(notice["content"]).plain, message)
                self.assertEqual(text_value(notice["content"]).style, "red")
                footer = self.owner.transcript[-1]
                self.assertEqual(footer["role"], "notice")
                self.assertEqual(text_value(footer["content"]).plain, "Failed after 3s")
                self.assertEqual(text_value(footer["content"]).style, "red")
                self.assertIsNone(self.owner.pending_reply)
                self.assertIsNone(self.owner.reply_entry)
                self.assertIsNone(self.owner.request_started_at)
                self.assertEqual(self.owner.last_request_duration, 3.75)
                self.assertTrue(cancel_events[0].is_set())
                self.assertEqual(self.owner.session.default_buffer.document, self.draft)

    async def test_cancelled_worker_cannot_change_later_reply_or_its_context(self) -> None:
        old_release, new_release, old_finished = Event(), Event(), Event()
        seen: list[tuple[ReplyEvents, Event]] = []

        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
            seen.append((events, cancel_event))
            if len(seen) == 1:
                try:
                    events.put(("text", "old partial"))
                    old_release.wait(2)
                    events.put(("text", "late text"))
                    events.put(("complete", None))
                    events.put(("error", "late error"))
                finally:
                    old_finished.set()
            else:
                events.put(("text", "new answer"))
                new_release.wait(2)
                events.put(("complete", None))

        tasks: list[asyncio.Task[Literal["error", "rate_limit"] | None]] = []
        with patch.object(vin, "stream_chatgpt_response", side_effect=worker, create=True):
            old = self.start_reply("Cancelled question")
            tasks.append(old)
            try:
                await self.wait_until(lambda: bool(self.owner.transcript))
                self.owner.pending_reply = None
                old.cancel()
                new = self.start_reply("New question")
                tasks.append(new)
                with self.assertRaises(asyncio.CancelledError):
                    await old
                self.assertIs(self.owner.pending_reply, new)
                self.assertTrue(seen[0][1].is_set())
                await self.wait_until(lambda: len(self.owner.transcript) == 2)
                new_release.set()
                await asyncio.wait_for(new, timeout=2)
                old_release.set()
                await self.wait_until(old_finished.is_set)
            finally:
                old_release.set()
                new_release.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        self.assertIsNot(seen[0][0], seen[1][0])
        self.assertIsNot(seen[0][1], seen[1][1])
        self.assertEqual(len(self.owner.transcript), 3)
        self.assertEqual(self.owner.transcript[:2], [
            {"role": "assistant", "content": "old partial", "streaming": True},
            {"role": "assistant", "content": "new answer", "streaming": False},
        ])
        self.assert_completion_footer(self.owner.transcript[2])
        self.assertEqual(self.owner.messages, self.previous + [
            {"role": "user", "content": "New question"},
            {"role": "assistant", "content": "new answer"},
        ])
        self.assertIsNone(self.owner.pending_reply)
        self.assertIsNone(self.owner.reply_entry)
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


class ChatFlowTests(unittest.TestCase):
    def test_user_command_before_and_after_logout(self) -> None:
        output = StringIO()
        panels: list[RenderableType] = []
        exchange: History = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        show_panel = vin.VinApp.show_panel
        credentials = fixture_credentials("fake", "test-account")
        credentials["email"] = "learner@example.test"

        session = PromptSession[str](input=DummyInput(), output=DummyOutput())
        session.prompt_async = AsyncMock(
            side_effect=["/user", "/logout", "/user", "/help", "/quit"]
        )

        async def observe_panel(instance: vin.VinApp, content: RenderableType) -> None:
            if not panels:
                instance.transcript.extend(validated_transcript(decode_transcript(exchange)))
                instance.messages.extend(exchange)
            previous_transcript = list(instance.transcript)
            previous_messages = list(instance.messages)
            panels.append(content)
            buffer = session.default_buffer
            buffer.document = Document("next draft\nline", cursor_position=5)
            await show_panel(instance, content)
            self.assertEqual(instance.transcript, previous_transcript)
            self.assertEqual(instance.messages, previous_messages)
            self.assertEqual(instance.messages, exchange)
            self.assertEqual(
                (buffer.text, buffer.cursor_position), ("next draft\nline", 5)
            )
            self.assertIs(session.layout.current_window, instance.input_window)
            self.assertIsNone(instance.choice_menu)
            self.assertIsNone(instance.choice_future)

        with patch_tui(
            vin,
            create=True,
            PromptSession=Mock(
                side_effect=partial(configure_test_session, session)
            ),
            Application=partial(create_test_application, session),
            Console=partial(console_factory, Console(file=output, width=80, color_system=None)),
            load_active_session=lambda: credentials,
            refresh_credentials=preserve_credentials,
            logout_chatgpt=sign_out_registration,
            patch_stdout=Mock(return_value=nullcontext()),
            list_chatgpt_models=Mock(return_value=[{"slug": "test-model", "display_name": "Test Model"}]),
            choose_chatgpt_model=Mock(return_value="test-model"),
            load_model_preference=lambda: None,
            save_model_preference=Mock(),
            save_cached_chatgpt_models=Mock(),
        ):
            with patch.object(vin.VinApp, "show_panel", observe_panel):
                with create_pipe_input() as pipe:
                    with create_app_session(input=pipe, output=DummyOutput()):
                        entrypoint.main()

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(panels), 3)
        self.assertIsInstance(panels[0], vin.Text)
        self.assertEqual(text_value(panels[0]).plain, "ChatGPT: learner@example.test")
        self.assertIsInstance(panels[1], vin.Text)
        self.assertEqual(text_value(panels[1]).plain, "Signed out. Use /login to sign in")
        self.assertIsInstance(panels[2], vin.Markdown)
        self.assertEqual(mock_prompt(session).await_count, 5)
        entries = transcript_entries(session)
        rendered = "".join(
            fragment[1]
            for fragment in to_formatted_text(transcript_ui.render_transcript(entries, 80, 20))
        )
        self.assertNotIn("ChatGPT: learner@example.test", rendered)
        self.assertEqual(rendered.count("Signed out of Vin."), 1)
        self.assertNotIn("Signed out. Use /login to sign in", rendered)
        self.assertNotIn("Commands", rendered)
        self.assertIn("earlier question", rendered)
        self.assertIn("earlier answer", rendered)
        self.assertNotIn("Demo reply", rendered)
        self.assertFalse(
            any("/user" in line and "/login" in line for line in rendered.splitlines())
        )


class InitialModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = vin.VinApp()
        self.credentials: Credentials | None = fixture_credentials()
        self.models: list[ModelMetadata] = [
            {"slug": "old-model", "display_name": "Old Model"},
            {"slug": "new-model", "display_name": "New Model"},
        ]
        self.app.credentials = self.credentials
        self.app.models = [self.models[0]]
        self.app.selected_provider = "chatgpt"
        self.app.selected_model = "old-model"
        self.refresh = Mock(return_value=self.credentials)
        self.catalog = Mock(return_value=self.models)
        self.load = Mock(return_value={"provider": "chatgpt", "model": "old-model"})
        self.save = Mock()
        self.cache_save = Mock()
        self.inline = AsyncMock(return_value="new-model")
        replacements = patch.multiple(
            vin,
            create=True,
            refresh_credentials=self.refresh,
            list_chatgpt_models=self.catalog,
            load_model_preference=self.load,
            save_model_preference=self.save,
            save_cached_chatgpt_models=self.cache_save,
        )
        replacements.start()
        self.addCleanup(replacements.stop)
        picker = patch.object(self.app, "choose_inline", self.inline)
        picker.start()
        self.addCleanup(picker.stop)


    def test_first_selection_uses_inline_picker_and_saves_default(self) -> None:
        self.load.return_value = None

        asyncio.run(self.app.load_initial_model())

        self.inline.assert_awaited_once_with(
            "Choose a ChatGPT model",
            [("old-model", "Old Model"), ("new-model", "New Model")],
        )
        self.save.assert_called_once_with("chatgpt", "new-model")
        self.assertEqual(self.app.selected_model, "new-model")
        self.assertEqual(self.app.models, self.models)
        self.assertEqual(self.app.messages, [])
        self.assertEqual(self.app.transcript, [])


    def test_cancelled_selection_does_not_save_or_replace_active_model(self) -> None:
        self.load.return_value = None
        self.inline.side_effect = asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.app.load_initial_model())

        self.save.assert_not_called()
        self.assertEqual(self.app.models, [self.models[0]])
        self.assertEqual(self.app.selected_model, "old-model")
        self.assertEqual(self.app.selected_provider, "chatgpt")


class ModelCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.console = Mock(spec=Console)
        self.console_print = Mock()
        self.console_clear = Mock()
        self.console.print = self.console_print
        self.console.clear = self.console_clear
        self.credentials: Credentials | None = fixture_credentials()
        self.signed_in = fixture_credentials("fake-new-token", "new-fake-account")
        self.models: list[ModelMetadata] = [
            {"slug": "old-model", "display_name": "Old Model"},
            {"slug": "new-model", "display_name": "New Model"},
        ]
        self.session = PromptSession[str](input=DummyInput(), output=DummyOutput())
        self.factory = Mock(
            side_effect=partial(configure_test_session, self.session)
        )
        self.refresh = Mock(side_effect=preserve_credentials)
        self.load_session = Mock(side_effect=lambda: self.credentials)
        self.login = Mock(return_value=self.signed_in)
        self.logout = Mock(return_value=fixture_registration())
        self.login_picker = Mock(return_value="chatgpt")
        self.catalog = Mock(return_value=self.models)
        self.load = Mock(return_value={"provider": "chatgpt", "model": "old-model"})
        self.choose = Mock(return_value="new-model")
        self.inline = AsyncMock(return_value="new-model")
        self.save = Mock()
        self.cached = Mock(return_value=self.models)
        self.cache_save = Mock()

    def run_chat(self, commands: Sequence[str] | Callable[[str], Coroutine[object, object, str]]) -> None:
        self.vin_app = vin.VinApp()
        self.session.prompt_async = AsyncMock(side_effect=commands)

        with patch.object(self.vin_app, "choose_inline", self.inline), patch_tui(
            vin,
            create=True,
            PromptSession=self.factory,
            Application=partial(create_test_application, self.session),
            Console=partial(console_factory, self.console),
            load_active_session=self.load_session,
            refresh_credentials=self.refresh,
            logout_chatgpt=self.logout,
            choose_login_method=self.login_picker,
            login_with_chatgpt=self.login,
            list_chatgpt_models=self.catalog,
            load_model_preference=self.load,
            choose_chatgpt_model=self.choose,
            save_model_preference=self.save,
            load_cached_chatgpt_models=self.cached,
            save_cached_chatgpt_models=self.cache_save,
            patch_stdout=Mock(return_value=nullcontext()),
        ):
            with create_pipe_input() as pipe:
                with create_app_session(input=pipe, output=DummyOutput()):
                    self.vin_app.run()

    def assert_model_indicator(self, expected: str) -> None:
        visible_text: list[str] = []
        for window in self.session.layout.find_all_windows():
            if isinstance(window.content, vin.FormattedTextControl):
                rendered = window.content.create_content(80, 1)
                visible_text.append(
                    "".join(fragment[1] for fragment in rendered.get_line(0)).rstrip()
                )
        self.assertIn(expected, visible_text)

    def assert_transcript_live_indicator(self, elapsed: str | None = None) -> None:
        fragments = to_formatted_text(
            transcript_ui.render_transcript(self.vin_app.get_transcript_entries(), 80, 20)
        )
        text = "".join(fragment[1] for fragment in fragments)
        self.assertEqual(text.count("Thinking…"), 1)
        self.assertIn("Thinking… (", text)
        if elapsed is not None:
            self.assertIn(f"Thinking… ({elapsed})", text)

    def assert_error_notice(self, expected: str, *, command: str | None = None) -> None:
        entries = transcript_entries(self.session)
        self.assertEqual(len(entries), 2 if command is None else 3)
        if command is not None:
            self.assertEqual(entries[1], {"role": "user", "content": command})
        self.assertEqual(entries[-1]["role"], "notice")
        fragments = to_formatted_text(
            transcript_ui.render_transcript(entries[-1:], 80, 20)
        )
        text = "".join(fragment[1] for fragment in fragments)
        red_text = "".join(
            fragment[1] for fragment in fragments if "ansired" in fragment[0]
        )
        self.assertEqual(text, expected + "\n\n")
        self.assertEqual(red_text, expected)
        self.console_print.assert_not_called()


    def test_receiver_serializes_submissions_and_survives_response_cancellation(self) -> None:
        for outcome in ("complete", "error", "cancel", "exit", "rate_limit_empty", "rate_limit_draft"):
            with self.subTest(outcome=outcome):
                release = Event()
                text_ready = Event()
                finished = Event()
                requests: list[tuple[History, Event]] = []
                step = 0
                draft = Document("newer unsent\ndraft", cursor_position=5)

                async def wait_until(predicate: Callable[[], bool]) -> None:
                    async with asyncio.timeout(2):
                        while not predicate():
                            await REAL_SLEEP(0.001)

                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event) -> None:
                    requests.append((messages, cancel_event))
                    question = messages[-1]["content"]
                    if question == "first":
                        try:
                            text_ready.wait(2)
                            events.put(("text", "first partial"))
                            release.wait(2)
                            if outcome == "error":
                                events.put(("error", "Could not connect to ChatGPT."))
                            elif outcome.startswith("rate_limit"):
                                events.put(("rate_limit", "ChatGPT request failed (HTTP 429)."))
                            else:
                                events.put(("text", " done"))
                                events.put(("complete", None))
                        finally:
                            finished.set()
                    else:
                        events.put(("text", f"answer: {question}"))
                        events.put(("complete", None))

                async def read(_: str) -> str:
                    nonlocal step
                    step += 1
                    if step == 1:
                        return "first"

                    owner = self.vin_app

                    def accept(text: str) -> None:
                        buffer = self.session.default_buffer
                        buffer.document = Document(text)
                        owner.app.key_processor.feed(KeyPress(Keys.ControlM))
                        owner.app.key_processor.process_keys()
                        self.assertEqual(buffer.text, "")
                        buffer.document = draft

                    await wait_until(lambda: len(requests) == 1)
                    self.assertFalse(loading_container(self.session).filter())
                    self.assert_transcript_live_indicator()
                    accept("second")
                    text_ready.set()
                    await wait_until(lambda: owner.reply_entry is not None and
                                     owner.reply_entry["content"] == "first partial")
                    self.assertFalse(loading_container(self.session).filter())
                    self.assert_transcript_live_indicator()
                    accept("third")
                    for _tick in range(4):
                        await REAL_SLEEP(0)

                    self.assertEqual(list(owner.queued_prompts), ["second", "third"])
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(owner.messages, [])
                    self.assertEqual(
                        [entry["content"] for entry in owner.transcript
                         if entry["role"] == "user"],
                        ["first"],
                    )

                    if outcome == "exit":
                        owner.app.exit(result="/quit")
                        return "/quit"
                    if outcome == "cancel":
                        owner.app.key_processor.feed(KeyPress(Keys.ControlC))
                        owner.app.key_processor.process_keys()
                    release.set()
                    questions = ["first", "second", "third"]
                    if outcome.startswith("rate_limit"):
                        await wait_until(lambda: owner.queue_paused)
                        self.assertIsNone(owner.pending_reply)
                        self.assertEqual(len(requests), 1)
                        self.assertEqual(list(owner.queued_prompts), ["second", "third"])
                        self.assertEqual(owner.messages, [])
                        self.assertEqual(self.session.default_buffer.document, draft)

                        accept("/model")
                        await wait_until(lambda: owner.selected_model == "new-model")
                        self.assertTrue(owner.queue_paused)
                        self.assertEqual(len(requests), 1)
                        if outcome == "rate_limit_draft":
                            questions.append("fourth")
                            accept("fourth")
                        else:
                            accept("")
                        self.assertFalse(owner.queue_paused)

                    # Finalization clears pending_reply before the receiver drains
                    # prompt wakeups. Commands are accepted only after both settle.
                    await wait_until(lambda: len(requests) == len(questions)
                                     and owner.pending_reply is None
                                     and owner.submitted.empty())
                    self.assertEqual(self.session.default_buffer.document, draft)
                    self.assertEqual(
                        [request[-1]["content"] for request, _ in requests],
                        questions,
                    )
                    first_pair: History = [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "first partial done"},
                    ] if outcome == "complete" else []
                    expected_history = first_pair.copy()
                    for index, question in enumerate(questions[1:], start=1):
                        self.assertEqual(requests[index][0], expected_history + [
                            {"role": "user", "content": question},
                        ])
                        expected_history.extend([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": f"answer: {question}"},
                        ])
                    self.assertEqual(owner.messages, expected_history)
                    return "/quit"

                with patch.object(vin, "stream_chatgpt_response", side_effect=worker):
                    try:
                        self.run_chat(read)
                    finally:
                        text_ready.set()
                        release.set()
                        self.assertTrue(finished.wait(2))
                self.assertTrue(all(cancel.is_set() for _, cancel in requests))
                self.assertIsNone(self.vin_app.pending_reply)
                if outcome == "exit":
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(list(self.vin_app.queued_prompts), ["second", "third"])
                    self.assertEqual(self.vin_app.messages, [])


    def test_returning_startup_restores_cached_model_without_network_requests(self) -> None:
        self.refresh.side_effect = RuntimeError("Startup must not wait for token refresh.")
        self.catalog.side_effect = RuntimeError("Startup must not wait for discovery.")
        self.run_chat(["/quit"])

        self.assert_model_indicator(" ChatGPT · Old Model")
        self.refresh.assert_not_called()
        self.catalog.assert_not_called()
        self.inline.assert_not_awaited()
        self.save.assert_not_called()
        self.assertFalse(self.vin_app.loading_model)
        self.assertEqual(mock_prompt(self.session).await_count, 1)
        self.assertEqual(self.vin_app.messages, [])
        self.cache_save.assert_not_called()


    def test_model_command_discovers_and_caches_choices_only_when_cache_is_missing(self) -> None:
        self.cached.return_value = []
        self.run_chat(["/model", "/quit"])

        self.refresh.assert_called_once_with(self.credentials)
        self.catalog.assert_called_once_with(self.credentials)
        self.cache_save.assert_called_once_with(self.credentials, self.models)
        self.inline.assert_awaited_once_with(
            "Choose a ChatGPT model",
            [("old-model", "Old Model"), ("new-model", "New Model")],
            default="old-model",
        )
        self.save.assert_called_once_with("chatgpt", "new-model")
        self.assert_model_indicator(" ChatGPT · New Model")
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.assertFalse(self.vin_app.loading_model)


    def test_ctrl_c_cancels_on_demand_discovery_and_discards_late_models(self) -> None:
        self.cached.return_value = []
        release_worker = Event()
        worker_finished = Event()
        context: dict[str, asyncio.AbstractEventLoop] = {}
        step = 0

        def cancel_from_ui() -> None:
            buffer = self.session.default_buffer
            buffer.document = Document("keep this draft", cursor_position=4)
            self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlC))
            self.vin_app.app.key_processor.process_keys()

        def discover(credentials: Credentials) -> list[ModelMetadata]:
            self.assertIn("loop", context, "Discovery must not run at launch.")
            context["loop"].call_soon_threadsafe(cancel_from_ui)
            if not release_worker.wait(2):
                raise RuntimeError("The UI did not return after cancellation.")
            worker_finished.set()
            return self.models

        async def read(_: str) -> str:
            nonlocal step
            step += 1
            if step == 1:
                context["loop"] = asyncio.get_running_loop()
                return "/model"
            buffer = self.session.default_buffer
            self.assertEqual((buffer.text, buffer.cursor_position), ("keep this draft", 4))
            self.assertFalse(self.vin_app.loading_model)
            self.assertIsNone(self.vin_app.pending_model_load)
            release_worker.set()
            return "/quit"

        self.catalog.side_effect = discover
        try:
            self.run_chat(read)
        finally:
            release_worker.set()

        self.assertTrue(worker_finished.is_set())
        self.assertEqual(step, 2)
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.assertEqual(self.vin_app.models, [])
        self.assertEqual(self.vin_app.selected_model, "old-model")
        self.assertEqual(
            self.vin_app.transcript[1:],
            [
                {"role": "user", "content": "/model"},
            ],
        )
        self.assertEqual(self.vin_app.messages, [])
        self.inline.assert_not_awaited()
        self.save.assert_not_called()
        self.cache_save.assert_not_called()


    def test_rapid_enter_queues_each_prompt_and_preserves_the_latest_unsent_draft(self) -> None:
        self.run_chat(["/quit"])
        bindings = session_bindings(self.session)
        submit = bindings.get_bindings_for_keys((Keys.ControlM,))[-1].handler
        accept = Mock(wraps=self.session.default_buffer.accept_handler)
        buffer = Buffer(document=Document("first\nline"), accept_handler=accept)
        event_app = Application[str](layout=Layout(Window(BufferControl(buffer))), input=DummyInput(), output=DummyOutput())
        exit_mock = Mock()
        event_app.exit = exit_mock
        event = key_event(event_app, Keys.ControlM)

        submit(event)
        self.assertEqual(buffer.text, "")
        buffer.text = "next draft\nline"
        submit(event)

        self.assertEqual(buffer.text, "")
        buffer.document = Document("latest unsent draft", cursor_position=5)
        self.assertEqual(list(self.vin_app.queued_prompts), ["first\nline", "next draft\nline"])
        self.assertEqual(buffer.document, Document("latest unsent draft", cursor_position=5))
        self.assertEqual(accept.call_count, 2)
        exit_mock.assert_not_called()


    def test_model_loading_keeps_draft_editable_and_blocks_submission_until_ready(self) -> None:
        self.run_chat(["/quit"])
        self.assertFalse(self.vin_app.loading_model)
        buffer = self.session.default_buffer
        loading = loading_container(self.session)
        transcript = list(self.vin_app.transcript)

        async def exercise_loading() -> None:
            buffer.document = Document("next draft\nline", cursor_position=5)
            self.vin_app.loading_model = True
            try:
                self.assertTrue(loading.filter())
                rendered = loading_control(self.session).create_content(80, 1)
                text = "".join(fragment[1] for fragment in rendered.get_line(0))
                self.assertIn("Loading models…", text)

                self.vin_app.app.key_processor.feed(KeyPress("x", "x"))
                self.vin_app.app.key_processor.process_keys()
                self.assertEqual(buffer.text, "next xdraft\nline")
                edited = buffer.document

                self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlM))
                self.vin_app.app.key_processor.process_keys()
                self.assertEqual(buffer.document, edited)
                self.assertTrue(self.vin_app.submitted.empty())
                self.assertEqual(self.vin_app.transcript, transcript)
                self.assertEqual(self.vin_app.messages, [])
                self.assertIs(self.session.layout.current_window, self.vin_app.input_window)

                self.vin_app.loading_model = False
                self.assertFalse(loading.filter())
                self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlM))
                self.vin_app.app.key_processor.process_keys()
                self.assertIsNone(self.vin_app.submitted.get_nowait())
                self.assertEqual(list(self.vin_app.queued_prompts), [edited.text])
                self.assertEqual(buffer.text, "")
            finally:
                self.vin_app.loading_model = False
                await self.vin_app.app.cancel_and_wait_for_background_tasks()

        with set_app(self.vin_app.app):
            asyncio.run(exercise_loading())


    def test_information_panel_preserves_draft_history_and_shutdown_cancellation(self) -> None:
        for key in (
            Keys.ControlM, Keys.ControlC, Keys.ControlD, Keys.Escape, "task cancellation"
        ):
            with self.subTest(key=key):
                self.run_chat(["/quit"])
                previous_exchange: History = [
                    {"role": "user", "content": "earlier message"},
                    {"role": "assistant", "content": "earlier answer"},
                ]
                self.vin_app.transcript.extend(validated_transcript(decode_transcript(previous_exchange)))
                self.vin_app.messages.extend(previous_exchange)
                previous_transcript = list(self.vin_app.transcript)
                previous_messages = list(self.vin_app.messages)
                bindings = session_bindings(self.session)
                buffer = self.session.default_buffer
                event_app = Application[str](layout=Layout(Window(BufferControl(buffer))), input=DummyInput(), output=DummyOutput())
                exit_mock = Mock()
                event_app.exit = exit_mock

                async def exercise_panel() -> None:
                    buffer.document = Document("next draft\nline", cursor_position=5)
                    task = asyncio.create_task(
                        self.vin_app.show_panel(
                            vin.Markdown("**Help**\n\nTemporary information")
                        )
                    )
                    try:
                        await REAL_SLEEP(0)
                        if task.done():
                            await task
                        self.assertIs(
                            self.session.layout.current_window,
                            self.vin_app.choice_menu,
                        )
                        self.assertIn(
                            self.vin_app.choice_menu,
                            list(self.session.layout.get_focusable_windows()),
                        )
                        self.assertNotIn(
                            self.vin_app.input_window,
                            list(self.session.layout.find_all_windows()),
                        )
                        panel = self.vin_app.choice_menu
                        assert isinstance(panel, Window)
                        rendered = panel.content.create_content(80, 20)
                        text = "\n".join(
                            "".join(fragment[1] for fragment in rendered.get_line(index))
                            for index in range(rendered.line_count)
                        )
                        self.assertIn("Help", text)
                        self.assertNotIn("**Help**", text)
                        self.assertFalse(rendered.show_cursor)

                        if key == "task cancellation":
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await asyncio.wait_for(task, timeout=1)
                        else:
                            assert isinstance(key, Keys)
                            binding = bindings.get_bindings_for_keys((key,))[-1]
                            self.assertTrue(binding.filter())
                            if key == Keys.Escape:
                                self.assertTrue(binding.eager())
                                self.vin_app.app.key_processor.feed(KeyPress(Keys.Escape))
                                self.vin_app.app.key_processor.process_keys()
                            else:
                                binding.handler(
                                    key_event(event_app, key)
                                )
                            self.assertIsNone(await asyncio.wait_for(task, timeout=1))

                        exit_mock.assert_not_called()
                        self.assertIs(
                            self.session.layout.current_window,
                            self.vin_app.input_window,
                        )
                        self.assertEqual(
                            (buffer.text, buffer.cursor_position),
                            ("next draft\nline", 5),
                        )
                        self.assertEqual(self.vin_app.transcript, previous_transcript)
                        self.assertEqual(self.vin_app.messages, previous_messages)
                        self.assertIsNone(self.vin_app.choice_menu)
                        self.assertIsNone(self.vin_app.choice_future)
                    finally:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        await self.vin_app.app.cancel_and_wait_for_background_tasks()

                with set_app(self.vin_app.app):
                    asyncio.run(exercise_panel())


    def test_escape_enter_still_inserts_newline_when_no_panel_is_open(self) -> None:
        self.run_chat(["/quit"])
        buffer = self.session.default_buffer

        async def exercise_newline() -> None:
            try:
                buffer.document = Document("draft", cursor_position=5)
                self.vin_app.app.key_processor.feed(KeyPress(Keys.Escape))
                self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlM))
                self.vin_app.app.key_processor.process_keys()

                self.assertEqual(buffer.text, "draft\n")
                self.assertEqual(buffer.cursor_position, 6)
                self.assertTrue(self.vin_app.submitted.empty())
                self.assertIsNone(self.vin_app.choice_future)
                self.assertIs(self.session.layout.current_window, self.vin_app.input_window)
            finally:
                await self.vin_app.app.cancel_and_wait_for_background_tasks()

        with set_app(self.vin_app.app):
            asyncio.run(exercise_newline())

    def test_first_run_uses_inline_sign_in_and_model_selection(self) -> None:
        self.credentials = None
        self.load.return_value = None
        self.inline.side_effect = ["chatgpt", "new-model"]
        self.run_chat(["/quit"])

        self.assertEqual(self.inline.await_args_list, [
            call("Sign in to Vin", [("chatgpt", "Sign in with ChatGPT")]),
            call(
                "Choose a ChatGPT model",
                [("old-model", "Old Model"), ("new-model", "New Model")],
            ),
        ])
        self.login_picker.assert_not_called()
        self.choose.assert_not_called()
        self.assertEqual(self.vin_app.credentials, self.signed_in)
        self.save.assert_called_once_with("chatgpt", "new-model")
        self.cache_save.assert_called_once_with(self.signed_in, self.models)
        self.assert_model_indicator(" ChatGPT · New Model")
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.assertEqual(self.vin_app.messages, [])


    def test_ctrl_c_signals_browser_worker_and_discards_late_result(self) -> None:
        self.credentials = None
        ui_thread = get_ident()
        context: SignInContext = {}

        async def choose_sign_in(message: str, options: Sequence[tuple[str, str]], default: str | None = None) -> str:
            context["loop"] = asyncio.get_running_loop()
            return "chatgpt"

        def cancel_from_ui() -> None:
            buffer = self.session.default_buffer
            buffer.document = Document("keep this draft", cursor_position=4)
            self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlC))
            self.vin_app.app.key_processor.process_keys()

        def sign_in(registration: Credentials | None, *, cancel_event: Event, notify: Callable[[str], None]) -> Credentials:
            self.assertNotEqual(get_ident(), ui_thread)
            context["cancel_event"] = cancel_event
            context["notify"] = notify
            present(context.get("loop")).call_soon_threadsafe(cancel_from_ui)
            if not cancel_event.wait(2):
                raise RuntimeError("The browser worker was not cancelled.")
            notify("Late sign-in success")
            return self.signed_in

        self.inline.side_effect = choose_sign_in
        self.login.side_effect = sign_in
        self.run_chat(["/quit"])

        self.assertTrue(present(context.get("cancel_event")).is_set())
        self.assertIsNone(self.vin_app.credentials)
        self.assertIsNone(self.vin_app.selected_model)
        self.assertEqual(self.vin_app.models, [])
        self.assertFalse(self.vin_app.loading_model)
        buffer = self.session.default_buffer
        self.assertEqual((buffer.text, buffer.cursor_position), ("keep this draft", 4))
        self.assertEqual(len(self.vin_app.transcript), 1)
        self.assertEqual(self.vin_app.messages, [])
        mock_prompt(self.session).assert_not_awaited()
        self.catalog.assert_not_called()
        self.save.assert_not_called()

    def test_browser_sign_in_failure_returns_to_input_with_notice(self) -> None:
        self.credentials = None
        self.inline.return_value = "chatgpt"

        def failed_sign_in(registration: Credentials | None, *, cancel_event: Event, notify: Callable[[str], None]) -> None:
            notify("Could not open the sign-in browser")
            return None

        self.login.side_effect = failed_sign_in
        self.run_chat(["/quit"])

        self.assertEqual(mock_prompt(self.session).await_count, 1)
        self.assertIn(
            "Could not open the sign-in browser",
            [str(entry["content"]) for entry in self.vin_app.transcript],
        )
        self.assertIsNone(self.vin_app.credentials)
        self.assertFalse(self.vin_app.loading_model)
        self.assert_model_indicator(" Not signed in · /login")
        self.catalog.assert_not_called()
        self.save.assert_not_called()
        self.console_print.assert_not_called()


    def test_model_command_opens_picker_and_saves_selection(self) -> None:
        async def pick(message: str, options: Sequence[tuple[str, str]], default: str | None = None) -> str:
            self.assertEqual(len(self.vin_app.transcript), 1)
            return "new-model"

        def save(provider: str, model: str) -> None:
            self.assertEqual(len(self.vin_app.transcript), 1)

        self.inline.side_effect = pick
        self.save.side_effect = save
        self.run_chat(["/model", "/quit"])

        self.choose.assert_not_called()
        self.inline.assert_awaited_once_with(
            "Choose a ChatGPT model",
            [("old-model", "Old Model"), ("new-model", "New Model")],
            default="old-model",
        )
        self.save.assert_called_once_with("chatgpt", "new-model")
        self.catalog.assert_not_called()
        self.refresh.assert_not_called()
        entries = transcript_entries(self.session)
        self.assertEqual(
            entries[1:],
            [
                {"role": "user", "content": "/model"},
                {"role": "notice", "content": "Model: ChatGPT · New Model"},
            ],
        )
        self.assertEqual(self.vin_app.messages, [])
        self.console_print.assert_not_called()
        self.assert_model_indicator(" ChatGPT · New Model")


    def test_cancelled_login_chooser_keeps_account_model_conversation_and_draft(self) -> None:
        exchange: History = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        step = 0

        async def choose_sign_in(message: str, options: Sequence[tuple[str, str]], default: str | None = None) -> str:
            self.session.default_buffer.document = Document("next draft", cursor_position=4)
            raise asyncio.CancelledError

        async def read(_: str) -> str:
            nonlocal step
            step += 1
            if step == 1:
                self.vin_app.transcript.extend(validated_transcript(decode_transcript(exchange)))
                self.vin_app.messages.extend(exchange)
                return "/login"
            buffer = self.session.default_buffer
            self.assertEqual((buffer.text, buffer.cursor_position), ("next draft", 4))
            self.assertEqual(self.vin_app.credentials, self.credentials)
            self.assertEqual(self.vin_app.selected_model, "old-model")
            self.assertEqual(self.vin_app.models, self.models)
            return "/quit"

        self.inline.side_effect = choose_sign_in
        self.run_chat(read)

        self.assertEqual(self.vin_app.messages, exchange)
        self.assertEqual(self.vin_app.transcript[1:], exchange)
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.logout.assert_not_called()
        self.login.assert_not_called()
        self.save.assert_not_called()
        self.assertFalse(self.vin_app.loading_model)


    def test_logout_worker_keeps_input_editable_and_routes_warning_inside_workspace(self) -> None:
        context: dict[str, asyncio.AbstractEventLoop] = {}
        ui_thread = get_ident()
        probe_done = Event()
        warning = "[red]Remote revocation was not confirmed.[/red]"
        registration = fixture_registration()

        def probe_ui() -> None:
            try:
                self.assertTrue(self.vin_app.app.is_running)
                self.assertTrue(self.vin_app.loading_model)
                self.assertEqual(self.vin_app.credentials, self.credentials)
                buffer = self.session.default_buffer
                buffer.document = Document("next draft", cursor_position=4)
                self.vin_app.app.key_processor.feed(KeyPress("x", "x"))
                self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlM))
                self.vin_app.app.key_processor.process_keys()
                self.assertEqual((buffer.text, buffer.cursor_position), ("nextx draft", 5))
                self.assertTrue(self.vin_app.submitted.empty())
                text = "".join(
                    fragment[1]
                    for fragment in to_formatted_text(loading_control(self.session).text)
                )
                self.assertIn("Signing out", text)
            finally:
                probe_done.set()

        def sign_out(saved: Credentials, *, notify: Callable[[str], None] = print) -> Credentials:
            self.assertNotEqual(get_ident(), ui_thread)
            saved.pop("access_token", None)
            saved.update(registration)
            notify(warning)
            context["loop"].call_soon_threadsafe(probe_ui)
            if not probe_done.wait(2):
                raise RuntimeError("The UI did not respond during sign-out.")
            return registration

        step = 0

        async def read(_: str) -> str:
            nonlocal step
            step += 1
            if step == 1:
                context["loop"] = asyncio.get_running_loop()
                return "/logout"
            buffer = self.session.default_buffer
            self.assertEqual((buffer.text, buffer.cursor_position), ("nextx draft", 5))
            return "/quit"

        self.logout.side_effect = sign_out
        output = StringIO()
        with redirect_stdout(output):
            self.run_chat(read)

        self.assertTrue(probe_done.is_set())
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.assertEqual(self.vin_app.credentials, registration)
        self.assertEqual(self.vin_app.models, [])
        self.assertIsNone(self.vin_app.selected_model)
        self.assertEqual(
            [str(entry["content"]) for entry in self.vin_app.transcript[1:]],
            [warning, "Signed out of Vin."],
        )
        self.assertEqual(self.vin_app.messages, [])
        self.assert_model_indicator(" Not signed in · /login")


    def test_cancelled_logout_finishes_cleanup_before_accepting_another_command(self) -> None:
        context: dict[str, asyncio.AbstractEventLoop] = {}
        ui_thread = get_ident()
        release_worker = Event()
        worker_finished = Event()
        step = 0

        async def cancel_from_ui() -> None:
            buffer = self.session.default_buffer
            buffer.document = Document("keep this draft", cursor_position=4)
            self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlC))
            self.vin_app.app.key_processor.process_keys()
            for _ in range(50):
                if self.vin_app.auth_status == "Finishing sign-out…":
                    break
                await REAL_SLEEP(0)
            self.assertEqual(self.vin_app.auth_status, "Finishing sign-out…")
            self.assertTrue(self.vin_app.app.is_running)
            self.assertTrue(self.vin_app.loading_model)
            self.assertEqual(present(self.vin_app.pending_auth).cancelling(), 1)
            self.vin_app.app.key_processor.feed(KeyPress(Keys.ControlC))
            self.vin_app.app.key_processor.process_keys()
            release_worker.set()

        def sign_out(saved: Credentials, *, notify: Callable[[str], None] = print) -> Credentials:
            self.assertNotEqual(get_ident(), ui_thread)
            context["loop"].call_soon_threadsafe(
                self.vin_app.app.create_background_task, cancel_from_ui()
            )
            if not release_worker.wait(2):
                raise RuntimeError("Sign-out cleanup was abandoned.")
            saved.pop("access_token", None)
            saved.update(fixture_registration())
            worker_finished.set()
            return saved

        async def read(_: str) -> str:
            nonlocal step
            step += 1
            if step == 1:
                context["loop"] = asyncio.get_running_loop()
                return "/logout"
            self.assertTrue(worker_finished.is_set())
            buffer = self.session.default_buffer
            self.assertEqual((buffer.text, buffer.cursor_position), ("keep this draft", 4))
            self.assertFalse(self.vin_app.loading_model)
            self.assertIsNone(self.vin_app.pending_auth)
            return "/quit"

        self.logout.side_effect = sign_out
        try:
            self.run_chat(read)
        finally:
            release_worker.set()

        self.assertEqual(step, 2)
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)
        self.assertEqual(self.vin_app.models, [])
        self.assertIsNone(self.vin_app.selected_model)
        self.assertEqual(self.vin_app.credentials, fixture_registration())
        self.assertEqual(self.vin_app.messages, [])


    def test_help_renders_only_in_panel_and_preserves_conversation(self) -> None:
        snapshots: list[list[TranscriptEntry]] = []
        panels: list[RenderableType] = []
        commands = iter(["/help", "/quit"])
        exchange: History = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        show_panel = vin.VinApp.show_panel

        async def observe_panel(instance: vin.VinApp, content: RenderableType) -> None:
            panels.append(content)
            buffer = self.session.default_buffer
            buffer.document = Document("next draft\nline", cursor_position=5)
            await show_panel(instance, content)
            self.assertEqual(
                (buffer.text, buffer.cursor_position), ("next draft\nline", 5)
            )
            self.assertIs(self.session.layout.current_window, instance.input_window)
            self.assertIsNone(instance.choice_menu)
            self.assertIsNone(instance.choice_future)

        async def read(_: str) -> str:
            if not snapshots:
                self.vin_app.transcript.extend(validated_transcript(decode_transcript(exchange)))
                self.vin_app.messages.extend(exchange)
            entries = transcript_entries(self.session)
            snapshots.append(list(entries))
            return next(commands)

        with patch.object(vin.VinApp, "show_panel", observe_panel):
            self.run_chat(read)

        self.assertEqual(len(panels), 1)
        panel_text = "".join(
            fragment[1]
            for fragment in to_formatted_text(
                transcript_ui.render_transcript(
                    [{"role": "notice", "content": panels[0]}], 80, 20
                )
            )
        )
        self.assertEqual(panel_text.count("Commands"), 1)
        self.assertIn("/model", panel_text)
        self.assertNotIn("**Commands**", panel_text)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(self.vin_app.transcript, snapshots[0])
        self.assertEqual(self.vin_app.messages, exchange)
        self.console_print.assert_not_called()
        self.console_clear.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(mock_run(self.vin_app.app).await_count, 1)


    def test_failed_model_save_returns_to_input(self) -> None:
        self.save.side_effect = RuntimeError("Could not save model preference")
        self.run_chat(["/model", "/quit"])

        self.assertEqual(mock_prompt(self.session).await_count, 2)
        self.assert_error_notice("Could not save model preference", command="/model")
        self.assert_model_indicator(" ChatGPT · Old Model")
