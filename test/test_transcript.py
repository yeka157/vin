"""Transcript rendering: live activity, headings, diffs, tool groups and tool status."""

import asyncio
from collections.abc import Callable, Sequence
from io import StringIO
from queue import SimpleQueue
from threading import Event
from types import NotImplementedType
from typing import Literal
import unittest
from unittest.mock import patch

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.input import DummyInput
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from test.tui_typing_helpers import key_event
from vin.provider.events import ReplyEvents
from vin.types.events import ReplyEvent, ToolEventKind
from vin.tui import app as app_module, bindings, layout, transcript
from vin.tui import turns as turns_module
from vin.tui.app import VinApp
from vin.tui.transcript import TranscriptControl, print_assistant_message, render_transcript
from vin.types.auth import Credentials
from vin.types.history import History
from vin.types.transcript import ToolEntry, TranscriptEntry


class ActivityTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.owner = app_module.VinApp()
        self.owner.session = PromptSession(input=DummyInput(), output=DummyOutput())
        self.owner.session.default_buffer = Buffer(document=Document("next draft", cursor_position=3))
        self.now = 100.0
        for module in (app_module, turns_module, bindings, layout, transcript):
            self.enterContext(patch.object(module, "monotonic", side_effect=lambda: self.now, create=True))

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def start_turn(self, text: str = "Inspect the file") -> tuple[asyncio.Task[Literal["error", "rate_limit"] | None], ReplyEvents]:
        entered, release, stopped = Event(), Event(), Event()
        queues: SimpleQueue[ReplyEvents] = SimpleQueue()

        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
            try:
                queues.put(events)
                entered.set()
                release.wait(5)
            finally:
                stopped.set()

        self.enterContext(patch.object(turns_module, "stream_chatgpt_response", side_effect=worker))
        self.owner.transcript.append({"role": "user", "content": text})
        task = asyncio.create_task(self.owner.stream_reply(text))
        self.owner.pending_reply = task

        async def cleanup() -> None:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.wait_until(stopped.is_set)

        self.addAsyncCleanup(cleanup)
        await self.wait_until(entered.is_set)
        return task, queues.get_nowait()

    async def emit(self, events: ReplyEvents, *items: ReplyEvent) -> None:
        for item in items:
            if item[0] == "tool_started":
                events.tools_used.set()
            events.put(item)
        await self.wait_until(events.empty)

    def visible(self, width: int = 100) -> str:
        content = self.owner.transcript_control.create_content(width, 500)
        return "\n".join(
            "".join(fragment[1] for fragment in content.get_line(line))
            for line in range(content.line_count)
        )

    def assert_live_tail(self, after: str, elapsed: str) -> str:
        text = self.visible()
        self.assertEqual(text.count("Thinking…"), 1, "Keep exactly one live indicator in the transcript.")
        self.assertIn(f"Thinking… ({elapsed})", text)
        self.assertGreater(text.index("Thinking…"), text.index(after))
        self.assertNotIn("Worked for", text, "The completed footer must not appear during work.")
        self.assertEqual(self.owner.session.default_buffer.text, "next draft")
        return text


    async def test_live_indicator_stays_below_streaming_text_and_tool_activity(self) -> None:
        _, events = await self.start_turn()
        await self.emit(events, ("text", "Checking the workspace."))
        self.now = 102.0
        self.assert_live_tail("Checking the workspace.", "2s")
        await self.emit(events, ("tool_started", "read"))
        self.now = 104.0
        self.assert_live_tail("read", "4s")
        await self.emit(events, ("tool_finished", "read"), ("text", "The file is valid."))
        self.now = 106.0
        self.assert_live_tail("The file is valid.", "6s")

    async def test_completed_turn_replaces_indicator_with_one_frozen_footer(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("text", "Verified."))
        self.now = 1542.0
        await self.emit(events, ("complete", None))
        await asyncio.wait_for(task, 2)
        finished = self.visible()
        self.assertNotIn("Thinking…", finished)
        self.assertEqual(finished.count("Worked for 24m 2s"), 1)
        self.assertGreater(finished.index("Worked for"), finished.index("Verified."))
        self.assertRegex(finished, r"done \d{1,2}:\d{2} [AP]M")
        self.now = 2000.0
        self.assertEqual(self.visible(), finished, "Finished timing must not keep ticking.")


    async def test_failed_tool_and_request_do_not_show_a_success_footer(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", "read"), ("tool_failed", "read"))
        self.now = 107.0
        await self.emit(events, ("error", "The request failed."))
        await asyncio.wait_for(task, 2)
        text = self.visible().lower()
        self.assertIn("1 tool call · 1 failed", text)
        self.assertIn("failed", text)
        self.assertIn("7s", text)
        self.assertNotIn("worked for", text)
        self.assertNotIn("thinking…", text)


    async def test_keyboard_cancellation_freezes_interruption_footer_and_discards_late_events(self) -> None:
        task, events = await self.start_turn()
        await self.emit(events, ("tool_started", "bash"))
        self.now = 109.0
        keys = bindings.create_bindings(self.owner)
        keys.get_bindings_for_keys((Keys.ControlC,))[-1].handler(
            key_event(self.owner.session.app, Keys.ControlC)
        )
        with self.assertRaises(asyncio.CancelledError):
            await task
        cancelled = self.visible()
        self.assertIn("interrupted", cancelled.lower())
        self.assertEqual(cancelled.count("Interrupted after"), 1)
        self.assertIn("9s", cancelled)
        self.assertNotIn("Worked for", cancelled)
        self.assertNotIn("Thinking…", cancelled)
        events.put(("tool_finished", "bash"))
        events.put(("text", "Late result must stay hidden."))
        self.now = 300.0
        self.assertEqual(self.visible(), cancelled)
        self.assertEqual(self.owner.session.default_buffer.document, Document("next draft", cursor_position=3))


class HeadingTests(unittest.TestCase):
    def test_assistant_headings_start_at_the_left_edge(self) -> None:
        output = StringIO()
        console = Console(file=output, width=40, color_system=None)
        print_assistant_message(console, "# Vin\n\n## Two")
        lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines, [" Vin", " Two"])


class DiffRenderingTests(unittest.TestCase):
    def row(self, diff: str) -> ToolEntry:
        return {"role": "tool", "name": "edit", "detail": "f", "status": "finished", "diff": diff,
                "expanded": True,
                "content": Text("● edit f · finished", style="dim")}

    def render(self, entries: Sequence[TranscriptEntry], width: int = 80) -> tuple[FormattedText, str]:
        fragments = to_formatted_text(render_transcript(entries, width, 100))
        return fragments, "".join(fragment[1] for fragment in fragments)


    def test_diff_preserves_literal_markup_leading_indentation_and_newline_marker(self) -> None:
        diff = ("--- f\n+++ f\n@@ -1,2 +1,2 @@\n     context\n"
                "-    [red]before[/red]\n+    **after**\n\\ No newline at end of file\n")
        _, text = self.render([self.row(diff)])
        lines = [line.rstrip() for line in text.splitlines()]
        self.assertIn("  2 │ -     [red]before[/red]", lines)
        self.assertIn("  2 │ +     **after**", lines)
        self.assertIn("  1 │       context", lines)
        self.assertIn("\\ No newline at end of file", text)

    def test_long_diff_lines_wrap_without_losing_text(self) -> None:
        addition = "+" + "abcdefghij" * 8
        _, text = self.render([self.row("--- f\n+++ f\n@@ -1 +1 @@\n-old\n" + addition + "\n")], width=24)
        self.assertIn(addition, "".join(text.split()))
        self.assertTrue(all(len(line) <= 24 for line in text.splitlines()))


    def test_malformed_diff_falls_back_to_literal_text_without_invented_numbers(self) -> None:
        diff = "--- f\n+++ f\n@@ broken @@\n-[red]before[/red]\n+**after**\n"
        _, text = self.render([self.row(diff)])
        self.assertIn("@@ broken @@", text)
        self.assertIn("-[red]before[/red]", text)
        self.assertIn("+**after**", text)
        self.assertNotIn("│", text)


class ToolGroupClickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first: ToolEntry = {
            "role": "tool", "name": "read", "detail": "src/first.py",
            "status": "finished", "content": Text("● read src/first.py · finished"),
        }
        self.entries: list[TranscriptEntry] = [self.first]
        self.control = TranscriptControl(lambda: self.entries)

    def render(self, width: int = 100, height: int = 100) -> list[str]:
        content = self.control.create_content(width, height)
        return ["".join(part[1] for part in content.get_line(line))
                for line in range(content.line_count)]

    def mouse(self, y: int, event_type: MouseEventType = MouseEventType.MOUSE_UP, button: MouseButton = MouseButton.LEFT) -> NotImplementedType | None:
        return self.control.mouse_handler(MouseEvent(
            position=Point(x=1, y=y), event_type=event_type,
            button=button, modifiers=frozenset(),
        ))

    def test_left_release_opens_then_closes_without_double_toggling_on_press(self) -> None:
        self.render()
        self.assertIsNone(self.mouse(0, MouseEventType.MOUSE_DOWN))
        self.assertFalse(self.first.get("expanded", False))
        self.assertIsNone(self.mouse(0))
        self.assertIs(self.first.get("expanded"), True)
        self.assertIn("src/first.py", "\n".join(self.render()))
        self.assertIsNone(self.mouse(0))
        self.assertIs(self.first.get("expanded"), False)
        self.assertNotIn("src/first.py", "\n".join(self.render()))


    def test_window_routes_clicks_without_changing_composer_draft_or_focus(self) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.current import create_app_session, set_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.layout import BufferControl, HSplit, Layout, Window
        from prompt_toolkit.layout.mouse_handlers import MouseHandlers
        from prompt_toolkit.layout.screen import Screen, WritePosition
        from prompt_toolkit.output import DummyOutput

        async def exercise() -> None:
            with create_app_session(input=DummyInput(), output=DummyOutput()):
                draft = Document("next unsent\ndraft", cursor_position=4)
                buffer = Buffer(document=draft)
                composer = BufferControl(buffer=buffer)
                pane = Window(content=self.control)
                root = HSplit([pane, Window(content=composer, height=3)])
                app: Application[None] = Application(layout=Layout(root, focused_element=composer), mouse_support=True)

                with set_app(app):
                    try:
                        def render_screen() -> tuple[Screen, MouseHandlers, WritePosition]:
                            app.render_counter += 1
                            screen, handlers = Screen(), MouseHandlers()
                            root.write_to_screen(
                                screen, handlers, WritePosition(0, 0, 60, 14), "", True, None,
                            )
                            # Application._redraw refreshes these after painting;
                            # mouse routing uses them to find the active modal area.
                            app.layout.update_parents_relations()
                            bounds = screen.visible_windows_to_write_positions[pane]
                            return screen, handlers, bounds

                        for marker, expected in (("▸", True), ("▾", False)):
                            screen, handlers, bounds = render_screen()
                            y = next(
                                y for y in range(bounds.ypos, bounds.ypos + bounds.height)
                                if screen.data_buffer[y][bounds.xpos].char == marker
                            )
                            position = Point(x=bounds.xpos + 1, y=y)
                            result = handlers.mouse_handlers[y][position.x](MouseEvent(
                                position=position, event_type=MouseEventType.MOUSE_UP,
                                button=MouseButton.LEFT, modifiers=frozenset(),
                            ))
                            self.assertIsNone(result)
                            self.assertIs(self.first.get("expanded"), expected)
                            self.assertEqual(buffer.document, draft)
                            self.assertIs(app.layout.current_control, composer)
                    finally:
                        await app.cancel_and_wait_for_background_tasks()

        asyncio.run(exercise())


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries: list[TranscriptEntry] = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "alpha beta gamma\n\ndelta epsilon"},
        ]
        self.copied: list[str] = []
        self.control = TranscriptControl(lambda: self.entries, self.copied.append)

    def render(self, width: int = 60, height: int = 40) -> list[StyleAndTextTuples]:
        content = self.control.create_content(width, height)
        return [content.get_line(line) for line in range(content.line_count)]

    def text_of(self, line: StyleAndTextTuples) -> str:
        return "".join(part[1] for part in line)

    def find(self, word: str) -> tuple[int, int]:
        """(row, column) of a word on screen."""
        for row, line in enumerate(self.render()):
            text = self.text_of(line)
            if word in text:
                return row, text.index(word)
        raise AssertionError(f"{word!r} is not on screen")

    def mouse(self, kind: MouseEventType, row: int, column: int, button: MouseButton = MouseButton.LEFT) -> NotImplementedType | None:
        return self.control.mouse_handler(MouseEvent(
            position=Point(x=column, y=row), event_type=kind, button=button, modifiers=frozenset()))

    def drag(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self.assertIsNone(self.mouse(MouseEventType.MOUSE_DOWN, *start))
        self.assertIsNone(self.mouse(MouseEventType.MOUSE_MOVE, *end))
        self.assertIsNone(self.mouse(MouseEventType.MOUSE_UP, *end))

    def highlighted(self) -> str:
        return "".join(part[1] for line in self.render() for part in line if "reverse" in part[0])

    def test_drag_within_a_line_copies_and_highlights_the_cells(self) -> None:
        row, column = self.find("beta")
        self.drag((row, column), (row, column + len("beta gamma") - 1))
        self.assertEqual(self.copied, ["beta gamma"])
        self.assertEqual(self.highlighted(), "beta gamma")

    def test_drag_across_lines_copies_each_line_and_drops_trailing_spaces(self) -> None:
        start = self.find("gamma")
        end_row, end_column = self.find("delta")
        self.drag(start, (end_row, end_column + len("delta") - 1))
        self.assertEqual(self.copied, ["gamma\n\ndelta"])

    def test_dragging_backwards_selects_the_same_text(self) -> None:
        row, column = self.find("beta")
        self.drag((row, column + len("beta gamma") - 1), (row, column))
        self.assertEqual(self.copied, ["beta gamma"])

    def test_highlight_shows_while_dragging_before_release(self) -> None:
        row, column = self.find("alpha")
        _ = self.mouse(MouseEventType.MOUSE_DOWN, row, column)
        _ = self.mouse(MouseEventType.MOUSE_MOVE, row, column + 4)
        self.assertEqual(self.highlighted(), "alpha")
        self.assertEqual(self.copied, [])

    def test_a_click_without_dragging_copies_nothing_and_clears_the_selection(self) -> None:
        row, column = self.find("alpha")
        self.drag((row, column), (row, column + 4))
        self.assertIsNone(self.mouse(MouseEventType.MOUSE_DOWN, row, column))
        self.assertIs(self.mouse(MouseEventType.MOUSE_UP, row, column), NotImplemented)
        self.assertEqual(self.copied, ["alpha"])
        self.assertEqual(self.highlighted(), "")

    def test_moving_without_a_pressed_button_selects_nothing(self) -> None:
        row, column = self.find("alpha")
        self.assertIs(self.mouse(MouseEventType.MOUSE_MOVE, row, column, MouseButton.NONE), NotImplemented)
        self.assertEqual(self.highlighted(), "")

    def test_selection_follows_the_text_when_scrolling(self) -> None:
        self.entries.extend({"role": "assistant", "content": f"line {number}"} for number in range(60))
        content = self.control.create_content(60, 10)
        self.control.scroll(-1000)
        content = self.control.create_content(60, 10)
        row, column = self.find_in(content, "alpha")
        self.drag((row, column), (row, column + 4))
        self.control.scroll(1)
        content = self.control.create_content(60, 10)
        highlighted = "".join(part[1] for line in range(content.line_count)
                              for part in content.get_line(line) if "reverse" in part[0])
        self.assertEqual(highlighted, "alpha")

    def find_in(self, content: UIContent, word: str) -> tuple[int, int]:
        for row in range(content.line_count):
            text = self.text_of(content.get_line(row))
            if word in text:
                return row, text.index(word)
        raise AssertionError(f"{word!r} is not on screen")

    def test_dragging_on_the_last_row_scrolls_down(self) -> None:
        self.entries.extend({"role": "assistant", "content": f"line {number}"} for number in range(60))
        _ = self.control.create_content(60, 10)
        self.control.scroll(-1000)
        _ = self.control.create_content(60, 10)
        _ = self.mouse(MouseEventType.MOUSE_DOWN, 0, 2)
        _ = self.mouse(MouseEventType.MOUSE_MOVE, 9, 2)
        self.assertEqual(self.control.top_line, 1)

    def test_indented_code_keeps_its_indentation_relative_to_the_margin(self) -> None:
        self.entries[1] = {"role": "assistant", "content": "```\ndef f():\n    return 1\n```"}
        def_row, _ = self.find("def f")
        return_row, return_column = self.find("return 1")
        self.drag((def_row, 0), (return_row, return_column + len("return 1") - 1))
        self.assertEqual(self.copied, ["def f():\n    return 1"])

    def test_resizing_clears_the_selection(self) -> None:
        row, column = self.find("alpha")
        self.drag((row, column), (row, column + 4))
        _ = self.render(width=40)
        self.assertEqual(self.highlighted(), "")

    def test_wide_characters_are_selected_by_screen_column(self) -> None:
        self.entries[1] = {"role": "assistant", "content": "中文 text"}
        row, column = self.find("中文")
        self.drag((row, column), (row, column + 3))
        self.assertEqual(self.copied, ["中文"])


class ToolGroupRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = VinApp()

    def tool(self, name: str, detail: str, status: Literal["running", "finished", "failed", "interrupted"] = "finished") -> ToolEntry:
        return {
            "role": "tool", "name": name, "detail": detail, "status": status,
            "content": Text(f"● {name} {detail} · {status}", style="dim"),
        }

    def visible(self, width: int = 100) -> str:
        fragments = to_formatted_text(
            render_transcript(self.owner.get_transcript_entries(), width, 100)
        )
        return "".join(fragment[1] for fragment in fragments)

    def test_active_group_keeps_completed_calls_visible_until_next_phase(self) -> None:
        self.owner.request_started_at = 1.0
        self.owner.transcript.extend([
            self.tool("read", "src/first.py"),
            self.tool("bash", "python -m unittest", "running"),
        ])
        text = self.visible()
        self.assertIn("read src/first.py · finished", text)
        self.assertIn("bash python -m unittest · running", text)
        self.assertGreater(text.index("Thinking…"), text.index("python -m unittest"))
        self.assertNotIn("2 tool calls", text)

    def test_new_assistant_update_collapses_group_while_turn_remains_active(self) -> None:
        self.owner.request_started_at = 1.0
        self.owner.transcript.extend([
            self.tool("read", "src/first.py"),
            self.tool("bash", "exit 7", "failed"),
        ])
        self.assertIn("src/first.py", self.visible())
        self.owner.transcript.append({"role": "assistant", "content": "Checking the failure."})
        text = self.visible()
        self.assertIn("2 tool calls · 1 finished · 1 failed", text)
        self.assertNotIn("src/first.py", text)
        self.assertNotIn("exit 7", text)
        self.assertLess(text.index("2 tool calls"), text.index("Checking the failure."))
        self.assertGreater(text.index("Thinking…"), text.index("Checking the failure."))
        self.assertEqual(len(self.owner.transcript), 3, "Collapse must retain original records.")


    def test_manually_reopened_closed_group_reveals_retained_diff(self) -> None:
        row = self.tool("edit", "src/example.py")
        row["diff"] = "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"
        self.owner.transcript.append(row)
        self.assertNotIn("+ new", self.visible())
        row["expanded"] = True
        text = self.visible()
        self.assertIn("▾ 1 tool call · 1 finished", text)
        self.assertIn("src/example.py", text)
        self.assertIn("- old", text)
        self.assertIn("+ new", text)
        self.assertLess(text.index("▾ 1 tool call"), text.index("src/example.py"))
        row["expanded"] = False
        collapsed = self.visible()
        self.assertIn("▸ 1 tool call · 1 finished", collapsed)
        self.assertNotIn("+ new", collapsed)
        self.assertIn("+new", row["diff"], "Closing details must retain the diff.")

    def test_opening_one_group_leaves_other_finished_group_collapsed(self) -> None:
        first = self.tool("read", "src/first.py")
        first["expanded"] = True
        self.owner.transcript.extend([
            first,
            {"role": "assistant", "content": "Next phase."},
            self.tool("bash", "python -m unittest"),
        ])
        text = self.visible()
        self.assertIn("src/first.py", text)
        self.assertNotIn("python -m unittest", text)
        self.assertEqual(text.count("▾ 1 tool call · 1 finished"), 1)
        self.assertEqual(text.count("▸ 1 tool call · 1 finished"), 1)


class ToolStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.owner = app_module.VinApp()
        self.assertTrue(hasattr(self.owner, "tool_status"), "Initialize VinApp.tool_status to None.")
        self.assertIsNone(self.owner.tool_status)
        self.draft = Document("next draft", cursor_position=3)
        self.owner.session = PromptSession(input=DummyInput(), output=DummyOutput())
        self.owner.session.default_buffer = Buffer(document=self.draft)

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    def start(self) -> asyncio.Task[str | None]:
        task = asyncio.create_task(self.owner.stream_reply("Inspect the file"))
        self.owner.pending_reply = task
        return task

    async def test_repeated_edits_retain_their_own_diffs_separately_from_assistant_text(self) -> None:
        first = "--- f\n+++ f\n@@ -1 +1 @@\n-before\n+middle\n"
        second = "--- f\n+++ f\n@@ -1 +1 @@\n-middle\n+after\n"

        def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
            events.tools_used.set()
            for diff in (first, second):
                events.put(("tool_started", {"name": "edit", "detail": "f"}))
                events.put(("tool_diff", diff))
                events.put(("tool_finished", "edit"))
            events.put(("text", "Done"))
            events.put(("complete", None))

        with patch.object(turns_module, "stream_chatgpt_response", side_effect=worker):
            await asyncio.wait_for(self.start(), 2)
        rows = [row for row in self.owner.transcript if row["role"] == "tool"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.get("diff") for row in rows], [first, second])
        self.assertTrue(all(str(row["content"]) == "● edit f · finished" for row in rows))
        answers = [row["content"] for row in self.owner.transcript if row["role"] == "assistant"]
        self.assertEqual(answers, ["Done"])
        self.assertEqual(self.owner.messages[-1]["content"], "Done")
        self.assertEqual(self.owner.session.default_buffer.document, self.draft)


    async def test_running_and_terminal_status_preserve_editable_draft(self) -> None:
        terminals: list[tuple[Literal["tool_finished", "tool_failed"], str]] = [("tool_finished", "finished"), ("tool_failed", "failed")]
        terminal: ToolEventKind
        for terminal, expected in terminals:
            with self.subTest(terminal=terminal):
                previous_rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                release_status, release_answer, stopped = Event(), Event(), Event()
                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
                    try:
                        events.tools_used.set()
                        events.put(("tool_started", "read"))
                        release_status.wait(2)
                        terminal_event: tuple[ToolEventKind, str] = (terminal, "read")
                        events.put(terminal_event)
                        release_answer.wait(2)
                        events.put(("text", "Done"))
                        events.put(("complete", None))
                    finally:
                        stopped.set()
                with patch.object(turns_module, "stream_chatgpt_response", side_effect=worker):
                    task = self.start()
                    try:
                        await self.wait_until(lambda: self.owner.tool_status == ("read", "running"))
                        rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                        self.assertEqual(len(rows), len(previous_rows) + 1)
                        tool_row = rows[-1]
                        self.assertEqual(tool_row.get("status"), "running")
                        self.assertIn("read", str(tool_row["content"]))
                        self.assertIn("running", str(tool_row["content"]))
                        self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                        typed = Document("typing while tool runs", cursor_position=5)
                        self.owner.session.default_buffer.set_document(typed)
                        release_status.set()
                        await self.wait_until(lambda: self.owner.tool_status == ("read", expected))
                        rows = [entry for entry in self.owner.transcript if entry["role"] == "tool"]
                        self.assertEqual(len(rows), len(previous_rows) + 1, "Update the existing row rather than appending another.")
                        self.assertIs(rows[-1], tool_row)
                        self.assertIn(expected, str(tool_row["content"]))
                        self.assertEqual(tool_row.get("status"), expected)
                        self.assertNotIn("running", str(tool_row["content"]))
                        self.assertEqual(self.owner.session.default_buffer.document, typed)
                        release_answer.set()
                        await asyncio.wait_for(task, 2)
                        self.assertEqual(self.owner.tool_status, ("read", expected))
                        self.assertEqual(self.owner.session.default_buffer.document, typed)
                    finally:
                        release_status.set()
                        release_answer.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        await self.wait_until(stopped.is_set)
                        self.owner.session.default_buffer.set_document(self.draft)

    async def test_task_and_keyboard_cancellation_finish_running_status(self) -> None:
        for keyboard in (False, True):
            with self.subTest(keyboard=keyboard):
                release, stopped = Event(), Event()
                def worker(credentials: Credentials | None, model: str | None, messages: History, events: ReplyEvents, cancel_event: Event, **_options: object) -> None:
                    try:
                        events.tools_used.set()
                        events.put(("tool_started", {"name": "bash", "detail": "printf '[red]hello[/red]'"}))
                        release.wait(2)
                        events.put(("tool_finished", "bash"))
                    finally:
                        stopped.set()
                with patch.object(turns_module, "stream_chatgpt_response", side_effect=worker):
                    task = self.start()
                    try:
                        await self.wait_until(lambda: self.owner.tool_status == ("bash", "running"))
                        tool_row = [entry for entry in self.owner.transcript if entry["role"] == "tool"][-1]
                        if keyboard:
                            keys = bindings.create_bindings(self.owner)
                            keys.get_bindings_for_keys((Keys.ControlC,))[-1].handler(
                                key_event(self.owner.session.app, Keys.ControlC))
                        else:
                            task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        self.assertEqual(self.owner.tool_status, ("bash", "failed"))
                        self.assertIn("interrupted", str(tool_row["content"]))
                        self.assertEqual(tool_row.get("status"), "interrupted")
                        self.assertIn("printf '[red]hello[/red]'", str(tool_row["content"]))
                        self.assertNotIn("running", str(tool_row["content"]))
                        content = tool_row["content"]
                        assert isinstance(content, Text)
                        self.assertEqual(content.style, "yellow")
                        interrupted_content = str(tool_row["content"])
                        release.set()
                        await self.wait_until(stopped.is_set)
                        self.assertEqual(self.owner.tool_status, ("bash", "failed"))
                        self.assertEqual(str(tool_row["content"]), interrupted_content)
                        self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        await self.wait_until(stopped.is_set)


if __name__ == "__main__":
    unittest.main()


class RenderCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        transcript._RENDERED.clear()  # pyright: ignore[reportPrivateUsage]
        self.addCleanup(transcript._RENDERED.clear)  # pyright: ignore[reportPrivateUsage]
        self.entries: list[TranscriptEntry] = [
            {"role": "assistant", "content": f"**answer {n}**"} for n in range(5)]

    def count_draws(self, *frames: tuple[list[TranscriptEntry], int]) -> int:
        with patch.object(transcript, "print_assistant_message",
                          wraps=transcript.print_assistant_message) as draw:
            for entries, width in frames:
                transcript.transcript_lines(entries, width, 24)
        return draw.call_count

    def test_finished_entries_are_drawn_once_across_frames(self) -> None:
        self.assertEqual(self.count_draws((self.entries, 80), (self.entries, 80)), 5)

    def test_streaming_entry_is_drawn_every_frame(self) -> None:
        self.entries.append({"role": "assistant", "content": "partial", "streaming": True})
        self.assertEqual(self.count_draws((self.entries, 80), (self.entries, 80)), 5 + 2)

    def test_width_or_content_changes_redraw(self) -> None:
        self.assertEqual(self.count_draws((self.entries, 80), (self.entries, 100)), 10)
        self.entries[0]["content"] = "changed"
        self.assertEqual(self.count_draws((self.entries, 100)), 1)

    def test_cached_lines_match_a_fresh_render(self) -> None:
        first = transcript.render_transcript(self.entries, 80, 24)
        cached = transcript.render_transcript(self.entries, 80, 24)
        transcript._RENDERED.clear()  # pyright: ignore[reportPrivateUsage]
        self.assertEqual(cached, first)
        self.assertEqual(transcript.render_transcript(self.entries, 80, 24), first)

    def test_an_equal_but_different_entry_never_reuses_another_entrys_lines(self) -> None:
        original: TranscriptEntry = {"role": "assistant", "content": "same text"}
        twin: TranscriptEntry = {"role": "assistant", "content": "same text"}
        transcript.transcript_lines([original], 80, 24)
        cache = transcript._RENDERED  # pyright: ignore[reportPrivateUsage]
        owner, key, _lines = cache[id(original)]
        cache[id(twin)] = (owner, key, [[("", "STALE")]])
        rendered = transcript.transcript_lines([twin], 80, 24)
        self.assertNotIn("STALE", str(rendered))

    def test_entries_that_are_gone_leave_the_cache(self) -> None:
        transcript.transcript_lines(self.entries, 80, 24)
        transcript.transcript_lines(self.entries[:2], 80, 24)
        self.assertEqual(set(transcript._RENDERED), {id(entry) for entry in self.entries[:2]})  # pyright: ignore[reportPrivateUsage]

    def test_a_collapsed_tool_group_is_one_header_line_and_one_blank_line(self) -> None:
        tool: ToolEntry = {"role": "tool", "name": "read", "status": "finished", "detail": "a.py",
                           "content": Text("● read a.py · finished")}
        entries: list[TranscriptEntry] = [tool, {"role": "assistant", "content": "Done."}]
        headers: list[tuple[int, int, ToolEntry]] = []
        lines = transcript.transcript_lines(entries, 80, 24, group_headers=headers)
        text = ["".join(fragment[1] for fragment in line) for line in lines]
        self.assertEqual(headers, [(0, 1, tool)])
        self.assertEqual(text[0].strip(), "▸ 1 tool call · 1 finished")
        self.assertEqual(text[1], "")
        self.assertIn("Done.", text[2])
