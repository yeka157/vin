"""Transcript message rendering, viewport scrolling, and mouse handling."""

from io import StringIO
from collections.abc import Callable, Mapping, Sequence
from types import NotImplementedType
from typing import Literal, TypedDict

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import (
    ANSI,
    StyleAndTextTuples,
    split_lines,
    to_formatted_text,
)
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Console
from rich.markdown import Markdown, Heading
from rich.padding import Padding
from rich.text import Text

from vin.tui.diff import print_diff as print_tool_diff
from vin.tui.diff import format_edit_heading
from vin.types import NoticeEntry, ToolEntry, TranscriptEntry


class LeftHeading(Heading):
    LEVEL_ALIGN = {tag: "left" for tag in Heading.LEVEL_ALIGN}


class LeftMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": LeftHeading}


class ToolGroup(TypedDict):
    role: Literal["tool_group"]
    entries: list[ToolEntry]


type GroupedEntry = TranscriptEntry | ToolGroup
type GroupHeader = tuple[int, int, ToolEntry]


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_user_message(console: Console, text: str) -> None:
    console.print(
        Padding(
            Text("› " + text),
            (0, 0),
            style="#ffffff on #393939",
            expand=True,
        )
    )
    console.print()


def print_assistant_message(
    console: Console,
    text: str,
    *,
    streaming: bool = False,
    intermediate: bool = False,
) -> None:
    content = Text(text) if streaming else LeftMarkdown(text, code_theme="ansi_dark")
    console.print(
        Padding(content, (0, 0, 0, 1)),
    )
    console.print()


def group_tool_entries(entries: Sequence[TranscriptEntry]) -> list[GroupedEntry]:
    result: list[GroupedEntry] = []
    current_group: ToolGroup | None = None
    for entry in entries:
        if entry["role"] == "tool":
            if current_group is None:
                current_group = {"role": "tool_group", "entries": []}
                result.append(current_group)
            current_group["entries"].append(entry)
        else:
            result.append(entry)
            current_group = None

    return result


def print_activity_entry(console: Console, entry: ToolEntry | NoticeEntry) -> None:
    heading = None

    if (
        entry["role"] == "tool"
        and entry.get("name") == "edit"
        and entry.get("status") == "finished"
        and entry.get("diff")
    ):
        heading = format_edit_heading(entry.get("detail", ""), entry.get("diff", ""))

    console.print(heading if heading is not None else entry["content"], markup=False)
    if entry["role"] == "tool" and entry.get("diff"):
        print_tool_diff(console, entry.get("diff", ""))
    console.print()


_RENDERED: dict[int, tuple[object, tuple[object, ...], list[StyleAndTextTuples]]] = {}


def _draw(width: int, height: int, draw: Callable[[Console], None]) -> str:
    buffer = StringIO()
    draw(
        Console(
            file=buffer,
            width=width,
            height=height,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
        )
    )
    return buffer.getvalue()


def _lines(text: str) -> list[StyleAndTextTuples]:
    lines: list[StyleAndTextTuples] = []
    for line in split_lines(to_formatted_text(ANSI(text))):
        merged: StyleAndTextTuples = []
        for fragment in line:
            style, piece = fragment[0], fragment[1]
            if merged and merged[-1][0] == style:
                merged[-1] = (style, merged[-1][1] + piece)
            else:
                merged.append((style, piece))
        lines.append(merged)
    if text.endswith("\n"):
        lines.pop()
    return lines


def _draw_entry(
    entry: Mapping[str, object],
    width: int,
    height: int,
    draw: Callable[[Console], None],
    seen: set[int],
) -> list[StyleAndTextTuples]:
    key = (
        width,
        *sorted(
            (
                name,
                (
                    value
                    if isinstance(value, str | int | float | bool | None)
                    else id(value)
                ),
            )
            for name, value in entry.items()
        ),
    )
    seen.add(id(entry))
    cached = _RENDERED.get(id(entry))
    if cached is not None and cached[0] is entry and cached[1] == key:
        return cached[2]
    lines = _lines(_draw(width, height, draw))
    _RENDERED[id(entry)] = (entry, key, lines)
    return lines


def transcript_lines(
    entries: Sequence[TranscriptEntry],
    width: int,
    height: int,
    *,
    group_headers: list[GroupHeader] | None = None,
) -> list[StyleAndTextTuples]:
    if group_headers is not None:
        group_headers.clear()
    parts: list[StyleAndTextTuples] = []
    seen: set[int] = set()

    def add(lines: list[StyleAndTextTuples]) -> None:
        parts.extend(lines)

    grouped_entries = group_tool_entries(entries)
    for index, entry in enumerate(grouped_entries):
        if entry["role"] == "user":
            content = entry["content"]
            add(
                _draw_entry(
                    entry,
                    width,
                    height,
                    lambda console: print_user_message(console, content),
                    seen,
                )
            )
        elif entry["role"] == "assistant":
            assistant = entry

            def draw_assistant(console: Console) -> None:
                print_assistant_message(
                    console,
                    assistant["content"],
                    streaming=assistant.get("streaming", False),
                    intermediate=assistant.get("intermediate", False),
                )

            if assistant.get("streaming", False):
                add(_lines(_draw(width, height, draw_assistant)))
            else:
                add(_draw_entry(assistant, width, height, draw_assistant, seen))
        elif entry["role"] == "tool_group":
            first_tool = entry["entries"][0]
            active = is_tool_group_active(grouped_entries, index)
            expanded = active or first_tool.get("expanded", False)

            if not active:
                summary = format_tool_group_summary(entry["entries"])
                arrow = "▾" if expanded else "▸"
                start_line = len(parts)
                add(
                    _lines(
                        _draw(
                            width,
                            height,
                            lambda console: console.print(
                                Text(f"{arrow} {summary}", style="dim")
                            ),
                        )
                    )
                )
                if group_headers is not None:
                    group_headers.append((start_line, len(parts), first_tool))

            if expanded:
                for tool_entry in entry["entries"]:
                    add(
                        _draw_entry(
                            tool_entry,
                            width,
                            height,
                            lambda console, tool=tool_entry: print_activity_entry(
                                console, tool
                            ),
                            seen,
                        )
                    )
            else:
                add([[]])
        else:
            activity = entry
            add(
                _draw_entry(
                    activity,
                    width,
                    height,
                    lambda console: print_activity_entry(console, activity),
                    seen,
                )
            )

    for stale in _RENDERED.keys() - seen:
        del _RENDERED[stale]
    return parts


def render_transcript(
    entries: Sequence[TranscriptEntry],
    width: int,
    height: int,
) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
    for line in transcript_lines(entries, width, height):
        fragments.extend(line)
        fragments.append(("", "\n"))
    return fragments


class TranscriptControl(UIControl):
    def __init__(self, get_entries: Callable[[], Sequence[TranscriptEntry]]) -> None:
        self.get_entries = get_entries
        self.top_line: int | None = None
        self.line_count = 1
        self.viewport_height = 1
        self.group_headers: list[GroupHeader] = []
        self.visible_start = 0

    def scroll(self, lines: int) -> None:
        last_top_line = max(0, self.line_count - self.viewport_height)
        current_top_line = last_top_line if self.top_line is None else self.top_line
        new_top_line = max(0, min(last_top_line, current_top_line + lines))
        self.top_line = None if new_top_line == last_top_line else new_top_line

    def scroll_to_bottom(self) -> None:
        self.top_line = None

    def mouse_handler(self, mouse_event: MouseEvent) -> NotImplementedType | None:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)
        elif (
            mouse_event.event_type == MouseEventType.MOUSE_UP
            and mouse_event.button == MouseButton.LEFT
        ):
            line = self.visible_start + mouse_event.position.y

            for start, end, first_tool in self.group_headers:
                if start <= line < end:
                    first_tool["expanded"] = not first_tool.get("expanded", False)
                    self.top_line = self.visible_start
                    return None

            return NotImplemented
        else:
            return NotImplemented
        return None

    def create_content(self, width: int, height: int | None) -> UIContent:
        lines = transcript_lines(
            self.get_entries(),
            max(1, width),
            max(1, height or 1),
            group_headers=self.group_headers,
        ) + [[]]
        self.line_count = len(lines)
        self.viewport_height = max(1, height or 1)
        self.scroll(0)
        start = self.top_line
        if start is None:
            start = max(0, self.line_count - self.viewport_height)

        self.visible_start = start
        visible_count = min(self.viewport_height, self.line_count - start)

        return UIContent(
            get_line=lambda line_number: lines[start + line_number],
            line_count=visible_count,
            cursor_position=Point(x=0, y=max(0, visible_count - 1)),
            show_cursor=False,
        )


def format_tool_group_summary(entries: Sequence[ToolEntry]) -> str:
    counts = {
        "running": 0,
        "finished": 0,
        "failed": 0,
        "interrupted": 0,
    }

    for entry in entries:
        status = entry["status"]
        counts[status] += 1
    total = len(entries)
    label = "tool call" if total == 1 else "tool calls"
    parts = [f"{total} {label}"]

    for status, count in counts.items():
        if count > 0:
            parts.append(f"{count} {status}")

    return " · ".join(parts)


def is_tool_group_active(grouped_entries: Sequence[GroupedEntry], index: int) -> bool:
    next = index + 1
    if next >= len(grouped_entries):
        return False

    next_entry = grouped_entries[next]

    return next_entry["role"] == "notice" and next_entry.get("transient", False)
