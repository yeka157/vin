"""Transcript message rendering, viewport scrolling, and mouse handling."""

from io import StringIO

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

from vin.tui.diff import print_diff as print_tool_diff
from vin.tui.diff import format_edit_heading


def format_duration(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_user_message(console, text):
    console.print(
        Padding(
            Text("› " + text),
            (0, 0),
            style="#ffffff on #393939",
            expand=True,
        )
    )
    console.print()


def print_assistant_message(console, text, *, streaming=False, intermediate=False):
    content = Text(text) if streaming else Markdown(text, code_theme="ansi_dark")
    console.print(
        Padding(content, (0, 0, 0, 1)),
    )
    console.print()


def group_tool_entries(entries):
    result = []
    current_group = None
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


def print_activity_entry(console, entry):
    heading = None

    if (
        entry["role"] == "tool"
        and entry.get("name") == "edit"
        and entry.get("status") == "finished"
        and entry.get("diff")
    ):
        heading = format_edit_heading(entry.get("detail", ""), entry["diff"])

    console.print(heading if heading is not None else entry["content"], markup=False)
    if entry["role"] == "tool" and entry.get("diff"):
        print_tool_diff(console, entry["diff"])
    console.print()


def render_transcript(entries, width, height, *, group_headers=None):
    if group_headers is not None:
        group_headers.clear()
    output = StringIO()
    transcript_console = Console(
        file=output,
        width=width,
        height=height,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
    )

    grouped_entries = group_tool_entries(entries)
    for index, entry in enumerate(grouped_entries):
        if entry["role"] == "user":
            print_user_message(transcript_console, entry["content"])
        elif entry["role"] == "assistant":
            print_assistant_message(
                transcript_console,
                entry["content"],
                streaming=entry.get("streaming", False),
                intermediate=entry.get("intermediate", False),
            )
        elif entry["role"] == "tool_group":
            first_tool = entry["entries"][0]
            active = is_tool_group_active(grouped_entries, index)
            expanded = active or first_tool.get("expanded", False)

            if not active:
                summary = format_tool_group_summary(entry["entries"])
                arrow = "▾" if expanded else "▸"

                if group_headers is not None:
                    start_line = output.getvalue().count("\n")

                transcript_console.print(Text(f"{arrow} {summary}", style="dim"))

                if group_headers is not None:
                    end_line = output.getvalue().count("\n")
                    group_headers.append((start_line, end_line, first_tool))

            if expanded:
                for tool_entry in entry["entries"]:
                    print_activity_entry(transcript_console, tool_entry)
            else:
                transcript_console.print()
        else:
            print_activity_entry(transcript_console, entry)

    return ANSI(output.getvalue())


class TranscriptControl(UIControl):
    def __init__(self, get_entries) -> None:
        self.get_entries = get_entries
        self.top_line = None
        self.line_count = 1
        self.viewport_height = 1
        self.group_headers = []
        self.visible_start = 0

    def scroll(self, lines):
        last_top_line = max(0, self.line_count - self.viewport_height)
        current_top_line = last_top_line if self.top_line is None else self.top_line
        new_top_line = max(0, min(last_top_line, current_top_line + lines))
        self.top_line = None if new_top_line == last_top_line else new_top_line

    def scroll_to_bottom(self):
        self.top_line = None

    def mouse_handler(self, mouse_event):
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

    def create_content(self, width: int, height: int | None):
        text = render_transcript(
            self.get_entries(),
            max(1, width),
            max(1, height or 1),
            group_headers=self.group_headers,
        )

        control = FormattedTextControl(text, show_cursor=False)
        content = control.create_content(width, height)
        self.line_count = content.line_count
        self.viewport_height = max(1, height or 1)
        self.scroll(0)
        start = self.top_line
        if start is None:
            start = max(0, self.line_count - self.viewport_height)

        self.visible_start = start
        visible_count = min(self.viewport_height, self.line_count - start)

        return UIContent(
            get_line=lambda line_number: content.get_line(start + line_number),
            line_count=visible_count,
            cursor_position=Point(x=0, y=max(0, visible_count - 1)),
            show_cursor=False,
        )


def format_tool_group_summary(entries):
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


def is_tool_group_active(grouped_entries, index):
    next = index + 1
    if next >= len(grouped_entries):
        return False

    next_entry = grouped_entries[next]

    return next_entry["role"] == "notice" and next_entry.get("transient", False)
