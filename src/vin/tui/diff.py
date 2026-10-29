import re

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text


def parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    pattern = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?"
    match = re.fullmatch(pattern, line)

    if match is None:
        return None

    return (
        int(match.group(1)),
        int(match.group(2) or "1"),
        int(match.group(3)),
        int(match.group(4) or "1"),
    )


def parse_diff_rows(
    diff: str,
) -> list[tuple[int | None, str, str]] | None:
    lines = diff.splitlines()

    if (
        len(lines) < 3
        or not lines[0].startswith("--- ")
        or not lines[1].startswith("+++ ")
    ):
        return None

    rows: list[tuple[int | None, str, str]] = []
    old_line = new_line = 0
    old_remaining = new_remaining = 0
    in_hunk = False

    for line in lines[2:]:
        header = parse_hunk_header(line)

        if header is not None:
            if old_remaining or new_remaining:
                return None

            if in_hunk:
                rows.append((None, "…", ""))

            old_line, old_remaining, new_line, new_remaining = header
            in_hunk = True
            continue

        if not in_hunk:
            return None

        if line == "\\ No newline at end of file":
            if not rows or rows[-1][1] not in (" ", "+", "-"):
                return None

            rows.append((None, "\\", "No newline at end of file"))
            continue

        if not line or line[0] not in (" ", "+", "-"):
            return None

        prefix = line[0]
        content = line[1:]

        if prefix == "-":
            rows.append((old_line, prefix, content))
            old_line += 1
            old_remaining -= 1
        elif prefix == "+":
            rows.append((new_line, prefix, content))
            new_line += 1
            new_remaining -= 1
        else:
            rows.append((new_line, prefix, content))
            old_line += 1
            new_line += 1
            old_remaining -= 1
            new_remaining -= 1

        if old_remaining < 0 or new_remaining < 0:
            return None

    if not in_hunk or old_remaining or new_remaining:
        return None

    return rows


def print_diff(console: Console, diff: str) -> None:
    rows = parse_diff_rows(diff)

    if rows is None:
        console.print(Padding(Text(diff), (0, 0, 0, 2)))
        return

    number_width = max(
        (len(str(number)) for number, _, _ in rows if number is not None), default=1
    )

    if console.width <= number_width + 8:
        console.print(Padding(Text(diff), (0, 0, 0, 2)))
        return

    table = Table.grid(padding=0, expand=True)
    table.add_column(width=number_width + 3, no_wrap=True)
    table.add_column(width=2, no_wrap=True)
    table.add_column(ratio=1, overflow="fold")

    for number, prefix, content in rows:
        label = (
            str(number).rjust(number_width)
            if number is not None
            else " " * number_width
        )

        foreground = {"+": "green", "-": "red"}.get(
            prefix, "dim" if number is None else None
        )

        background = {
            "+": "#14261c",
            "-": "#2b191d",
        }.get(prefix)

        table.add_row(
            Text(f"{label} │ ", style="dim"),
            Text(prefix + " ", style=foreground or ""),
            Text(content, style=foreground or ""),
            style=f" on {background}" if background else None,
        )

    console.print(Padding(table, (0, 0, 0, 2)))


def format_edit_heading(path: str, diff: str) -> Text | None:
    rows = parse_diff_rows(diff)

    if rows is None:
        return None

    added = sum(1 for row in rows if row[1] == "+")
    removed = sum(1 for row in rows if row[1] == "-")

    heading = Text()
    heading.append("● ", style="green")
    heading.append("Edit ", style="dim")
    heading.append(path or "file", style="bold")
    heading.append("  ")
    heading.append(f"+{added}", style="green")
    heading.append(f" −{removed}", style="red")

    return heading
