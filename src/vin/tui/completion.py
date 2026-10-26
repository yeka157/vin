from typing import Iterable
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from vin.sessions.storage import list_sessions


class SessionCompleter(Completer):
    def __init__(self, project_dir, commands) -> None:
        self.project_dir = project_dir
        self.commands = commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        parts = text.split(maxsplit=1)

        if (
            not parts
            or parts[0] != "/resume"
            or (len(parts) == 1 and not text[-1:].isspace())
        ):
            yield from self.commands.get_completions(document, complete_event)
            return

        query = parts[1] if len(parts) > 1 else ""
        search = query.strip().casefold()

        try:
            records = list_sessions(self.project_dir)
        except (OSError, RuntimeError, ValueError):
            return

        for record in records:
            if search in record["name"].casefold() or search in record["id"].casefold():
                yield Completion(
                    record["id"],
                    start_position=-len(query),
                    display=record["name"],
                    display_meta=f"{record['updated_at']} · {record['id'][:8]}",
                )
