from collections.abc import Callable
from queue import SimpleQueue
from threading import Event
from typing import Literal, cast, overload

from vin.types.events import (
    EventSender,
    ReplyEvent,
    SetMarker,
    TextEventKind,
    TokenUsage,
    ToolEventKind,
    ToolStarted,
)
from vin.types.history import History


class ReplyEvents(SimpleQueue[ReplyEvent]):
    def __init__(self) -> None:
        super().__init__()
        self.tools_used = Event()


def make_senders(
    events: SimpleQueue[ReplyEvent], cancel_event: Event
) -> tuple[EventSender, Callable[[str], EventSender]]:
    def make_send(put: Callable[[ReplyEvent], None]) -> EventSender:
        @overload
        def send(kind: TextEventKind, value: str) -> None: ...
        @overload
        def send(kind: ToolEventKind, value: ToolStarted | str) -> None: ...
        @overload
        def send(kind: Literal["usage"], value: TokenUsage) -> None: ...
        @overload
        def send(kind: Literal["complete"], value: History | None) -> None: ...
        def send(
            kind: TextEventKind | ToolEventKind | Literal["usage", "complete"],
            value: str | ToolStarted | TokenUsage | History | None = None,
        ) -> None:
            if kind == "tool_started":
                marker = cast(object, getattr(events, "tools_used", None))
                if isinstance(marker, SetMarker):
                    marker.set()
            if cancel_event.is_set():
                return
            if kind in ("text", "tool_diff", "error", "rate_limit") and isinstance(
                value, str
            ):
                put((kind, value))
            elif kind in ("tool_started", "tool_finished", "tool_failed"):
                if isinstance(value, str):
                    put((kind, value))
                elif isinstance(value, dict) and "name" in value:
                    put((kind, value))
            elif (
                kind == "usage" and isinstance(value, dict) and "input_tokens" in value
            ):
                put((kind, value))
            elif kind == "complete" and (value is None or isinstance(value, list)):
                put((kind, value))

        return send

    send = make_send(events.put)

    def subagent_send(task_id: str) -> EventSender:
        return make_send(
            lambda event: events.put(("subagent", {"task_id": task_id, "event": event}))
        )

    return send, subagent_send
