from collections.abc import Callable
from queue import SimpleQueue
from threading import Event
from typing import (
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    overload,
    runtime_checkable,
)

from vin.types import History


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ToolStarted(TypedDict):
    name: str
    detail: str
    task_id: NotRequired[str]
    prompt: NotRequired[str]


type TextEventKind = Literal["text", "tool_diff", "error", "rate_limit"]
type ToolEventKind = Literal["tool_started", "tool_finished", "tool_failed"]
type ReplyEvent = (
    tuple[TextEventKind, str]
    | tuple[ToolEventKind, ToolStarted | str]
    | tuple[Literal["usage"], TokenUsage]
    | tuple[Literal["complete"], History | None]
    | tuple[Literal["subagent"], SubagentEvent]
)


class EventSender(Protocol):
    @overload
    def __call__(self, kind: TextEventKind, value: str, /) -> None: ...
    @overload
    def __call__(self, kind: ToolEventKind, value: ToolStarted | str, /) -> None: ...
    @overload
    def __call__(self, kind: Literal["usage"], value: TokenUsage, /) -> None: ...
    @overload
    def __call__(self, kind: Literal["complete"], value: History | None, /) -> None: ...


class SubagentEvent(TypedDict):
    task_id: str
    event: ReplyEvent


class ReplyEvents(SimpleQueue[ReplyEvent]):
    def __init__(self) -> None:
        super().__init__()
        self.tools_used = Event()


@runtime_checkable
class SetMarker(Protocol):
    def set(self) -> object: ...


def make_senders(
    events: SimpleQueue[ReplyEvent], cancel_event: Event, *, compact_only: bool
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
            if compact_only and kind == "text":
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
