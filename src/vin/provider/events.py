from queue import SimpleQueue
from threading import Event
from typing import Literal, Protocol, TypedDict, overload

from vin.types import History


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ToolStarted(TypedDict):
    name: str
    detail: str


type TextEventKind = Literal["text", "tool_diff", "error", "rate_limit"]
type ToolEventKind = Literal["tool_started", "tool_finished", "tool_failed"]
type ReplyEvent = (
    tuple[TextEventKind, str]
    | tuple[ToolEventKind, ToolStarted | str]
    | tuple[Literal["usage"], TokenUsage]
    | tuple[Literal["complete"], History | None]
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


class ReplyEvents(SimpleQueue[ReplyEvent]):
    def __init__(self) -> None:
        super().__init__()
        self.tools_used = Event()
