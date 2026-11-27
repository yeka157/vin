"""Events a model run sends from the worker thread to the UI."""

from typing import Literal, NotRequired, Protocol, TypedDict, overload, runtime_checkable

from vin.types.history import History


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


class SubagentEvent(TypedDict):
    task_id: str
    event: ReplyEvent


class EventSender(Protocol):
    @overload
    def __call__(self, kind: TextEventKind, value: str, /) -> None: ...
    @overload
    def __call__(self, kind: ToolEventKind, value: ToolStarted | str, /) -> None: ...
    @overload
    def __call__(self, kind: Literal["usage"], value: TokenUsage, /) -> None: ...
    @overload
    def __call__(self, kind: Literal["complete"], value: History | None, /) -> None: ...


@runtime_checkable
class SetMarker(Protocol):
    def set(self) -> object: ...
