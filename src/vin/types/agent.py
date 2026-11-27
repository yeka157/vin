"""The contract between the agent loop and a provider."""

from collections.abc import Callable
from threading import Event
from typing import Literal, Protocol

from vin.types.events import EventSender
from vin.types.history import History

type Outcome = tuple[Literal["error", "rate_limit"], str]
type RequestFn = Callable[
    [History, EventSender, bool, bool], tuple[History, bool] | None
]
type SubagentSender = Callable[[str], EventSender]
type ErrorMessage = Callable[[Exception], Outcome]


class Complete[C](Protocol):
    """Send one model request in the provider's wire format and return its output items."""

    def __call__(
        self,
        client: C,
        model: str,
        history: History,
        send: EventSender,
        cancel_event: Event,
        *,
        compact_only: bool,
        subagent: bool,
        task: bool,
        search: bool,
        fetch: bool,
    ) -> tuple[History, bool] | None: ...
