"""What every provider shares: starting a reply on the worker thread and turning API errors into messages.

A provider supplies only what differs: how to connect, how to send one request in its own
wire format, and any status codes that need their own wording.
"""

import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from queue import SimpleQueue
from threading import Event
from typing import TYPE_CHECKING, Literal, Protocol

from vin.agent import CONTEXT_FULL, context_too_long, run_reply
from vin.provider.events import EventSender, ReplyEvent, make_senders
from vin.types import History

if TYPE_CHECKING:
    from openai import OpenAI

type Outcome = tuple[Literal["error", "rate_limit"], str]


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
    ) -> tuple[History, bool] | None: ...


def openai_client(
    base_url: str, headers: Mapping[str, str] | None = None
) -> Callable[[str], "OpenAI"]:
    """Connect with the OpenAI SDK, which speaks both the Responses and Chat Completions formats."""

    def connect(key: str) -> "OpenAI":
        import openai

        if headers is None:
            return openai.OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=2)
        return openai.OpenAI(
            api_key=key,
            base_url=base_url,
            default_headers=dict(headers),
            timeout=60,
            max_retries=2,
        )

    return connect


def error_message(
    error: Exception, name: str, secret: str, statuses: Mapping[int, Outcome]
) -> Outcome:
    import openai

    if isinstance(error, openai.APITimeoutError):
        return ("error", f"{name} request timed out. Please try again.")
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        if status in (400, 413) and context_too_long(error):
            return ("error", CONTEXT_FULL)
        if status in statuses:
            return statuses[status]
        message = f"{name} request failed (HTTP {status})."
        for label, value in (("code", error.code), ("param", error.param)):
            if (
                isinstance(value, str)
                and value != str(status)
                and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", value)
                and secret.strip() not in value
            ):
                message += f" {label}={value}."
        return ("rate_limit" if status == 429 else "error", message)
    if isinstance(error, openai.APIConnectionError):
        return ("error", f"Could not connect to {name}. Please try again.")
    if isinstance(error, openai.APIError):
        return ("error", f"{name} response did not complete. Please try again.")
    return ("error", f"{name} request failed. Please try again.")


def stream_reply[C: AbstractContextManager[object]](
    name: str,
    key: str | None,
    model: object,
    messages: History,
    events: SimpleQueue[ReplyEvent],
    cancel_event: Event,
    *,
    connect: Callable[[str], C],
    complete: Complete[C],
    statuses: Mapping[int, Outcome] | None = None,
    problem: str | None = None,
    compact_threshold: int | None = None,
    compact_only: bool = False,
    subagents: bool = True,
    subagent_budget: int = 8,
    search: bool = False,
) -> None:
    """Run on a worker thread; publish text, complete, or error queue events.

    problem is a provider-specific reason not to send (for example a missing plan scope).
    """
    send, subagent_send = make_senders(events, cancel_event)
    if cancel_event.is_set():
        return
    if not key or not key.strip():
        send("error", "Sign in with /login before sending a message.")
        return
    if problem is not None:
        send("error", problem)
        return
    if not isinstance(model, str) or not model.strip():
        send("error", "Choose a model with /model before sending a message.")
        return

    client = connect(key)
    selected = model
    secret = key
    known = statuses or {}

    def request(
        history: History, sender: EventSender, subagent: bool, compact: bool
    ) -> tuple[History, bool] | None:
        return complete(
            client,
            selected,
            history,
            sender,
            cancel_event,
            compact_only=compact,
            subagent=subagent,
            task=subagents,
            search=search,
        )

    run_reply(
        client,
        request,
        messages,
        send,
        subagent_send,
        cancel_event,
        error_message=lambda error: error_message(error, name, secret, known),
        compact_only=compact_only,
        compact_threshold=compact_threshold,
        subagents=subagents,
        subagent_budget=subagent_budget,
    )
