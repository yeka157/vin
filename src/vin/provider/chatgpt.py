import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from threading import Event
from queue import SimpleQueue
from typing import TYPE_CHECKING, cast

from vin.agent import AgentError
from vin.provider.shared import openai_client, stream_reply
from vin.provider.prompts import instructions
from vin.types.chatgpt import WireClient
from vin.types.data import is_record, is_list, json_value
from vin.types.history import History
from vin.types.models import ModelMetadata
from vin.types.events import EventSender, ReplyEvent

if TYPE_CHECKING:
    from http.client import HTTPResponse
    from openai import OpenAI
    from openai.types.responses import Response, ResponseOutputItem


def _wire_client(client: object) -> WireClient:
    # The SDK serializes wire JSON, including opaque saved/future provider items.
    # Its frozen input union cannot describe that forward-compatible history.
    if not isinstance(client, WireClient):
        raise TypeError("OpenAI client does not expose Responses.")
    return client


def _has_output_item(item: "ResponseOutputItem | None") -> bool:
    return item is not None


def list_chatgpt_models(credentials: object) -> list[ModelMetadata]:
    if not is_record(credentials) or not credentials:
        raise RuntimeError("Sign in with ChatGPT account first.")

    access_token = credentials.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("Sign in with ChatGPT account first.")

    scopes = credentials.get("scopes", [])
    if not is_list(scopes) or "chatgpt.tokens.use.direct" not in scopes:
        raise RuntimeError(
            "ChatGPT plan usage is not enabled. "
            "Sign in again and allow ChatGPT plan usage."
        )

    # Model request
    request = Request(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
    )

    try:
        with cast("HTTPResponse", urlopen(request, timeout=60)) as response:
            payload = cast(object, json.load(response))
    except HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"Could not load ChatGPT models (HTTP {status})") from None
    except OSError:
        raise RuntimeError("Could not connect to load ChatGPT models.") from None
    except json.JSONDecodeError, UnicodeDecodeError:
        raise RuntimeError("Model endpoint returned invalid JSON.") from None

    if not is_record(payload) or not is_list(payload.get("models")):
        raise RuntimeError("Model endpoint returned an invalid model list.")

    raw_models = payload["models"]
    assert is_list(raw_models)
    models: list[ModelMetadata] = []
    for model in raw_models:
        if not is_record(model):
            raise RuntimeError("Model endpoint returned an invalid model entry.")

        if model.get("visibility") != "list":
            continue

        for field in ("slug", "display_name"):
            value = model.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"Model endpoint returned an invalid {field}.")

        window = model.get("context_window")
        if type(window) is not int or window <= 0:
            model.pop("context_window", None)

        model["provider"] = "chatgpt"
        models.append(cast(ModelMetadata, model))

    return models


def _request_response(
    client: WireClient,
    model: str,
    inputs: History,
    send: EventSender,
    cancel_event: Event,
    *,
    compact_only: bool = False,
    subagent: bool = False,
    task: bool = True,
    search: bool = False,
    fetch: bool = False,
) -> tuple["Response", bool] | None:
    from vin.tools import response_tools

    if cancel_event.is_set():
        return None

    streamed_text: list[str] = []
    completed_output: dict[int, "ResponseOutputItem"] = {}
    added_output: set[int] = set()

    with client.responses.create(
        model=model,
        input=inputs,
        instructions=instructions(
            compact_only=compact_only, subagent=subagent, task=task
        ),
        tools=(
            []
            if compact_only
            else response_tools(
                task=task and not subagent,
                search=search,
                fetch=fetch,
            )
        ),
        reasoning={"effort": "medium"},
        include=["reasoning.encrypted_content"],
        store=False,
        stream=True,
    ) as stream:
        if cancel_event.is_set():
            return None

        for event in stream:
            if cancel_event.is_set():
                return None

            if (
                event.type == "response.output_text.delta"
                or event.type == "response.refusal.delta"
            ):
                if event.delta:
                    streamed_text.append(event.delta)
                    send("text", event.delta)

            elif event.type == "response.output_item.added":
                if _has_output_item(event.item):
                    added_output.add(event.output_index)

            elif event.type == "response.output_item.done":
                completed_output[event.output_index] = event.item

            elif event.type == "response.completed":
                if event.response.status != "completed":
                    raise AgentError("ChatGPT returned an incomplete response.")

                output = list(event.response.output or [])
                if output:
                    for index, item in completed_output.items():
                        if (
                            index < 0
                            or index >= len(output)
                            or output[index].id != item.id
                            or output[index].type != item.type
                        ):
                            raise AgentError(
                                "ChatGPT returned inconsistent response output."
                            )
                        # Done events contain finalized arguments and encrypted reasoning.
                        output[index] = item
                else:
                    if added_output - completed_output.keys():
                        raise AgentError("ChatGPT returned incomplete response output.")
                    output = [
                        completed_output[index] for index in sorted(completed_output)
                    ]

                response = event.response.model_copy(update={"output": output})
                text = "".join(
                    part.text if part.type == "output_text" else part.refusal
                    for item in output
                    if item.type == "message"
                    for part in item.content
                    if part.type in ("output_text", "refusal")
                )
                preview = "".join(streamed_text)
                if not text.startswith(preview):
                    raise AgentError(
                        "ChatGPT completed text did not match its streaming preview. "
                        "Please try again."
                    )
                if len(text) > len(preview):
                    send("text", text[len(preview) :])
                if response.usage:
                    send(
                        "usage",
                        {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                    )
                return response, bool(text.strip())

            elif event.type in ("response.failed", "response.incomplete", "error"):
                raise AgentError("ChatGPT response did not complete. Please try again.")

    raise AgentError("ChatGPT stream ended before completion.")


def _complete(
    client: "OpenAI",
    model: str,
    history: History,
    send: EventSender,
    cancel_event: Event,
    *,
    compact_only: bool,
    subagent: bool,
    task: bool,
    search: bool,
    fetch: bool = False,
) -> tuple[History, bool] | None:
    completed = _request_response(
        _wire_client(client),
        model,
        history,
        send,
        cancel_event,
        compact_only=compact_only,
        subagent=subagent,
        task=task,
        search=search,
        fetch=fetch,
    )
    if completed is None:
        return None

    response, has_text = completed
    items: History = []
    for item in response.output:
        dumped = json_value(cast(object, item.model_dump(exclude_none=True)))
        if not isinstance(dumped, dict):
            raise AgentError("ChatGPT returned invalid response output.")
        items.append(dumped)
    return items, has_text


def stream_chatgpt_response(
    credentials: object,
    model: object,
    messages: History,
    events: SimpleQueue[ReplyEvent],
    cancel_event: Event,
    *,
    compact_threshold: int | None = None,
    compact_only: bool = False,
    subagents: bool = True,
    subagent_budget: int = 8,
    search: bool = False,
    fetch: bool = False,
) -> None:
    """Run on a worker thread; publish text, complete, or error queue events."""
    fields = credentials if is_record(credentials) else {}
    token = fields.get("access_token")
    scopes: object = fields.get("scopes") or list[object]()
    stream_reply(
        "ChatGPT",
        token if isinstance(token, str) else None,
        model,
        messages,
        events,
        cancel_event,
        connect=openai_client("https://api.openai.com/v1"),
        complete=_complete,
        statuses={
            401: ("error", "ChatGPT request failed (HTTP 401). Sign in again with /login."),
            403: ("error", "ChatGPT request failed (HTTP 403). Sign in again with /login."),
        },
        problem=(
            None
            if is_list(scopes) and "chatgpt.tokens.use.direct" in scopes
            else "ChatGPT plan usage is not enabled. Sign in again with /login."
        ),
        compact_threshold=compact_threshold,
        compact_only=compact_only,
        subagents=subagents,
        subagent_budget=subagent_budget,
        search=search,
        fetch=fetch,
    )
