import json
from pathlib import Path
from queue import SimpleQueue
from threading import Event

from urllib.error import HTTPError, URLError
from vin.agent import CONTEXT_FULL, AgentError, context_too_long, run_reply
from vin.auth.credentials import save_private_json
from vin.provider.events import EventSender, ReplyEvent, make_senders
from vin.provider.model_cache import parse_models
from vin.types import History, ModelMetadata, is_string_list
from vin.provider.prompts import instructions
from vin.types import is_list, is_record
from typing import TYPE_CHECKING, Literal, cast
from http.client import HTTPResponse
from urllib.request import urlopen

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if is_list(content):
        return "".join(
            text
            for part in content
            if is_record(part) and isinstance(text := part.get("text"), str)
        )
    return ""


def to_chat_messages(
    history: History, prompt: str
) -> list["ChatCompletionMessageParam"]:
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": prompt}]
    assistant: ChatCompletionAssistantMessageParam | None = None
    for item in history:
        kind = item.get("type")
        if kind == "function_call":
            if assistant is None:
                assistant = {"role": "assistant", "content": None}
                messages.append(assistant)
            assistant["tool_calls"] = [
                *assistant.get("tool_calls", []),
                {
                    "id": str(item.get("call_id")),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name")),
                        "arguments": str(item.get("arguments")),
                    },
                },
            ]
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id")),
                    "content": str(item.get("output")),
                }
            )
            assistant = None
        elif kind == "message" and item.get("role") == "assistant":
            assistant = {"role": "assistant", "content": _text(item.get("content"))}
            messages.append(assistant)
        elif item.get("role") == "user":
            messages.append({"role": "user", "content": _text(item.get("content"))})
            assistant = None
    return messages


def _request_completion(
    client: "OpenAI",
    model: str,
    history: History,
    send: EventSender,
    cancel_event: Event,
    *,
    compact_only: bool,
    subagent: bool,
    task: bool,
) -> tuple[History, bool] | None:
    from vin.tools import response_tools
    from openai import omit

    if cancel_event.is_set():
        return None
    tools: list[ChatCompletionToolParam] = (
        []
        if compact_only
        else [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {},
                },
            }
            for tool in response_tools(task=task and not subagent)
        ]
    )
    text: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finished = False
    with client.chat.completions.create(
        model=model,
        messages=to_chat_messages(
            history,
            instructions(compact_only=compact_only, subagent=subagent, task=task),
        ),
        tools=tools or omit,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"reasoning": {"effort": "medium"}},
    ) as stream:
        for chunk in stream:
            if cancel_event.is_set():
                return None
            if chunk.usage:
                send(
                    "usage",
                    {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    },
                )
            for choice in chunk.choices:
                if choice.delta.content:
                    text.append(choice.delta.content)
                    send("text", choice.delta.content)
                for call in choice.delta.tool_calls or []:
                    part = calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        part["id"] = call.id
                    if call.function:
                        if call.function.name:
                            part["name"] = call.function.name
                        if call.function.arguments:
                            part["arguments"] += call.function.arguments

                if choice.finish_reason == "length":
                    raise AgentError("OpenRouter returned an incomplete response.")
                if choice.finish_reason is not None:
                    finished = True
    if not finished:
        raise AgentError("OpenRouter stream ended before completion.")

    reply = "".join(text)
    items: History = []

    if reply:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply}],
            }
        )

    for index in sorted(calls):
        part = calls[index]
        if not part["id"] or not part["name"]:
            raise AgentError("OpenRouter returned an invalid tool call.")
        items.append(
            {
                "type": "function_call",
                "call_id": part["id"],
                "name": part["name"],
                "arguments": part["arguments"] or "{}",
                "status": "completed",
            }
        )

    return items, bool(reply.strip())


def _error_message(error: Exception) -> tuple[Literal["error", "rate_limit"], str]:
    import openai

    if isinstance(error, openai.APITimeoutError):
        return ("error", "OpenRouter request timed out. Please try again.")
    if isinstance(error, openai.APIStatusError):
        if error.status_code in (400, 413) and context_too_long(error):
            return ("error", CONTEXT_FULL)
        if error.status_code == 401:
            return (
                "error",
                "OpenRouter key was rejected. Use /login to enter a new one.",
            )
        if error.status_code == 402:
            return (
                "error",
                "OpenRouter credits are used up. Add credits and try again.",
            )
        if error.status_code == 403:
            return (
                "error",
                "OpenRouter blocked this request (HTTP 403). Check your key's limits or try another model.",
            )
        if error.status_code == 429:
            return (
                "rate_limit",
                "OpenRouter rate limit reached (HTTP 429). Please wait and try again.",
            )
        return ("error", f"OpenRouter request failed (HTTP {error.status_code}).")
    if isinstance(error, openai.APIConnectionError):
        return ("error", "Could not connect to OpenRouter. Please try again.")
    if isinstance(error, openai.APIError):
        return ("error", "OpenRouter response did not complete. Please try again.")
    return ("error", "OpenRouter request failed. Please try again.")


def stream_openrouter_response(
    account: object,
    model: object,
    messages: History,
    events: SimpleQueue[ReplyEvent],
    cancel_event: Event,
    *,
    compact_threshold: int | None = None,
    compact_only: bool = False,
    subagents: bool = True,
    subagent_budget: int = 8,
) -> None:
    send, subagent_send = make_senders(events, cancel_event)
    if cancel_event.is_set():
        return
    key = account.get("key") if is_record(account) else None
    if not isinstance(key, str) or not key.strip():
        send("error", "Sign in with /login before sending a message.")
        return
    if not isinstance(model, str) or not model.strip():
        send("error", "Choose a model with /model before sending a message.")
        return

    import openai

    client = openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"X-OpenRouter-Title": "Vin"},
        timeout=60,
        max_retries=2,
    )

    def request(
        history: History, sender: EventSender, subagent: bool, compact: bool
    ) -> tuple[History, bool] | None:
        return _request_completion(
            client,
            model,
            history,
            sender,
            cancel_event,
            compact_only=compact,
            subagent=subagent,
            task=subagents,
        )

    run_reply(
        client,
        request,
        messages,
        send,
        subagent_send,
        cancel_event,
        error_message=_error_message,
        compact_only=compact_only,
        compact_threshold=compact_threshold,
        subagents=subagents,
        subagent_budget=subagent_budget,
    )


CATALOG_URL = "https://openrouter.ai/api/v1/models"
WHITELIST_PATH = Path.home() / ".config" / "vin" / "openrouter_models.json"


def _per_million(value: object) -> float | None:
    if type(value) is str or type(value) is int or type(value) is float:
        try:
            price = float(value)
            if price < 0:
                return None
            return round(price * 1_000_000, 4)
        except ValueError:
            return None

    return None


def list_openrouter_models() -> list[ModelMetadata]:
    try:
        with cast(HTTPResponse, urlopen(CATALOG_URL, timeout=15)) as response:
            body = cast(object, json.load(response))
    except HTTPError as error:
        raise RuntimeError(
            f"Could not load OpenRouter models (HTTP {error.code})."
        ) from None
    except URLError, TimeoutError, OSError:
        raise RuntimeError("Could not connect to OpenRouter.") from None
    except ValueError:
        raise RuntimeError("OpenRouter returned an invalid model list.") from None

    data = body.get("data") if is_record(body) else None
    if not is_list(data):
        raise RuntimeError("OpenRouter returned an invalid model list.")

    rows: list[tuple[int, ModelMetadata]] = []
    for entry in data:
        if (
            is_record(entry)
            and isinstance(slug := entry.get("id"), str)
            and slug.strip()
            and is_list(parameters := entry.get("supported_parameters"))
            and "tools" in parameters
        ):
            name = entry.get("name")
            model: ModelMetadata = {
                "slug": slug,
                "display_name": (
                    name if isinstance(name, str) and name.strip() else slug
                ),
                "provider": "openrouter",
            }
            window = entry.get("context_length")
            if type(window) is int and window > 0:
                model["context_window"] = window
            pricing = entry.get("pricing")
            if is_record(pricing):
                prompt = _per_million(pricing.get("prompt"))
                completion = _per_million(pricing.get("completion"))
                if prompt is not None and completion is not None:
                    model["input_price"] = prompt
                    model["output_price"] = completion
            architecture = entry.get("architecture")
            modalities = (
                architecture.get("input_modalities")
                if is_record(architecture)
                else None
            )
            features = (
                [m for m in modalities if m != "text"]
                if is_string_list(modalities)
                else []
            )
            if "reasoning" in parameters:
                features.append("reasoning")
            model["features"] = features

            created = entry.get("created")

            rows.append((created if type(created) is int else 0, model))
    return [model for _, model in sorted(rows, key=lambda row: row[0], reverse=True)]


def load_openrouter_whitelist() -> list[ModelMetadata]:
    try:
        payload = cast(object, json.loads(WHITELIST_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return []
    except OSError, ValueError:
        raise RuntimeError("Could not read saved OpenRouter models.") from None

    try:
        return parse_models(payload, "openrouter")
    except ValueError:
        raise RuntimeError("Saved OpenRouter models are invalid.") from None


def save_openrouter_whitelist(models: list[ModelMetadata]) -> None:
    try:
        save_private_json(WHITELIST_PATH, models)
    except OSError, RuntimeError:
        raise RuntimeError("Could not save OpenRouter models.") from None
