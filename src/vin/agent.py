from contextlib import AbstractContextManager
from threading import Event
import json
import re
from typing import Callable, Literal, cast
from vin.types import History, ProviderItem, is_record, is_list
from vin.provider.events import EventSender

type RequestFn = Callable[
    [History, EventSender, bool, bool], tuple[History, bool] | None
]

type SubagentSender = Callable[[str], EventSender]

type ErrorMessage = Callable[[Exception], tuple[Literal["error", "rate_limit"], str]]


class AgentError(RuntimeError):
    pass


def tool_detail(name: str, arguments: str) -> str:
    try:
        payload = cast(object, json.loads(arguments))
    except json.JSONDecodeError:
        return ""

    if not is_record(payload):
        return ""

    keys = {
        "read": "path",
        "write": "path",
        "edit": "path",
        "bash": "command",
        "task": "description",
    }

    key = keys.get(name)
    if key is None:
        return ""

    value = payload.get(key)
    if not isinstance(value, str):
        return ""

    value = value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    value = "".join(character for character in value if character.isprintable())
    value = " ".join(value.split())

    if len(value) > 120:
        return value[:119] + "…"

    return value


def execute_tool(
    name: str,
    arguments: str,
    cancel_event: Event,
    send: EventSender | None = None,
) -> str:
    from vin.tools import (
        TOOL_REGISTRY,
        check_cancelled,
        tool_cancellation,
        truncate_output,
    )

    with tool_cancellation(cancel_event):
        check_cancelled()
        failed = False

        try:
            (result,) = TOOL_REGISTRY.execute(
                {
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }
            )
        except Exception as error:
            check_cancelled()
            failed = True
            result = f"Error ({type(error).__name__}): {error}"

        if name == "bash" and not failed:
            match = re.match(r"Exit code: (-?\d+)\n", str(result))
            if match is not None:
                failed = int(match.group(1)) != 0
        output = truncate_output(str(result))
        check_cancelled()

        if send is not None:
            if name == "edit" and not failed:
                send("tool_diff", output)
            send("tool_failed" if failed else "tool_finished", name)

        return output


SUMMARY_PREFIX = "Summary of the conversation so far:\n\n"
MAX_CALLS = 8
SKIPPED_CALL = (
    "Not run: at most 8 tool calls run per response. Call it again if still needed."
)
BUDGET_USED = "Not run: the tool budget for this turn is used up."
FINAL_REQUEST_NOTE = "The tool budget for this turn is used up. Do not call tools. Summarize what you did, what you verified, and what remains."
CONTEXT_FULL = "The conversation is too long for this model. Use /compact to summarize it, or /clear to start fresh."


def context_too_long(error: object) -> bool:
    code = cast(object, getattr(error, "code", None))
    message = cast(object, getattr(error, "message", ""))
    text = message.lower() if isinstance(message, str) else ""
    return (
        code == "context_length_exceeded"
        or "context length" in text
        or "context_length" in text
    )


def close_open_calls(history: History) -> History:
    answered = {
        call_id
        for item in history
        if item.get("type") == "function_call_output"
        and isinstance(call_id := item.get("call_id"), str)
    }
    closed: History = []
    for item in history:
        closed.append(item)
        if item.get("type") == "function_call" and item.get("call_id") not in answered:
            closed.append(
                {
                    "type": "function_call_output",
                    "call_id": item.get("call_id"),
                    "output": "Not run: the turn was interrupted.",
                }
            )
    return closed


def quiet(_kind: str, _value: object) -> None:
    return


def estimate_tokens(history: History) -> int:
    return len(json.dumps(history)) // 3


def is_summary(item: ProviderItem) -> bool:
    content = item.get("content")
    return (
        item.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(SUMMARY_PREFIX)
    )


def compact_history(
    request: RequestFn,
    history: History,
    cancel_event: Event,
    *,
    subagent: bool,
    keep_last_user: bool,
) -> History | None:
    if cancel_event.is_set():
        return None
    completed = request(history, quiet, subagent, True)
    if completed is None:
        return None
    output = completed[0]
    if any(item.get("type") == "function_call" for item in output):
        raise AgentError("Compaction returned unexpected tool calls.")
    summary = final_answer(output)
    if not summary.strip():
        raise AgentError("Compaction returned no summary. Please try again.")
    compacted: History = [{"role": "user", "content": SUMMARY_PREFIX + summary}]
    last_user = next(
        (item for item in reversed(history) if item.get("role") == "user"), None
    )
    if keep_last_user and last_user is not None:
        compacted.append(last_user)
    return compacted


def run_agent_turn(
    request: RequestFn,
    inputs: History,
    send: EventSender,
    cancel_event: Event,
    *,
    compact_only: bool = False,
    compact_threshold: int | None = None,
    subagent: bool = False,
    max_requests: int = 16,
    subagent_budget: int = 8,
    subagent_sender: SubagentSender | None = None,
    subagents: bool = True,
) -> History | None:
    history = inputs

    if compact_only:
        return compact_history(
            request, history, cancel_event, subagent=subagent, keep_last_user=False
        )

    for attempt in range(max_requests):
        final = attempt == max_requests - 1
        if cancel_event.is_set():
            return None
        if final:
            history.append({"role": "user", "content": FINAL_REQUEST_NOTE})
        if (
            compact_threshold is not None
            and len(history) > 2
            and estimate_tokens(history) >= compact_threshold
        ):
            compacted = compact_history(
                request, history, cancel_event, subagent=subagent, keep_last_user=True
            )
            if compacted is None:
                return None
            history[:] = compacted

        completed = request(history, send, subagent, False)

        if completed is None:
            return None

        output, has_text = completed

        calls = [item for item in output if item.get("type") == "function_call"]

        if any(call.get("status") != "completed" for call in calls):
            raise AgentError("Response contains an incomplete tool call.")

        history.extend(output)

        if not calls:
            if not has_text:
                types = sorted(
                    {
                        (
                            "message"
                            if item.get("type") == "message"
                            else (
                                "reasoning"
                                if item.get("type") == "reasoning"
                                else "other"
                            )
                        )
                        for item in output
                    }
                )
                raise AgentError(
                    "The model returned an empty response "
                    f"(status=completed; output={','.join(types) or 'none'}). "
                    "Please try again."
                )

            return history

        if final:
            history.extend(
                {
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": BUDGET_USED,
                }
                for call in calls
            )
            return history

        for index, call in enumerate(calls):
            if cancel_event.is_set():
                return None

            name, arguments, call_id = (
                call.get("name"),
                call.get("arguments"),
                call.get("call_id"),
            )

            if (
                not isinstance(name, str)
                or not isinstance(arguments, str)
                or not isinstance(call_id, str)
            ):
                raise AgentError("The model returned invalid response output.")

            if index >= MAX_CALLS:
                result = SKIPPED_CALL
            elif name == "task" and subagents and not subagent:
                result = run_task(
                    request,
                    call_id,
                    arguments,
                    send,
                    cancel_event,
                    subagent_budget,
                    subagent_sender,
                )
            else:
                send(
                    "tool_started",
                    {
                        "name": name,
                        "detail": tool_detail(name, arguments),
                    },
                )
                result = execute_tool(name, arguments, cancel_event, send)

            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                }
            )
            if cancel_event.is_set():
                return None
    raise AgentError(f"Model request limit exceeded: maximum {max_requests}.")


def run_reply(
    client: AbstractContextManager[object],
    request: RequestFn,
    messages: History,
    send: EventSender,
    subagent_send: SubagentSender,
    cancel_event: Event,
    *,
    error_message: ErrorMessage,
    compact_only: bool = False,
    compact_threshold: int | None = None,
    subagents: bool = True,
    subagent_budget: int = 8,
) -> None:
    try:
        with client:
            history = run_agent_turn(
                request,
                messages,
                send,
                cancel_event,
                compact_only=compact_only,
                compact_threshold=compact_threshold,
                subagent_sender=subagent_send,
                subagents=subagents,
                subagent_budget=subagent_budget,
            )
    except AgentError as error:
        send("error", str(error))
        return
    except Exception as error:
        kind, text = error_message(error)
        send(kind, text)
        return
    if history is None and not compact_only:
        return
    send("complete", history)


def final_answer(history: History) -> str:
    for item in reversed(history):
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content")
            if not is_list(content):
                continue
            return "".join(
                text
                for part in content
                if is_record(part)
                and part.get("type") == "output_text"
                and isinstance(text := part.get("text"), str)
            )
    return ""


def run_task(
    request: RequestFn,
    call_id: str,
    arguments: str,
    send: EventSender,
    cancel_event: Event,
    budget: int,
    subagent_sender: SubagentSender | None,
) -> str:
    from vin.tools import truncate_output

    try:
        payload = cast(object, json.loads(arguments))
    except json.JSONDecodeError:
        payload = None

    description = payload.get("description") if is_record(payload) else None
    prompt = payload.get("prompt") if is_record(payload) else None

    if (
        not isinstance(description, str)
        or not isinstance(prompt, str)
        or not prompt.strip()
    ):
        send("tool_started", {"name": "task", "detail": tool_detail("task", arguments)})
        send("tool_failed", "task")
        return "Error: task needs a description and a prompt."

    send(
        "tool_started",
        {
            "name": "task",
            "detail": tool_detail("task", arguments),
            "task_id": call_id,
            "prompt": prompt,
        },
    )

    child = subagent_sender(call_id) if subagent_sender is not None else quiet

    try:
        history = run_agent_turn(
            request,
            [{"role": "user", "content": prompt}],
            child,
            cancel_event,
            subagent=True,
            max_requests=budget,
        )
    except AgentError as error:
        send("tool_failed", "task")
        return f"Error: subagent failed: {error}"

    if history is None:
        return ""
    send("tool_finished", "task")
    return truncate_output(
        final_answer(history)
        or "Subagent stopped: its tool budget ran out before it wrote an answer."
    )
