from threading import Event
import json
import re
from typing import Callable, cast
from vin.types import History, is_record
from vin.provider.events import EventSender

type RequestFn = Callable[[History], tuple[History, bool] | None]


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


def context_after_compaction(history: History) -> History:
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        if item.get("type") == "compaction":
            return history[index:]
    return history.copy()


def run_agent_turn(
    request: RequestFn,
    inputs: History,
    send: EventSender,
    cancel_event: Event,
    *,
    compact_only: bool = False,
) -> History | None:
    history = list(inputs)

    for _ in range(16):
        if cancel_event.is_set():
            return None

        completed = request(history)

        if completed is None:
            return None

        output, has_text = completed

        calls = [item for item in output if item.get("type") == "function_call"]

        if compact_only and calls:
            raise AgentError("Compaction returned unexpected tool calls.")

        if len(calls) > 8:
            raise AgentError("Tool call limit exceeded: maximum 8.")

        if any(call.get("status") != "completed" for call in calls):
            raise AgentError("Response contains an incomplete tool call.")

        history.extend(output)

        has_compaction = any(item.get("type") == "compaction" for item in output)

        if has_compaction:
            history = context_after_compaction(history)

        if compact_only:
            return history if has_compaction else None

        if not calls:
            if not has_text:
                types = sorted(
                    {
                        "message" if item.get("type") == "message" else
                        "reasoning" if item.get("type") == "reasoning" else
                        "other"
                        for item in output
                    }
                )
                raise AgentError(
                    "ChatGPT returned an empty response "
                    f"(status=completed; output={','.join(types) or 'none'}). "
                    "Please try again."
                )

            return history

        for call in calls:
            if cancel_event.is_set():
                return None

            name, arguments, call_id = call.get("name"), call.get("arguments"), call.get("call_id")

            if not isinstance(name, str) or not isinstance(arguments, str) or not isinstance(call_id, str):
                raise AgentError("ChatGPT returned invalid response output.")

            send(
                "tool_started",
                {
                    "name": name,
                    "detail": tool_detail(name, arguments),
                },
            )
            result = execute_tool(name, arguments, cancel_event, send)

            if cancel_event.is_set():
                return None

            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                }
            )
    raise AgentError("Model request limit exceeded: maximum 16.")
