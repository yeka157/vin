from time import monotonic
from typing import Literal
from rich.text import Text
from vin.provider.events import ReplyEvent
from vin.types import Subagent


def apply_subagent_event(subagent: Subagent, event: ReplyEvent) -> None:
    transcript = subagent["transcript"]
    if event[0] == "text":
        last = transcript[-1] if transcript else None
        if last is not None and last["role"] == "assistant" and last.get("streaming"):
            last["content"] += event[1]
        else:
            transcript.append(
                {"role": "assistant", "content": event[1], "streaming": True}
            )
    elif event[0] == "usage":
        subagent["tokens"] += event[1]["total_tokens"]
    elif event[0] == "tool_started":
        last = transcript[-1] if transcript else None
        if last is not None and last["role"] == "assistant":
            last["streaming"] = False
        value = event[1]
        name = value["name"] if isinstance(value, dict) else value
        detail = value.get("detail", "") if isinstance(value, dict) else ""
        label = f"{name} {detail}" if detail else name
        transcript.append(
            {
                "role": "tool",
                "name": name,
                "detail": detail,
                "label": label,
                "status": "running",
                "content": Text(f"● {label} · running", style="dim"),
            }
        )
    elif event[0] == "tool_finished" or event[0] == "tool_failed":
        state = "finished" if event[0] == "tool_finished" else "failed"
        for entry in reversed(transcript):
            if entry["role"] == "tool" and entry["status"] == "running":
                label = entry.get("label", entry["name"])
                entry["status"] = state
                entry["content"] = Text(
                    f"● {label} · {state}", style="red" if state == "failed" else "dim"
                )
                break
    elif event[0] == "tool_diff":
        for entry in reversed(transcript):
            if entry["role"] == "tool":
                if entry["name"] == "edit":
                    entry["diff"] = event[1]
                break


def finish_subagent(
    subagent: Subagent, status: Literal["finished", "failed", "interrupted"]
) -> None:
    subagent["status"] = status
    subagent["duration"] = monotonic() - subagent["started_at"]
    last = subagent["transcript"][-1] if subagent["transcript"] else None
    if last is not None and last["role"] == "assistant":
        last["streaming"] = False

    if status == "interrupted":
        for entry in subagent["transcript"]:
            if entry["role"] == "tool" and entry["status"] == "running":
                label = entry.get("label", entry["name"])
                entry["status"] = "interrupted"
                entry["content"] = Text(f"● {label} · interrupted", style="yellow")
