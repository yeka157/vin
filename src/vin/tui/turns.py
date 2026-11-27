"""Running one model turn on a worker thread and showing its events as they arrive."""

import asyncio
from datetime import datetime
from queue import Empty
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Literal

from rich.text import Text

from vin.agent import close_open_calls, is_summary
from vin.auth.chatgpt import TokenExchangeError
from vin.provider.chatgpt import stream_chatgpt_response
from vin.provider.events import ReplyEvents
from vin.provider.openrouter import stream_openrouter_response
from vin.tui.menus import task_is_cancelling
from vin.tui.subagents import apply_subagent_event, finish_subagent
from vin.tui.transcript import format_duration
from vin.types.auth import AccountRecord, OpenRouterAccount
from vin.types.settings import ReplyOptions
from vin.types.transcript import AssistantEntry, Subagent, ToolEntry

if TYPE_CHECKING:
    from vin.tui.app import VinApp


async def stream_reply(owner: "VinApp", text: str, *, compact_only: bool = False
) -> Literal["error", "rate_limit"] | None:
    events = ReplyEvents()
    cancel_event = Event()
    entry: AssistantEntry = {"role": "assistant", "content": "", "streaming": True}
    request = [message.copy() for message in owner.messages]
    if not compact_only:
        request.append({"role": "user", "content": text})
    owner.reply_entry = entry
    owner.request_started_at = monotonic()
    owner.last_request_duration = None
    completed = False
    text_dirty = False
    tool_entry: ToolEntry | None = None
    subagent: Subagent | None = None
    owner.tool_status = None

    if not compact_only:
        owner.panel_start = len(owner.subagents)
        owner.panel_index = None

    try:
        if compact_only and not request:
            owner.transcript.append(
                {"role": "notice", "content": "No conversation context to compact."}
            )
            completed = True
            return
        credentials: AccountRecord | OpenRouterAccount | None
        if owner.selected_provider == "openrouter":
            credentials = owner.openrouter
        else:
            try:
                credentials = await owner.prepare_request_credentials()
            except Exception as error:
                message = "Could not refresh ChatGPT sign-in. Please try again."
                if isinstance(error, TokenExchangeError) and error.requires_login:
                    message = (
                        "ChatGPT sign-in has expired. Sign in again with /login."
                    )
                owner.transcript.append(
                    {"role": "notice", "content": Text(message, style="red")}
                )
                return "error"

        compact_threshold: int | None = None
        for model in owner.models:
            if model["slug"] == owner.selected_model:
                window = model.get("context_window")
                if window is not None:
                    compact_threshold = window * 4 // 5
                break

        if (
            compact_threshold is None
            and owner.selected_model is not None
            and not compact_only
        ):
            notice = (
                "Automatic compaction is off because this model's "
                "context limit is unknown."
            )
            if not any(
                row["role"] == "notice" and row["content"] == notice
                for row in owner.transcript
            ):
                owner.transcript.append({"role": "notice", "content": notice})

        options: ReplyOptions = {}
        if compact_threshold is not None:
            options["compact_threshold"] = compact_threshold
        if compact_only:
            options["compact_only"] = True

        if not compact_only:
            options["subagents"] = owner.settings["subagents"]
            options["subagent_budget"] = owner.settings["subagent_budget"]
            options["search"] = owner.search_on()
            options["fetch"] = owner.settings["fetch"]

        Thread(
            target=(
                stream_openrouter_response
                if owner.selected_provider == "openrouter"
                else stream_chatgpt_response
            ),
            args=(
                (credentials or {}).copy(),
                owner.selected_model,
                request,
                events,
                cancel_event,
            ),
            kwargs=options,
            daemon=True,
        ).start()

        while True:
            try:
                event = events.get_nowait()
                kind = event[0]
            except Empty:
                if text_dirty and owner.checkpoint_session():
                    text_dirty = False
                await asyncio.sleep(0.03)
                continue

            if event[0] == "text":
                if not entry["content"]:
                    owner.transcript.append(entry)
                entry["content"] += event[1]
                text_dirty = True
            elif event[0] == "usage":
                owner.last_usage = event[1].copy()
            elif event[0] == "tool_diff":
                if tool_entry is not None and tool_entry["name"] == "edit":
                    tool_entry["diff"] = event[1]
            elif (
                event[0] == "tool_started"
                or event[0] == "tool_finished"
                or event[0] == "tool_failed"
            ):
                value = event[1]
                name = value["name"] if isinstance(value, dict) else value
                detail = value.get("detail", "") if isinstance(value, dict) else ""
                state: Literal["running", "finished", "failed"] = (
                    "running"
                    if kind == "tool_started"
                    else "finished" if kind == "tool_finished" else "failed"
                )
                owner.tool_status = (name, state)
                if kind == "tool_started":
                    entry["intermediate"] = True
                    entry["streaming"] = False
                    entry = {"role": "assistant", "content": "", "streaming": True}
                    owner.reply_entry = entry
                    label = f"{name} {detail}" if detail else name
                    tool_entry = {
                        "role": "tool",
                        "name": name,
                        "detail": detail,
                        "label": label,
                        "status": "running",
                        "content": Text(f"● {label} · running", style="dim"),
                    }
                    owner.transcript.append(tool_entry)
                    if isinstance(value, dict) and "task_id" in value:
                        subagent = {
                            "id": value["task_id"],
                            "description": detail,
                            "status": "running",
                            "started_at": monotonic(),
                            "duration": None,
                            "tokens": 0,
                            "transcript": [
                                {"role": "user", "content": value.get("prompt", "")}
                            ],
                        }
                        owner.subagents.append(subagent)
                elif tool_entry is not None:
                    assert "label" in tool_entry
                    tool_entry["status"] = state
                    tool_entry["content"] = Text(
                        f"● {tool_entry['label']} · {state}",
                        style="red" if kind == "tool_failed" else "dim",
                    )
                    tool_entry = None
                    if subagent is not None:
                        finish_subagent(
                            subagent,
                            "failed" if kind == "tool_failed" else "finished",
                        )
                        subagent = None
            elif event[0] == "subagent":
                target = next(
                    (
                        item
                        for item in owner.subagents
                        if item["id"] == event[1]["task_id"]
                    ),
                    None,
                )
                if target is not None:
                    apply_subagent_event(target, event[1]["event"])
            elif event[0] == "complete":
                value = event[1]
                entry["streaming"] = False
                if value is None and compact_only:
                    owner.transcript.append(
                        {"role": "notice", "content": "Nothing to compact yet."}
                    )
                elif value is None:
                    owner.messages.extend(
                        [
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": entry["content"]},
                        ]
                    )
                else:
                    newly_compacted = (
                        bool(value)
                        and is_summary(value[0])
                        and value[0] not in owner.messages
                    )
                    owner.messages[:] = value
                    if newly_compacted:
                        owner.transcript.append(
                            {
                                "role": "notice",
                                "content": "Conversation compacted.",
                            }
                        )
                completed = True
                return
            elif event[0] == "error" or event[0] == "rate_limit":
                owner.transcript.append(
                    {"role": "notice", "content": Text(event[1], style="red")}
                )
                return event[0]

            if kind in (
                "tool_started",
                "tool_finished",
                "tool_failed",
                "tool_diff",
            ):
                if owner.active_turn is not None:
                    owner.active_turn["tools_used"] = True
                if owner.checkpoint_session(force=True):
                    text_dirty = False
            elif kind in ("text", "subagent"):
                if owner.checkpoint_session():
                    text_dirty = False

            owner.session.app.invalidate()
            await asyncio.sleep(0)
    finally:
        cancel_event.set()
        if not completed and tool_entry is not None:
            assert "label" in tool_entry
            interrupted = task_is_cancelling()
            interrupted_state: Literal["interrupted", "failed"] = (
                "interrupted" if interrupted else "failed"
            )
            tool_entry["status"] = interrupted_state
            tool_entry["content"] = Text(
                f"● {tool_entry['label']} · {interrupted_state}",
                style="yellow" if interrupted else "red",
            )
            if subagent is not None:
                finish_subagent(subagent, interrupted_state)
        if not completed and events.tools_used.is_set():
            owner.messages[:] = close_open_calls(list(request)) + [
                {
                    "role": "assistant",
                    "content": "Turn interrupted after tool use. Changes may already exist; read relevant files before retrying.",
                }
            ]
        if owner.pending_reply is asyncio.current_task():
            if (
                not completed
                and owner.tool_status is not None
                and owner.tool_status[1] == "running"
            ):
                owner.tool_status = (owner.tool_status[0], "failed")
            assert owner.request_started_at is not None
            owner.last_request_duration = monotonic() - owner.request_started_at
            if completed:
                if not compact_only:
                    finished_at = datetime.now().astimezone().strftime("%I:%M %p")
                    owner.transcript.append(
                        {
                            "role": "notice",
                            "content": Text(
                                f"* Worked for {format_duration(owner.last_request_duration)}"
                                f" · done {finished_at}",
                                style="dim",
                            ),
                        }
                    )
            else:
                interrupted = task_is_cancelling()
                label = "Interrupted" if interrupted else "Failed"
                owner.transcript.append(
                    {
                        "role": "notice",
                        "content": Text(
                            f"{label} after {format_duration(owner.last_request_duration)}",
                            style="yellow" if interrupted else "red",
                        ),
                    }
                )
            owner.active_turn = None
            owner.checkpoint_session(force=True)
            owner.request_started_at = None
            owner.pending_reply = None
            owner.reply_entry = None
        owner.session.app.invalidate()
