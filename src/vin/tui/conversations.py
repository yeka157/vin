"""Session checkpoints and resume/rename commands for the terminal application."""

import asyncio
from copy import deepcopy
from time import monotonic
from typing import TYPE_CHECKING

from rich.text import Text

from vin.sessions.records import snapshot_session
from vin.sessions.recovery import restore_session
from vin.sessions.storage import list_sessions, load_session, save_session

if TYPE_CHECKING:
    from vin.tui.app import VinApp


def checkpoint_session(owner: "VinApp", *, force: bool = False) -> bool:
    if owner.session_record is None:
        return False

    now = monotonic()

    if (
        not force
        and owner.last_session_checkpoint is not None
        and now - owner.last_session_checkpoint < 0.5
    ):
        return False

    owner.last_session_checkpoint = now

    try:
        snapshot = snapshot_session(
            owner.session_record,
            owner.messages,
            owner.transcript,
            active_turn=owner.active_turn,
        )

        save_session(
            owner.project_dir,
            snapshot,
            expected_record=owner.saved_session_record,
        )
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Conversation was not saved: {error}"
        if message != owner.session_save_error:
            owner.transcript.append(
                {"role": "notice", "content": Text(message, style="red")}
            )
        owner.session_save_error = message
        return False

    owner.session_record = snapshot
    owner.saved_session_record = deepcopy(snapshot)
    owner.session_save_error = None
    return True


async def resume_conversation(owner: "VinApp", query: str | None = None) -> str | None:
    records = list_sessions(owner.project_dir)
    query = (query or "").strip()

    if query:
        matches = [record for record in records if record["id"] == query] or [
            record for record in records if record["name"] == query
        ]
        if not matches:
            raise RuntimeError(f"No saved conversation matches: {query}")
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple conversations have this title. "
                "Use /resume to choose one."
            )
        selected_id = matches[0]["id"]
    else:
        if not records:
            return "No saved conversations for this project."

        options = [
            (
                record["id"],
                f"{record['name']} · {record['updated_at']} · {record['id'][:8]}",
            )
            for record in records
        ]
        try:
            selected_id = await owner.choose_inline(
                "Resume a conversation",
                options,
            )
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            return

    record = load_session(owner.project_dir, selected_id)
    if record is None:
        raise RuntimeError("Saved conversation is no longer available.")
    messages, transcript = restore_session(record, owner.project_dir)

    owner.messages = messages
    owner.transcript = transcript
    owner.session_record = deepcopy(record)
    owner.saved_session_record = deepcopy(record)
    owner.active_turn = None
    owner.last_session_checkpoint = None
    owner.session_save_error = None
    owner.transcript_control.scroll_to_bottom()
    return f"Resumed: {record['name']}"


def rename_conversation(owner: "VinApp", name: object) -> str | None:
    if owner.session_record is None:
        raise RuntimeError("No conversation to rename. Send a message first.")
    if not isinstance(name, str):
        raise ValueError("Use a session name between 1 and 80 characters.")

    name = " ".join(name.split())
    if not name or len(name) > 80:
        raise ValueError("Use a session name between 1 and 80 characters.")
    previous = owner.session_record
    owner.session_record = deepcopy(previous)
    owner.session_record["name"] = name
    if not owner.checkpoint_session(force=True):
        owner.session_record = previous
        return

    return f"Renamed: {name}"
