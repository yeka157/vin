from typing import cast
from vin.types import ActiveTurn, History, Pathish, SessionRecord, TranscriptEntry, is_dictionary, is_list

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import UUID, uuid4

from vin.sessions.transcript import decode_transcript, encode_transcript


def new_session(project_dir: Pathish, first_message: object) -> SessionRecord:
    if not isinstance(first_message, str):
        raise ValueError("")
    elif not first_message.strip():
        raise ValueError("")

    now = datetime.now(timezone.utc).isoformat()

    return {
        "version": 1,
        "id": str(uuid4()),
        "project_dir": str(Path(project_dir).resolve()),
        "name": " ".join(first_message.split())[:80],
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "transcript": [],
        "active_turn": None,
    }


def validate_session(record: object, project_dir: Pathish) -> None:
    if (
        not is_dictionary(record)
        or type(record.get("version")) is not int
        or record["version"] != 1
    ):
        raise ValueError("Unsupported session record.")

    session_id = record.get("id")

    if not isinstance(session_id, str):
        raise ValueError("Invalid session ID.")

    parsed_id = UUID(session_id)
    if parsed_id.version != 4 or str(parsed_id) != session_id:
        raise ValueError("Invalid session ID.")

    if record.get("project_dir") != str(Path(project_dir).resolve()):
        raise ValueError("Session belongs to a different project.")

    name = record.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError("Invalid session name.")

    timestamps: list[datetime] = []

    for field in ("created_at", "updated_at"):
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError("Invalid session timestamp.")

        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("Invalid session timestamp.") from None

        if parsed.tzinfo is None:
            raise ValueError("Session timestamps must include a timezone.")

        timestamps.append(parsed)

    if timestamps[1] < timestamps[0]:
        raise ValueError("Session update time precedes creation time.")

    messages = record.get("messages")
    if not is_list(messages) or any(
        not is_dictionary(message) for message in messages
    ):
        raise ValueError("Invalid session context.")

    entries = decode_transcript(record.get("transcript"))

    for entry in entries:
        if entry["role"] in ("user", "assistant") and not isinstance(
            entry["content"], str
        ):
            raise ValueError("Invalid saved mesasge content.")

        for field in ("streaming", "intermediate", "expanded", "persist", "transient"):
            if field in entry and type(cast(dict[str, object], entry)[field]) is not bool:
                raise ValueError("Invalid saved entry flag.")

        if entry.get("persist") is False or entry.get("transient") is True:
            raise ValueError("Display-only entries cannot be saved.")

        if entry["role"] != "tool":
            continue

        tool_name = entry.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Invalid saved tool name.")

        if entry.get("status") not in ("running", "finished", "failed", "interrupted"):
            raise ValueError("Invalid saved tool status.")

        for field in ("detail", "label", "diff"):
            if field in entry and not isinstance(cast(dict[str, object], entry)[field], str):
                raise ValueError("Invalid saved tool metadata")

    if "active_turn" not in record:
        raise ValueError("Missing active turn.")

    active_turn = record["active_turn"]

    if active_turn is not None:
        if not is_dictionary(active_turn):
            raise ValueError("Invalid active turn.")

        prompt = active_turn.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Invalid active-turn prompt")

        if type(active_turn.get("tools_used")) is not bool:
            raise ValueError("Invalid active-turn tool flag.")

    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Session data cannot be saved as JSON.") from None


def snapshot_session(record: SessionRecord, messages: History, transcript: list[TranscriptEntry], *, active_turn: ActiveTurn | None) -> SessionRecord:
    snapshot = deepcopy(record)
    snapshot["messages"] = deepcopy(messages)
    snapshot["transcript"] = deepcopy(encode_transcript(transcript))
    snapshot["active_turn"] = deepcopy(active_turn)
    snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()

    validate_session(snapshot, snapshot["project_dir"])
    return snapshot
