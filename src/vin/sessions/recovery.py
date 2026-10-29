from typing import cast
from vin.types import History, Pathish, SessionRecord, TranscriptEntry

from copy import deepcopy

from rich.text import Text
from vin.sessions.records import validate_session
from vin.sessions.transcript import decode_transcript


def restore_session(record: SessionRecord, project_dir: Pathish) -> tuple[History, list[TranscriptEntry]]:
    validate_session(record, project_dir)

    messages = deepcopy(record["messages"])
    transcript = cast(list[TranscriptEntry], decode_transcript(deepcopy(record["transcript"])))

    if record["active_turn"] is not None:
        if record["active_turn"]["tools_used"]:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": record["active_turn"]["prompt"],
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Turn interrupted after tool use. "
                            "Changes may already exist; read relevant "
                            "files before retrying."
                        ),
                    },
                ]
            )
        for entry in reversed(transcript):
            if entry["role"] == "user":
                break

            if entry["role"] == "assistant" and entry.get("streaming", False):
                entry["streaming"] = False
            elif entry["role"] == "tool" and entry["status"] == "running":
                entry["status"] = "interrupted"
                label = entry.get("label")
                if not label:
                    label = f"{entry['name']} {entry.get('detail', '')}".strip()
                entry["content"] = Text(f"● {label} · interrupted", style="yellow")
        transcript.append(
            {
                "role": "notice",
                "content": Text("Previous turn was interrupted.", style="yellow"),
            }
        )

    return messages, transcript
