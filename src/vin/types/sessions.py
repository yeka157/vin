"""Saved sessions and the transcript entries as they are written to disk."""

from typing import Literal, NotRequired, Required, TypedDict

from rich.text import Text

from vin.types.history import History


class SavedSpan(TypedDict):
    start: int
    end: int
    style: str


class SavedText(TypedDict):
    type: Literal["rich_text"]
    text: str
    style: str
    spans: list[SavedSpan]


class DecodedEntry(TypedDict, total=False):
    role: Required[Literal["user", "assistant", "tool", "notice"]]
    content: Required[str | Text]
    streaming: object
    intermediate: object
    expanded: object
    persist: object
    transient: object
    name: object
    status: object
    detail: object
    label: object
    diff: object


class EncodedEntry(TypedDict, total=False):
    role: Required[Literal["user", "assistant", "tool", "notice"]]
    content: Required[str | SavedText]
    streaming: object
    intermediate: object
    expanded: object
    persist: object
    transient: object
    name: object
    status: object
    detail: object
    label: object
    diff: object


class SavedSubagent(TypedDict):
    id: str
    description: str
    status: Literal["running", "finished", "failed", "interrupted"]
    duration: float | None
    tokens: int
    transcript: list[EncodedEntry]


class ActiveTurn(TypedDict):
    prompt: str
    tools_used: bool


class SessionRecord(TypedDict):
    version: Literal[1]
    id: str
    project_dir: str
    name: str
    created_at: str
    updated_at: str
    messages: History
    transcript: list[EncodedEntry]
    active_turn: ActiveTurn | None
    subagents: NotRequired[list[SavedSubagent]]
