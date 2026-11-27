"""What the transcript shows: entries, tool groups and subagents."""

from typing import Literal, NotRequired, TypedDict

from rich.console import RenderableType
from rich.text import Text


class EntryFlags(TypedDict):
    streaming: NotRequired[bool]
    intermediate: NotRequired[bool]
    expanded: NotRequired[bool]
    persist: NotRequired[bool]
    transient: NotRequired[bool]


class UserEntry(EntryFlags):
    role: Literal["user"]
    content: str


class AssistantEntry(EntryFlags):
    role: Literal["assistant"]
    content: str


class ToolEntry(EntryFlags):
    role: Literal["tool"]
    content: str | Text
    name: str
    status: Literal["running", "finished", "failed", "interrupted"]
    detail: NotRequired[str]
    label: NotRequired[str]
    diff: NotRequired[str]


class NoticeEntry(EntryFlags):
    role: Literal["notice"]
    content: RenderableType


type TranscriptEntry = UserEntry | AssistantEntry | ToolEntry | NoticeEntry


class ToolGroup(TypedDict):
    role: Literal["tool_group"]
    entries: list[ToolEntry]


type GroupedEntry = TranscriptEntry | ToolGroup
type GroupHeader = tuple[int, int, ToolEntry]


class Subagent(TypedDict):
    id: str
    description: str
    status: Literal["running", "finished", "failed", "interrupted"]
    started_at: float
    duration: float | None
    tokens: int
    transcript: list[TranscriptEntry]
