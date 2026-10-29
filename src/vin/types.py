"""Named application records and checked boundaries for external data."""

from collections.abc import Mapping
from os import PathLike
from typing import Literal, Never, NotRequired, Required, TypeIs, TypedDict, cast

from rich.console import RenderableType
from rich.text import Text


type Pathish = str | PathLike[str]
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
# Provider history deliberately preserves opaque SDK output and future provider fields.
type ProviderItem = dict[str, JSONValue]
type History = list[ProviderItem]


def is_record(value: object) -> TypeIs[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in cast(dict[object, object], value))


def is_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def is_string_list(value: object) -> TypeIs[list[str]]:
    return is_list(value) and all(isinstance(item, str) for item in value)


def is_json(value: object) -> TypeIs[JSONValue]:
    def visit(item: object, parents: set[int]) -> bool:
        if item is None or isinstance(item, (str, bool, int, float)):
            return True
        if is_list(item) or is_record(item):
            identity = id(item)
            if identity in parents:
                return False
            parents.add(identity)
            children = item.values() if is_record(item) else item
            result = all(visit(child, parents) for child in children)
            parents.remove(identity)
            return result
        return False

    return visit(value, set())


def json_value(value: object) -> JSONValue:
    if not is_json(value):
        raise ValueError("Value is not JSON data.")
    return value


class Credentials(TypedDict):
    issuer: str
    subject: str
    client_id: str
    ext_agent_host_id: str
    email: NotRequired[JSONValue]
    id_token: NotRequired[JSONValue]
    access_token: NotRequired[str]
    refresh_token: NotRequired[str | None]
    token_type: NotRequired[str]
    expires_in: NotRequired[int]
    scopes: NotRequired[list[str]]
    saved_at: NotRequired[str]


class Registration(TypedDict):
    """Saved identity without tokens; remaining JSON fields are unvalidated."""

    issuer: str
    subject: str
    client_id: str
    ext_agent_host_id: str
    email: NotRequired[JSONValue]
    access_token: NotRequired[Never]
    refresh_token: NotRequired[Never]
    id_token: NotRequired[Never]
    scopes: NotRequired[JSONValue]
    expires_in: NotRequired[JSONValue]
    saved_at: NotRequired[JSONValue]
    token_type: NotRequired[JSONValue]


type AccountRecord = Credentials | Registration


class ModelMetadata(TypedDict):
    slug: str
    display_name: str


class ModelPreference(TypedDict):
    provider: Literal["chatgpt"]
    model: str


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


class SavedSpan(TypedDict):
    start: int
    end: int
    style: str


class SavedText(TypedDict):
    type: Literal["rich_text"]
    text: str
    style: str
    spans: list[SavedSpan]


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


type ExternalRecord = Mapping[str, object]


class Identity(TypedDict):
    iss: str
    sub: str
    email: NotRequired[JSONValue]


class LoginAttempt(TypedDict):
    state: str
    nonce: str
    code_verifier: str
    code_challenge: str


class AuthorizationCallback(TypedDict):
    code: str
    client_id: str


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


def is_dictionary(value: object) -> TypeIs[dict[object, object]]:
    return isinstance(value, dict)


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
