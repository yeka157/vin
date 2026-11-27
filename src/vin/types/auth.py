"""Saved accounts and the ChatGPT sign-in flow."""

from threading import Event, Lock
from typing import Never, NotRequired, Protocol, TypedDict

from vin.types.data import JSONValue


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


class OpenRouterAccount(TypedDict):
    key: str
    label: str


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


class CallbackServerState(Protocol):
    server_port: int
    expected_state: str
    expected_client_id: str | None
    callback_result: AuthorizationCallback | None
    callback_error: RuntimeError | None
    callback_done: Event
    callback_lock: Lock

    def serve_forever(self, poll_interval: float = 0.5) -> None: ...
    def shutdown(self) -> None: ...
