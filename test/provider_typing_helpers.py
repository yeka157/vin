"""Typed fixtures and checked observations shared by provider contract tests."""

from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Literal, NotRequired, TypedDict
from unittest.mock import Mock

from openai.resources.responses import Responses
from openai.types.responses import (
    FunctionToolParam, ResponseIncludable,
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_output_text_param import Logprob
from openai.types.shared_params import Reasoning
from pydantic import TypeAdapter

from vin.types import Credentials, History


def credentials(access_token: str) -> Credentials:
    return {
        "issuer": "https://auth.openai.com", "subject": "test-account",
        "client_id": "test-client", "ext_agent_host_id": "test-host",
        "access_token": access_token, "scopes": ["chatgpt.tokens.use.direct"],
    }


class ResponsesDouble:
    def __init__(self) -> None:
        self.create = Mock(spec=Responses.create)


class ClientDouble:
    def __init__(self, close: Callable[[], None]) -> None:
        self.responses = ResponsesDouble()
        self.close = close
        self.exit_error: Exception | None = None

    def __enter__(self) -> ClientDouble:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if self.exit_error is not None:
            raise self.exit_error
        self.close()


type OutputItem = ResponseFunctionToolCallParam | ResponseOutputMessageParam | ResponseReasoningItemParam


class WireInputTokens(TypedDict):
    cached_tokens: int
    cache_write_tokens: NotRequired[int]


class WireOutputTokens(TypedDict):
    reasoning_tokens: int


class WireUsage(TypedDict):
    input_tokens: int
    input_tokens_details: WireInputTokens
    output_tokens: int
    output_tokens_details: WireOutputTokens
    total_tokens: int


class WireResponse(TypedDict):
    id: str
    object: Literal["response"]
    created_at: int
    model: str
    status: str
    parallel_tool_calls: bool
    tool_choice: str
    tools: list[FunctionToolParam]
    output: Sequence[OutputItem]
    usage: NotRequired[WireUsage]


class WireCompleted(TypedDict):
    type: str
    sequence_number: int
    response: WireResponse


class WireItemDone(TypedDict):
    type: str
    sequence_number: int
    output_index: int
    item: OutputItem


class WireDelta(TypedDict):
    type: str
    delta: str
    item_id: str
    output_index: int
    content_index: int
    sequence_number: int
    logprobs: list[Logprob]


class WireArgumentsDelta(TypedDict):
    type: str
    delta: str
    item_id: str
    output_index: int
    sequence_number: int


type WireEvent = WireCompleted | WireItemDone | WireDelta | WireArgumentsDelta
type HTTPFixture = list[WireEvent] | tuple[int, Mapping[str, object]]


class RequestSnapshot(TypedDict):
    model: str
    input: History
    tools: list[FunctionToolParam]
    reasoning: Reasoning
    include: list[ResponseIncludable]
    store: bool
    stream: bool
    instructions: str


REQUEST_ADAPTER = TypeAdapter(RequestSnapshot)
