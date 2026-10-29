"""The synchronous Tools surface used by Vin (aisuite 0.2.0)."""
from collections.abc import Callable
from typing import Generic, Literal, TypeVar, TypedDict

_Result = TypeVar("_Result")

class FunctionCall(TypedDict):
    name: str
    arguments: str

class ToolCall(TypedDict):
    function: FunctionCall

class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]

class OpenAITool(TypedDict):
    type: Literal["function"]
    function: FunctionSchema

class Tools(Generic[_Result]):
    def __init__(self, tools: list[Callable[..., _Result]] | None = None) -> None: ...
    def tools(self, format: str = "openai") -> list[OpenAITool]: ...
    def execute(self, tool_calls: ToolCall | list[ToolCall]) -> list[_Result]: ...
