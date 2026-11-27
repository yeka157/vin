"""The part of the OpenAI SDK's Responses client that Vin calls."""

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from vin.types.history import History

if TYPE_CHECKING:
    from openai.types.responses import (
        FunctionToolParam,
        ResponseIncludable,
        ResponseStreamEvent,
    )
    from openai.types.shared_params import Reasoning


class WireResponses(Protocol):
    def create(
        self,
        *,
        model: str,
        input: History,
        instructions: str,
        tools: list["FunctionToolParam"],
        reasoning: "Reasoning",
        include: list["ResponseIncludable"],
        store: bool,
        stream: Literal[True],
    ) -> AbstractContextManager[Iterable["ResponseStreamEvent"]]: ...


@runtime_checkable
class WireClient(Protocol):
    responses: WireResponses
