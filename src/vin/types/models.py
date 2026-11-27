"""Providers, their models and the saved model choice."""

from typing import Literal, NotRequired, TypedDict

type Provider = Literal["chatgpt", "openrouter"]


class ModelMetadata(TypedDict):
    slug: str
    display_name: str
    context_window: NotRequired[int]
    input_price: NotRequired[float]
    output_price: NotRequired[float]
    features: NotRequired[list[str]]
    provider: Provider


class ModelPreference(TypedDict):
    provider: Provider
    model: str
