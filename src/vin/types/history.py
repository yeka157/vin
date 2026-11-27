"""The conversation history sent to a model."""

from vin.types.data import JSONValue

# Provider history deliberately preserves opaque SDK output and future provider fields.
type ProviderItem = dict[str, JSONValue]
type History = list[ProviderItem]
