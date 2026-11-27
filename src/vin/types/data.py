"""JSON values and the checks that narrow untyped external data."""

from collections.abc import Mapping
from os import PathLike
from typing import TypeIs, cast

type Pathish = str | PathLike[str]
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[
    str, JSONValue
]
type ExternalRecord = Mapping[str, object]


def is_record(value: object) -> TypeIs[dict[str, object]]:
    return isinstance(value, dict) and all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    )


def is_dictionary(value: object) -> TypeIs[dict[object, object]]:
    return isinstance(value, dict)


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
