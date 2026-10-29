"""Assertions that narrow decoded session data in persistence tests."""

import json
from typing import TypeVar, cast

from rich.text import Text

from vin.types import DecodedEntry, JSONValue, TranscriptEntry, is_json, is_list, is_record
from vin.sessions.records import new_session, validate_session
from vin.sessions.transcript import encode_transcript

T = TypeVar("T")


def present(value: T | None) -> T:
    assert value is not None
    return value


def record_value(value: object) -> dict[str, JSONValue]:
    assert is_json(value)
    assert isinstance(value, dict)
    return value


def list_value(value: object) -> list[object]:
    assert is_list(value)
    return value


def text_value(value: object) -> Text:
    assert isinstance(value, Text)
    return value


def string_value(value: object) -> str:
    assert isinstance(value, str)
    return value


def mapping_value(value: object) -> dict[str, object]:
    assert is_record(value)
    return value


def decoded_json(text: str) -> object:
    return cast(object, json.loads(text))


def validated_transcript(entries: list[DecodedEntry]) -> list[TranscriptEntry]:
    record = new_session(".", "Decoded transcript fixture")
    record["transcript"] = encode_transcript(entries)
    validate_session(record, record["project_dir"])
    # Session validation checks role-specific fields and flags on these entries.
    return cast(list[TranscriptEntry], entries)
