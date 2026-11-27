"""Saved settings and the options a turn runs with."""

from typing import TypedDict


class SubagentSettings(TypedDict):
    enabled: bool
    budget: int
    fetch: bool


class SearchSettings(TypedDict):
    key: str
    enabled: bool


class ReplyOptions(TypedDict, total=False):
    compact_threshold: int
    compact_only: bool
    subagents: bool
    subagent_budget: int
    search: bool
    fetch: bool
