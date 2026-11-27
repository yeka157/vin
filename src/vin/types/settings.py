"""Saved settings and the options a turn runs with."""

from typing import TypedDict


class Settings(TypedDict):
    subagents: bool
    subagent_budget: int
    search: bool
    fetch: bool


class SettingsChange(TypedDict, total=False):
    subagents: bool
    subagent_budget: int
    search: bool
    fetch: bool


class ReplyOptions(TypedDict, total=False):
    compact_threshold: int
    compact_only: bool
    subagents: bool
    subagent_budget: int
    search: bool
    fetch: bool
