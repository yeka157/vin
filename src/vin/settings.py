"""Saved settings in ~/.config/vin/settings.json, one flat record for every topic in /settings.

Secrets stay in their own files (the web search key is in search.json).
"""

import json
from pathlib import Path
from typing import cast

from vin.auth.credentials import save_private_json
from vin.types.data import is_record
from vin.types.settings import Settings

SETTINGS_PATH = Path.home() / ".config" / "vin" / "settings.json"
DEFAULT_SETTINGS: Settings = {
    "subagents": True,
    "subagent_budget": 8,
    "search": False,
    "fetch": True,
}
# Names an older Vin saved under, mapped to the current ones.
_RENAMED = {"enabled": "subagents", "budget": "subagent_budget"}


def valid_settings(value: object) -> bool:
    if not is_record(value) or set(value) != set(DEFAULT_SETTINGS):
        return False
    budget = value["subagent_budget"]
    return (
        all(type(value[name]) is bool for name in ("subagents", "search", "fetch"))
        and type(budget) is int
        and 1 <= budget <= 32
    )


def load_settings() -> Settings:
    saved: object
    try:
        saved = cast(object, json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        saved = dict[str, object]()
    except (OSError, ValueError):
        raise RuntimeError("Could not read saved settings.") from None

    if not is_record(saved):
        raise RuntimeError("Saved settings are invalid.")
    settings = {_RENAMED.get(name, name): value for name, value in saved.items()}
    if "search" not in settings:
        from vin.auth.search import saved_search_enabled

        settings["search"] = saved_search_enabled()
    settings = {**DEFAULT_SETTINGS, **settings}
    if not valid_settings(settings):
        raise RuntimeError("Saved settings are invalid.")
    return cast(Settings, settings)


def save_settings(settings: Settings) -> Path:
    if not valid_settings(settings):
        raise RuntimeError("Invalid settings.")

    try:
        return save_private_json(SETTINGS_PATH, dict(settings))
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save settings.") from None
