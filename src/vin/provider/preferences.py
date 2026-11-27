from typing import cast
from vin.types.data import is_record
from vin.types.models import ModelPreference
from vin.types.settings import SubagentSettings

import json
from pathlib import Path

from vin.auth.credentials import save_private_json

PREFERENCES_PATH = Path.home() / ".config" / "vin" / "preferences.json"
SETTINGS_PATH = Path.home() / ".config" / "vin" / "settings.json"
DEFAULT_SUBAGENT_SETTINGS: SubagentSettings = {
    "enabled": True,
    "budget": 8,
    "fetch": True,
}


def load_model_preference() -> ModelPreference | None:
    try:
        preference = cast(
            object, json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError("Could not read saved model preference.") from None

    if not is_record(preference):
        raise RuntimeError("Saved model preference is invalid.")

    model = preference.get("model")

    if (
        preference.get("provider") not in ("chatgpt", "openrouter")
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise RuntimeError("Saved model preference is invalid.")

    return cast(ModelPreference, preference)


def save_model_preference(provider: str, model: object) -> Path:
    if (
        provider not in ("chatgpt", "openrouter")
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise RuntimeError("Invalid model preference.")

    try:
        return save_private_json(
            PREFERENCES_PATH,
            {"provider": provider, "model": model},
        )
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save model preference") from None


def valid_subagent_settings(value: object) -> bool:
    if not is_record(value):
        return False
    budget = value.get("budget")
    return (
        type(value.get("enabled")) is bool
        and isinstance(budget, int)
        and not isinstance(budget, bool)
        and 1 <= budget <= 32
        and type(value.get("fetch")) is bool
    )


def load_subagent_settings() -> SubagentSettings:
    try:
        settings = cast(object, json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        if is_record(settings) and "fetch" not in settings:
            settings["fetch"] = True
    except FileNotFoundError:
        return DEFAULT_SUBAGENT_SETTINGS.copy()
    except (OSError, ValueError):
        raise RuntimeError("Could not read saved settings.") from None

    if not valid_subagent_settings(settings):
        raise RuntimeError("Saved settings are invalid.")

    return cast(SubagentSettings, settings)


def save_subagent_settings(settings: SubagentSettings) -> Path:
    if not valid_subagent_settings(settings):
        raise RuntimeError("Invalid settings.")

    try:
        return save_private_json(SETTINGS_PATH, dict(settings))
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save settings.") from None
