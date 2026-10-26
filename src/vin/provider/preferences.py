import json
from pathlib import Path

from vin.auth.credentials import save_private_json

PREFERENCES_PATH = Path.home() / ".config" / "vin" / "preferences.json"

def load_model_preference():
    try:
        preference = json.loads(
            PREFERENCES_PATH.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError(
            "Could not read saved model preference."
        ) from None

    if not isinstance(preference, dict):
        raise RuntimeError("Saved model preference is invalid.")

    model = preference.get("model")

    if (preference.get("provider") != "chatgpt" or not isinstance(model, str) or not model.strip()):
        raise RuntimeError("Saved model preference is invalid.")

    return preference

def save_model_preference(provider, model):
    if provider != "chatgpt" or not isinstance(model, str) or not model.strip():
        raise RuntimeError("Invalid model preference.")

    try:
        return save_private_json(PREFERENCES_PATH, {"provider": provider, "model": model},)
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save model preference") from None
