import json
from hashlib import sha256
from pathlib import Path

from vin.auth.credentials import save_private_json

MODEL_CACHE_DIR = Path.home() / ".cache" / "vin" / "chatgpt"


def _cache_path(credentials):
    identity = [credentials.get(field) for field in ("issuer", "client_id", "subject")]
    if any(not isinstance(value, str) or not value.strip() for value in identity):
        raise ValueError("Missing account identity")

    account_key = sha256(json.dumps(identity).encode("utf-8")).hexdigest()
    return MODEL_CACHE_DIR / f"{account_key}.json"


def _model_metadata(models):
    if not isinstance(models, list):
        raise ValueError("Invalid model cache")

    result = []
    for model in models:
        if not isinstance(model, dict) or any(
            not isinstance(model.get(field), str) or not model[field].strip()
            for field in ("slug", "display_name")
        ):
            raise ValueError("Invalid model cache")

        result.append({"slug": model["slug"], "display_name": model["display_name"]})

    return result


def load_cached_chatgpt_models(credentials):
    try:
        payload = json.loads(_cache_path(credentials).read_text(encoding="utf-8"))
        return _model_metadata(payload)
    except (OSError, ValueError):
        return []


def save_cached_chatgpt_models(credentials, models):
    try:
        metadata = _model_metadata(models)
        save_private_json(_cache_path(credentials), metadata)
    except (OSError, ValueError, RuntimeError):
        pass
