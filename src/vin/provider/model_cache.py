from collections.abc import Mapping
from typing import cast
from vin.types import ModelMetadata, is_record, is_list

import json
from hashlib import sha256
from pathlib import Path

from vin.auth.credentials import save_private_json

MODEL_CACHE_DIR = Path.home() / ".cache" / "vin" / "chatgpt"


def _cache_path(credentials: Mapping[str, object] | None) -> Path:
    identity = [(credentials or {}).get(field) for field in ("issuer", "client_id", "subject")]
    if any(not isinstance(value, str) or not value.strip() for value in identity):
        raise ValueError("Missing account identity")

    account_key = sha256(json.dumps(identity).encode("utf-8")).hexdigest()
    return MODEL_CACHE_DIR / f"{account_key}.json"


def _model_metadata(models: object) -> list[ModelMetadata]:
    if not is_list(models):
        raise ValueError("Invalid model cache")

    result: list[ModelMetadata] = []
    for model in models:
        if not is_record(model) or any(
            not isinstance(model.get(field), str) or not cast(str, model[field]).strip()
            for field in ("slug", "display_name")
        ):
            raise ValueError("Invalid model cache")

        result.append({"slug": cast(str, model["slug"]), "display_name": cast(str, model["display_name"])})

    return result


def load_cached_chatgpt_models(credentials: Mapping[str, object]) -> list[ModelMetadata]:
    try:
        payload = cast(object, json.loads(_cache_path(credentials).read_text(encoding="utf-8")))
        return _model_metadata(payload)
    except (OSError, ValueError):
        return []


def save_cached_chatgpt_models(credentials: Mapping[str, object] | None, models: object) -> None:
    try:
        metadata = _model_metadata(models)
        save_private_json(_cache_path(credentials), metadata)
    except (OSError, ValueError, RuntimeError):
        pass
