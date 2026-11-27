from http.client import HTTPResponse
import json
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vin.auth.credentials import save_private_json
from vin.types.data import is_record
from vin.types.settings import SearchSettings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SEARCH_KEY_PATH = Path.home() / ".config" / "vin" / "search.json"


def load_search_key() -> str | None:
    settings = load_search_settings()
    return settings["key"] if settings is not None else None


def load_search_settings() -> SearchSettings | None:
    try:
        account = cast(object, json.loads(SEARCH_KEY_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except OSError, ValueError:
        raise RuntimeError("Could not read the saved web search key.") from None

    if not is_record(account):
        raise RuntimeError("Saved web search key is invalid.")
    key = account.get("key")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("Saved web search key is invalid.")
    enabled = account.get("enabled")
    if type(enabled) is not bool:
        raise RuntimeError("Saved web search key is invalid.")
    return {"key": key.strip(), "enabled": enabled}


def save_search_settings(settings: SearchSettings) -> Path:
    try:
        return save_private_json(SEARCH_KEY_PATH, dict(settings))
    except OSError, RuntimeError:
        raise RuntimeError("Could not save the web search key.") from None


def delete_search_settings() -> None:
    try:
        SEARCH_KEY_PATH.unlink(missing_ok=True)
    except OSError:
        raise RuntimeError("Could not remove the saved web search key.") from None


def check_search_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise RuntimeError("Paste a Tavily API key.")

    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(
            {"query": "vin", "max_results": 1, "search_depth": "basic"}
        ).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with cast(HTTPResponse, urlopen(request, timeout=15)) as response:
            _ = response.read()
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError("Tavily key was rejected.") from None
        raise RuntimeError(
            f"Could not check the Tavily key (HTTP {error.code})."
        ) from None
    except OSError:
        raise RuntimeError("Could not connect to Tavily.") from None

    return key
