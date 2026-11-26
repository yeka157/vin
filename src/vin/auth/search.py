import json
from pathlib import Path
from typing import cast

from vin.types import is_record

SEARCH_KEY_PATH = Path.home() / ".config" / "vin" / "search.json"


def load_search_key() -> str | None:
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
    return key.strip()
