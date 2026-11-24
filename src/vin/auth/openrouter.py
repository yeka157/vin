import json
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .credentials import save_private_json
from vin.types import OpenRouterAccount, is_record


ACCOUNT_PATH = Path.home() / ".config" / "vin" / "openrouter.json"
KEY_URL = "https://openrouter.ai/api/v1/key"


def check_openrouter_key(key: str) -> OpenRouterAccount:
    key = key.strip()
    if not key:
        raise RuntimeError("Paste an OpenRouter API key.")

    try:
        request = Request(KEY_URL, headers={"Authorization": f"Bearer {key}"})
        with cast(HTTPResponse, urlopen(request, timeout=15)) as response:
            body = cast(object, json.load(response))
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError("OpenRouter key was rejected.") from None
        raise RuntimeError(
            f"Could not check the OpenRouter key (HTTP {error.code})."
        ) from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Could not connect to OpenRouter.") from None
    except ValueError:
        raise RuntimeError("OpenRouter returned an invalid response.") from None

    data = body.get("data") if is_record(body) else None
    label = data.get("label") if is_record(data) else None
    return {"key": key, "label": label if isinstance(label, str) else ""}


def save_openrouter_account(account: OpenRouterAccount) -> Path:
    try:
        return save_private_json(ACCOUNT_PATH, dict(account))
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save the OpenRouter key.") from None


def load_openrouter_account() -> OpenRouterAccount | None:
    try:
        account = cast(object, json.loads(ACCOUNT_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError("Could not read the saved OpenRouter key.") from None

    if (
        not is_record(account)
        or not isinstance(account.get("label"), str)
        or not isinstance(account.get("key"), str)
        or not account.get("key")
    ):
        raise RuntimeError("Saved OpenRouter key is invalid.")

    return cast(OpenRouterAccount, account)


def delete_openrouter_account() -> None:
    try:
        ACCOUNT_PATH.unlink(missing_ok=True)
    except OSError:
        raise RuntimeError("Could not remove the saved OpenRouter key.") from None


def mask_key(key: str, hidden: int = 3) -> str:
    if len(key) > 13:
        return key[:9] + "•" * hidden + key[-4:]
    return "•" * len(key)
