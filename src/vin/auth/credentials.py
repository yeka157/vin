import json
import os
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

HOST_ID_PATH = Path.home() / ".config" / "vin" / "host_id"
CREDENTIALS_DIR = HOST_ID_PATH.parent / "accounts"
ACTIVE_SESSION_PATH = HOST_ID_PATH.parent / "active_session.json"
OPENAI_ISSUER = "https://auth.openai.com"


def get_host_id():
    try:
        host_id = HOST_ID_PATH.read_text(encoding="utf-8").strip()
        parsed_id = UUID(host_id)
        if parsed_id.version != 4 or host_id != parsed_id.urn:
            raise ValueError("Saved host ID must be a UUIDv4 URN")
        return host_id
    except FileNotFoundError:
        HOST_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        host_id = uuid4().urn
        HOST_ID_PATH.write_text(host_id, encoding="utf-8")
        return host_id


def build_credential_record(tokens, identity, client_id, host_id, scopes, received_at):
    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Token response did not contain an access token")

    refresh_token = tokens.get("refresh_token")
    if "offline_access" in scopes or refresh_token is not None:
        if not isinstance(refresh_token, str) or not refresh_token:
            raise RuntimeError("Token response did not contain a valid refresh token")

    token_type = tokens.get("token_type")
    if not isinstance(token_type, str) or token_type.lower() != "bearer":
        raise RuntimeError("Expected a Bearer access token")

    expires_in = tokens.get("expires_in")
    if type(expires_in) is not int or expires_in <= 0:
        raise RuntimeError("Token response contained an invalid expiry")

    return {
        "issuer": identity["iss"],
        "subject": identity["sub"],
        "email": identity.get("email"),
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "id_token": tokens["id_token"],
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scopes": sorted(scopes),
        "saved_at": received_at.isoformat(),
    }


def save_private_json(destination, data):
    destination = Path(destination)
    temporary_path = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            os.fchmod(temporary.fileno(), 0o600)
            json.dump(data, temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, destination)
    except OSError:
        raise RuntimeError("Could not save sign-in data") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def save_credential_record(credentials):
    account_key = json.dumps(
        [
            credentials["issuer"],
            credentials["client_id"],
            credentials["subject"],
        ]
    ).encode("utf-8")
    filename = sha256(account_key).hexdigest() + ".json"
    return save_private_json(CREDENTIALS_DIR / filename, credentials)


def is_access_token_expired(credentials):
    saved_at = credentials.get("saved_at")
    expires_in = credentials.get("expires_in")
    if not isinstance(saved_at, str) or type(expires_in) is not int or expires_in <= 0:
        raise RuntimeError("Saved credentials contained invalid expiry data")

    try:
        received_at = datetime.fromisoformat(saved_at)
        if received_at.tzinfo is None:
            raise ValueError("Expected a timezone-aware timestamp")
        expires_at = received_at + timedelta(seconds=expires_in)
    except (ValueError, OverflowError):
        raise RuntimeError("Saved credentials contained invalid expiry data") from None

    return datetime.now(timezone.utc) >= expires_at


def load_credential_record(path):
    try:
        with Path(path).open("r", encoding="utf-8") as saved:
            info = os.fstat(saved.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("Saved credentials must be private to your OS user")
            credentials = json.load(saved)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError("Could not read saved sign-in credentials") from None

    if not isinstance(credentials, dict):
        raise RuntimeError("Expected a saved credential object")

    for field in ("issuer", "subject", "client_id", "ext_agent_host_id"):
        value = credentials.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Saved credentials contained an invalid {field}")

    if credentials["issuer"] != OPENAI_ISSUER:
        raise RuntimeError("Saved credentials have an unexpected issuer")
    if credentials["client_id"] == "dynamic_agent_client":
        raise RuntimeError("Saved credentials must use an issued client ID")

    if all(
        field not in credentials
        for field in ("access_token", "refresh_token", "id_token")
    ):
        return credentials

    if not isinstance(credentials.get("id_token"), str) or not credentials["id_token"]:
        raise RuntimeError("Saved credentials contained an invalid id_token")

    scopes = credentials.get("scopes")
    if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
        raise RuntimeError("Saved credentials contained invalid scopes")

    is_access_token_expired(credentials)
    identity = {
        "iss": credentials["issuer"],
        "sub": credentials["subject"],
        "email": credentials.get("email"),
    }

    return build_credential_record(
        credentials,
        identity,
        credentials["client_id"],
        credentials["ext_agent_host_id"],
        set(scopes),
        datetime.fromisoformat(credentials["saved_at"]),
    )


def load_active_session():
    try:
        with ACTIVE_SESSION_PATH.open("r", encoding="utf-8") as saved:
            info = os.fstat(saved.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("Saved session must be private to your OS user")
            session = json.load(saved)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError("Could not read the saved session") from None

    if not isinstance(session, dict):
        raise RuntimeError("Expected a saved session object")

    filename = session.get("account_file")
    if (
        not isinstance(filename, str)
        or len(filename) != 69
        or not filename.endswith(".json")
        or any(char not in "0123456789abcdef" for char in filename[:-5])
    ):
        raise RuntimeError("Saved session contained an invalid account filename")

    return load_credential_record(CREDENTIALS_DIR / filename)


def list_saved_accounts():
    try:
        paths = sorted(CREDENTIALS_DIR.iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        raise RuntimeError("Could not list saved accounts") from None

    accounts = []
    for path in paths:
        if path.suffix != ".json":
            continue
        credentials = load_credential_record(path)
        if credentials is not None:
            accounts.append(credentials)
    return accounts
