from contextlib import contextmanager
from datetime import datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import UUID

from vin.auth.credentials import save_private_json
from vin.sessions.records import validate_session


SESSIONS_DIR = Path.home() / ".local" / "share" / "vin" / "sessions"


def project_key(project_dir):
    project_path = Path(project_dir).resolve()
    return sha256(str(project_path).encode("utf-8")).hexdigest()


def session_path(project_dir, session_id):
    if not isinstance(session_id, str):
        raise ValueError("")
    parsed_id = UUID(session_id)
    if parsed_id.version != 4 or str(parsed_id) != session_id:
        raise ValueError("Invalid session ID.")

    return SESSIONS_DIR / project_key(project_dir) / f"{session_id}.json"


@contextmanager
def session_lock(project_dir, session_id):
    lock_path = session_path(project_dir, session_id).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    with lock_path.open("a") as lock_file:
        os.fchmod(lock_file.fileno(), 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class SessionConflictError(RuntimeError):
    pass


def save_session(project_dir, record, *, expected_record=None):
    validate_session(record, project_dir)
    if expected_record is not None:
        validate_session(expected_record, project_dir)
        if expected_record["id"] != record["id"]:
            raise ValueError("Expected record belongs to another sesssion.")

    destination = session_path(project_dir, record["id"])

    try:
        with session_lock(project_dir, record["id"]):
            current = load_session(project_dir, record["id"])
            if current != expected_record:
                raise SessionConflictError(
                    "Saved session changed; resume it again before saving."
                )
            return save_private_json(destination, record)
    except SessionConflictError:
        raise
    except (OSError, RuntimeError):
        raise RuntimeError("Could not save session.") from None


def load_session(project_dir, session_id):
    destination = session_path(project_dir, session_id)

    try:
        with destination.open("r", encoding="utf-8") as saved:
            info = os.fstat(saved.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("Saved sessions must be private to your OS user.")

            record = json.load(saved)

        validate_session(record, project_dir)
        if record["id"] != session_id:
            raise ValueError("Session ID does not match its filename.")

        return record
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeError("Could not read saved session.") from None


def list_sessions(project_dir):
    directory = SESSIONS_DIR / project_key(project_dir)

    try:
        paths = list(directory.iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        raise RuntimeError("Could not list saved sessions.") from None

    records = []
    for path in paths:
        if path.suffix != ".json":
            continue

        try:
            record = load_session(project_dir, path.stem)
        except ValueError:
            continue

        if record is not None:
            records.append(record)

    return sorted(
        records,
        key=lambda record: (datetime.fromisoformat(record["updated_at"]), record["id"]),
        reverse=True,
    )
