__all__ = ["LAUNCH_DIR", "MAX_FILE_BYTES", "MAX_OUTPUT_CHARS", "FILE_MUTATION_LOCK", "TOOL_REGISTRY", "read", "write", "edit", "bash", "_tool_cancellation", "_response_tools", "_check_cancelled", "_truncate_output"]

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
import difflib
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import tempfile
from threading import Event, Lock
import time
import unicodedata
from aisuite.utils.tools import Tools
from openai.types.responses import FunctionToolParam

LAUNCH_DIR = Path.cwd().resolve()
MAX_FILE_BYTES = 2_000_000
MAX_OUTPUT_CHARS = 20_000
FILE_MUTATION_LOCK = Lock()
_TOOL_CANCEL_EVENT: ContextVar[Event | None] = ContextVar(
    "vin_tool_cancel_event", default=None
)


@contextmanager
def _tool_cancellation(cancel_event: Event) -> Generator[None]:
    token = _TOOL_CANCEL_EVENT.set(cancel_event)

    try:
        yield
    finally:
        _TOOL_CANCEL_EVENT.reset(token)


def _check_cancelled() -> None:
    cancel_event = _TOOL_CANCEL_EVENT.get()
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Tool execution cancelled.")


def _resolve_path(path: str) -> Path:
    resolved = (LAUNCH_DIR / path).resolve()

    if resolved == LAUNCH_DIR or not resolved.is_relative_to(LAUNCH_DIR):
        raise ValueError("Path must point to a file inside the launch directory.")

    return resolved


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    marker = "\n[output truncated at 20,000 characters]"
    return text[: MAX_OUTPUT_CHARS - len(marker)] + marker


def read(path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a UTF-8 workspace file with line numbers."""
    if offset < 1:
        raise ValueError("Offset must be at least 1.")

    if not 1 <= limit <= 2_000:
        raise ValueError("Limit must be between 1 and 2,000")

    resolved = _resolve_path(path)

    with resolved.open("rb") as file:
        data = file.read(MAX_FILE_BYTES + 1)

    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 2 MB limit.")

    lines = data.decode("utf-8-sig").splitlines()

    start = offset - 1
    selected = lines[start : start + limit]

    output = "\n".join(
        f"{number}: {line}" for number, line in enumerate(selected, start=offset)
    )

    return _truncate_output(output)


def _atomic_write(
    target: Path,
    data: bytes,
    *,
    expected: tuple[bytes, os.stat_result] | None = None,
) -> None:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 2 MB limit.")

    target.parent.mkdir(parents=True, exist_ok=True)

    mode = None
    if target.exists():
        mode = stat.S_IMODE(target.stat().st_mode)

    descriptor, name = tempfile.mkstemp(
        prefix=".vin-",
        dir=target.parent,
    )
    temporary = Path(name)

    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)

            if mode is not None:
                os.fchmod(file.fileno(), mode)

            file.flush()
            os.fsync(file.fileno())

        if _resolve_path(str(target)) != target:
            raise ValueError("Path changed before saving.")

        if expected is not None:
            original_data, original_stat = expected

            try:
                with target.open("rb") as current:
                    current_data = current.read(MAX_FILE_BYTES + 1)
                    current_stat = os.fstat(current.fileno())
            except FileNotFoundError:
                raise ValueError("File changed before saving; read it again.") from None

            fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_mode",
            )

            metadata_changed = any(
                getattr(current_stat, field) != getattr(original_stat, field)
                for field in fields
            )

            if current_data != original_data or metadata_changed:
                raise ValueError("File changed before saving; read it again.")

        os.replace(temporary, target)

    finally:
        temporary.unlink(missing_ok=True)


def write(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 workspace file."""
    data = content.encode("utf-8")

    with FILE_MUTATION_LOCK:
        target = _resolve_path(path)
        _atomic_write(target, data)

    return f"Wrote {len(data)} bytes to {path}."


def _normalize_for_match(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    text = "\n".join(line.rstrip() for line in text.split("\n"))

    text = re.sub(r"[\u2018-\u201b]", "'", text)
    text = re.sub(r"[\u201c-\u201f]", '"', text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(
        r"[\u00a0\u2002-\u200a\u202f\u205f\u3000]",
        " ",
        text,
    )

    return text


def _find_unique_match(
    content: str,
    old_text: str,
) -> tuple[str, int, int]:
    if not old_text.strip():
        raise ValueError("Search text must not be blank.")

    exact_start = content.find(old_text)

    normalized_content = _normalize_for_match(content)
    normalized_old = _normalize_for_match(old_text)

    if not normalized_old.strip():
        raise ValueError("Normalized search text must not be blank.")

    normalized_start = normalized_content.find(normalized_old)

    exact_duplicates = exact_start >= 0 and content.find(old_text, exact_start + 1) >= 0

    normalized_duplicates = (
        normalized_start >= 0
        and normalized_content.find(normalized_old, normalized_start + 1) >= 0
    )

    if exact_duplicates or normalized_duplicates:
        raise ValueError("Ambiguous match; include more context.")

    if exact_start >= 0:
        return content, exact_start, len(old_text)

    if normalized_start < 0:
        raise ValueError("Search text was not found.")

    return normalized_content, normalized_start, len(normalized_old)


def _replace_match(
    original: str,
    base: str,
    start: int,
    length: int,
    new_text: str,
    *,
    newline: str = "\n",
) -> str:
    if base == original:
        return original[:start] + new_text + original[start + length :]

    original_lines = original.split("\n")
    base_lines = base.split("\n")

    if len(original_lines) != len(base_lines):
        raise ValueError("Matching views must have the same line count.")

    first = base.count("\n", 0, start)
    last = base.count("\n", 0, start + length - 1)

    block_start = sum(len(line) + 1 for line in base_lines[:first])

    block = "\n".join(base_lines[first : last + 1])
    if last < len(base_lines) - 1:
        block += "\n"

    relative_start = start - block_start
    changed_block = block[:relative_start] + new_text + block[relative_start + length :]
    changed_block = changed_block.replace("\n", newline)

    prefix = "\n".join(original_lines[:first])
    if first > 0:
        prefix += "\n"

    suffix = "\n".join(original_lines[last + 1 :])
    return prefix + changed_block + suffix


def edit(path: str, old_text: str, new_text: str) -> str:
    """Replace one unique text occurrence and return a unified diff."""
    with FILE_MUTATION_LOCK:
        target = _resolve_path(path)

        with target.open("rb") as file:
            data = file.read(MAX_FILE_BYTES + 1)
            metadata = os.fstat(file.fileno())

        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 2 MB limit.")

        decoded = data.decode("utf-8")
        bom = "\ufeff" if decoded.startswith("\ufeff") else ""
        original = decoded[len(bom) :]

        first_newline = original.find("\n")
        newline = (
            "\r\n"
            if first_newline > 0 and original[first_newline - 1] == "\r"
            else "\n"
        )

        plain = original.replace("\r\n", "\n")
        old = old_text.replace("\r\n", "\n").replace("\r", "\n")
        new = new_text.replace("\r\n", "\n").replace("\r", "\n")

        base, start, length = _find_unique_match(plain, old)
        changed = _replace_match(original, base, start, length, new, newline=newline)

        if changed == original:
            raise ValueError("Replacement makes no change.")

        diff_lines = difflib.unified_diff(
            plain.splitlines(keepends=True),
            changed.replace("\r\n", "\n").splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )

        diff = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in diff_lines
        )

        if _resolve_path(path) != target:
            raise ValueError("Path changed before saving.")

        _atomic_write(
            target,
            (bom + changed).encode("utf-8"),
            expected=(data, metadata),
        )

        return _truncate_output(diff)


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    process.wait()


def bash(command: str, timeout: int = 30) -> str:
    """Run noninteractive Bash in the launch directory."""
    if not 1 <= timeout <= 300:
        raise ValueError("Timeout must be between 1 and 300 seconds.")

    _check_cancelled()

    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)

    process = subprocess.Popen(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        cwd=LAUNCH_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )

    assert process.stdout is not None
    output = bytearray()
    capture_limit = MAX_OUTPUT_CHARS * 4
    timed_out = False
    completed = False

    try:
        deadline = time.monotonic() + timeout

        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)

            while selector.get_map() or process.poll() is None:
                _check_cancelled()

                if time.monotonic() >= deadline:
                    timed_out = True
                    break

                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 65_536)

                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue

                    remaining = capture_limit - len(output)

                    if remaining > 0:
                        output.extend(chunk[:remaining])
        completed = not timed_out
    finally:
        if not completed:
            _stop_process_group(process)
        else:
            process.wait()

        process.stdout.close()

    text = output.decode("utf-8", errors="replace")
    notice = f"Command timed out after {timeout} seconds.\n" if timed_out else ""

    return _truncate_output(f"Exit code: {process.returncode}\n{notice}{text}")


TOOL_REGISTRY = Tools([read, write, edit, bash])


def _response_tools() -> list[FunctionToolParam]:
    schemas: list[FunctionToolParam] = []

    for tool in TOOL_REGISTRY.tools():
        function = tool["function"]
        schemas.append(
            {
                "type": "function",
                **function,
                "strict": False,
            }
        )

    return schemas
