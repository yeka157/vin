"""Saved conversations: storage, records, recovery, rename and transcript encoding."""

from contextlib import AbstractContextManager
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from prompt_toolkit.buffer import Buffer
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.text import Text

from session_typing_helpers import decoded_json, list_value, mapping_value, present, record_value, text_value, validated_transcript
from vin.sessions import storage
from vin.sessions.records import new_session, snapshot_session, validate_session
from vin.sessions.recovery import restore_session
from vin.sessions.transcript import decode_content, decode_transcript, encode_transcript
from vin.tui import conversations
from vin.tui.app import VinApp
from vin.types.data import JSONValue, Pathish
from vin.types.history import History
from vin.types.sessions import ActiveTurn, DecodedEntry, SessionRecord
from vin.types.transcript import Subagent, ToolEntry, TranscriptEntry


def probe_file_lock(path: Pathish) -> str:
    """Try an OS lock from another process without blocking or timing sleeps."""
    script = '''
import fcntl
import sys
with open(sys.argv[1], "a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("blocked")
    else:
        print("acquired")
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)],
                            capture_output=True, text=True, timeout=5, check=True)
    return result.stdout.strip()


class SessionProjectIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def key(self, directory: Pathish) -> str:
        key = storage.project_key(directory)
        self.assertIsInstance(key, str, "project_key must return a hexadecimal string.")
        self.assertRegex(key, r"\A[0-9a-f]{64}\Z")
        return key

    def test_same_named_projects_in_different_locations_have_distinct_keys(self) -> None:
        first = self.root / "repository" / "vin"
        second = self.root / "experiments" / "vin"
        first.mkdir(parents=True)
        second.mkdir(parents=True)

        first_key = self.key(first)
        second_key = self.key(second)

        self.assertNotEqual(first_key, second_key)

    def test_equivalent_path_forms_keep_the_same_project_identity(self) -> None:
        project = self.root / "project with spaces"
        child = project / "child"
        child.mkdir(parents=True)

        expected = self.key(project.resolve())

        self.assertEqual(self.key(str(project)), expected)
        self.assertEqual(self.key(child / ".."), expected)
        self.assertEqual(self.key(project), expected)


class SessionFilePathTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "session-data"
        self.project = self.root / "project"
        self.project.mkdir()
        self.storage = storage
        location = patch.object(self.storage, "SESSIONS_DIR", self.destination, create=True)
        location.start()
        self.addCleanup(location.stop)

    def path(self, project: Pathish, session_id: object) -> Path:
        return self.storage.session_path(project, session_id)


    def test_invalid_or_noncanonical_ids_cannot_become_session_paths(self) -> None:
        session_id = "12345678-1234-4abc-8def-123456789abc"
        for invalid in (
            None, 42, "", "../outside", f"../{session_id}",
            f"/tmp/{session_id}", f"{session_id}.json", session_id.upper(),
            session_id.replace("-", ""), "{" + session_id + "}",
            "00000000-0000-0000-0000-000000000000",
        ):
            with self.subTest(session_id=invalid):
                with self.assertRaises(ValueError):
                    self.path(self.project, invalid)
        self.assertFalse(self.destination.exists())


class SessionSaveTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions import storage
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.destination = self.root / "session-data"
        self.storage = storage
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)
        self.record = new_session(self.project, "Review the queue")

    def save(self, record: SessionRecord, *, expected_record: SessionRecord | None = None) -> Path:
        return self.storage.save_session(self.project, record, expected_record=expected_record)


    def test_save_writes_private_json_at_the_session_path_without_mutation(self) -> None:
        from rich.text import Text
        from vin.sessions.transcript import encode_transcript

        self.record["messages"] = [
            {"role": "user", "content": "Review 界"},
            {"type": "reasoning", "encrypted_content": "fake-encrypted-reasoning"},
        ]
        self.record["transcript"] = encode_transcript([
            {"role": "user", "content": "Review 界"},
            {"role": "tool", "name": "edit", "status": "finished",
             "content": Text("Edited", style="dim"), "diff": "-old\n+new\n"},
        ])
        original = deepcopy(self.record)

        saved_path = self.save(self.record)

        expected = self.storage.session_path(self.project, self.record["id"])
        self.assertEqual(saved_path, expected)
        self.assertEqual(json.loads(expected.read_text(encoding="utf-8")), original)
        self.assertEqual(stat.S_IMODE(expected.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(expected.parent.stat().st_mode), 0o700)
        self.assertEqual(self.record, original)
        self.assertEqual(set(expected.parent.iterdir()), {expected, expected.with_suffix(".lock")})


    def test_failed_atomic_replacement_preserves_previous_file_and_cleans_temporary(self) -> None:
        path = self.save(self.record)
        previous = path.read_bytes()
        changed = self.record.copy()
        changed["name"] = "Renamed session"

        with patch("os.replace", side_effect=OSError("Replacement failed")):
            with self.assertRaises(RuntimeError) as error:
                self.save(changed, expected_record=self.record)

        self.assertIn("session", str(error.exception).lower())
        self.assertNotIn("sign-in", str(error.exception).lower())
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(set(path.parent.iterdir()), {path, path.with_suffix(".lock")})
        self.assertEqual(probe_file_lock(path.with_suffix(".lock")), "acquired")


class SessionLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions import storage
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.storage = storage
        self.destination = self.root / "session-data"
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)
        self.record = new_session(self.project, "Review the queue")

    def load(self, session_id: object) -> SessionRecord | None:
        return self.storage.load_session(self.project, session_id)


    def test_corrupt_invalid_or_mismatched_records_raise_session_errors(self) -> None:
        path = self.storage.save_session(self.project, self.record)
        cases = ["{broken", "[]", json.dumps(dict(self.record, version=2)),
                 json.dumps(dict(self.record, id=str(uuid4()))),
                 json.dumps(dict(self.record, project_dir=str(self.project / "other"))),
                 json.dumps(dict(self.record, messages=[{"number": float("nan")}])),
                 json.dumps(dict(self.record, active_turn={}))]
        for contents in cases:
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(RuntimeError) as error:
                    self.load(self.record["id"])
                self.assertIn("session", str(error.exception).lower())
                self.assertEqual(path.read_text(encoding="utf-8"), contents)

    def test_nonprivate_or_foreign_owned_session_is_rejected(self) -> None:
        path = self.storage.save_session(self.project, self.record)
        for mode in (0o644, 0o620):
            with self.subTest(mode=oct(mode)):
                path.chmod(mode)
                with self.assertRaises(RuntimeError):
                    self.load(self.record["id"])
        path.chmod(0o600)
        info = SimpleNamespace(st_uid=os.getuid() + 1, st_mode=0o100600)
        with patch("os.fstat", return_value=info):
            with self.assertRaises(RuntimeError):
                self.load(self.record["id"])


class SessionListTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions import storage

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.storage = storage
        self.destination = self.root / "session-data"
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)

    def listing(self) -> list[SessionRecord]:
        return self.storage.list_sessions(self.project)


    def record(self, project: Pathish | None = None) -> SessionRecord:
        from vin.sessions.records import new_session

        return new_session(self.project if project is None else project, "Same name")


    def test_only_current_project_sessions_are_listed_newest_first_by_instant(self) -> None:
        records: list[SessionRecord] = []
        for number, updated in ((1, "2026-10-24T10:00:00+00:00"),
                                (2, "2026-10-24T16:00:00+07:00"),
                                (3, "2026-10-24T10:00:00Z")):
            record = self.record()
            record.update(id=f"00000000-0000-4000-8000-{number:012d}",
                          created_at="2026-10-24T00:00:00Z", updated_at=updated)
            self.storage.save_session(self.project, record)
            records.append(record)
        other = self.root / "another-project"
        other.mkdir()
        self.storage.save_session(other, self.record(other))
        directory = self.destination / self.storage.project_key(self.project)
        previous = {path.name: path.read_bytes() for path in directory.iterdir()}

        listed = self.listing()

        self.assertEqual(listed, [records[2], records[0], records[1]])
        self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, previous)
        listed[0]["name"] = "Changed in memory"
        self.assertEqual(self.listing()[0], records[2])


class SessionLockTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions import storage

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.storage = storage
        self.destination = self.root / "session-data"
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)
        self.session_id = str(uuid4())

    def lock(self, session_id: object = None) -> AbstractContextManager[None]:
        return self.storage.session_lock(self.project, self.session_id if session_id is None else session_id)


    def probe(self, path: Pathish) -> str:
        return probe_file_lock(path)

    def test_private_lock_excludes_other_process_until_context_exits(self) -> None:
        path = self.storage.session_path(self.project, self.session_id)
        lock_path = path.with_suffix(".lock")

        with self.lock():
            self.assertFalse(path.exists(), "Taking a lock must not save a session.")
            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(lock_path.parent.stat().st_mode), 0o700)
            self.assertEqual(self.probe(lock_path), "blocked")

        self.assertTrue(lock_path.exists(), "Keep a stable lock file for future writers.")
        self.assertEqual(self.probe(lock_path), "acquired")


class SessionConflictTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions import storage
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()
        self.storage = storage
        self.destination = Path(temporary.name) / "session-data"
        location = patch.object(storage, "SESSIONS_DIR", self.destination)
        location.start()
        self.addCleanup(location.stop)
        self.record = new_session(self.project, "Review the queue")

    def test_update_requires_exact_previous_record_and_preserves_inputs(self) -> None:
        path = self.storage.save_session(self.project, self.record)
        baseline = deepcopy(self.record)
        updated = self.record.copy()
        updated["name"] = "Renamed session"
        original_updated = deepcopy(updated)

        result = self.storage.save_session(self.project, updated, expected_record=baseline)

        self.assertEqual(result, path)
        self.assertEqual(self.storage.load_session(self.project, self.record["id"]), updated)
        self.assertEqual(baseline, self.record)
        self.assertEqual(updated, original_updated)
        previous = path.read_bytes()
        stale = updated.copy()
        stale["name"] = "Stale"
        for kwargs in ({}, {"expected_record": baseline}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RuntimeError) as error:
                    self.storage.save_session(self.project, stale, **kwargs)
                self.assertIn("changed", str(error.exception).lower())
                self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(probe_file_lock(path.with_suffix(".lock")), "acquired")

    def test_another_process_update_cannot_be_overwritten_by_stale_parent(self) -> None:
        path = self.storage.save_session(self.project, self.record)
        script = '''
import json
from pathlib import Path
import sys
from vin.sessions import storage
storage.SESSIONS_DIR = Path(sys.argv[1])
baseline = json.load(sys.stdin)
updated = dict(baseline, name="Saved by another process")
storage.save_session(sys.argv[2], updated, expected_record=baseline)
'''
        child = subprocess.run([sys.executable, "-c", script, str(self.destination), str(self.project)],
                               input=json.dumps(self.record), capture_output=True, text=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr)
        previous = path.read_bytes()
        with self.assertRaises(RuntimeError) as error:
            stale = self.record.copy()
            stale["name"] = "Stale parent"
            self.storage.save_session(self.project, stale,
                                      expected_record=self.record)
        self.assertIn("changed", str(error.exception).lower())
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(json.loads(previous)["name"], "Saved by another process")


    def test_expected_existing_record_cannot_recreate_missing_or_overwrite_corrupt_file(self) -> None:
        path = self.storage.save_session(self.project, self.record)
        path.unlink()
        with self.assertRaises(RuntimeError):
            self.storage.save_session(self.project, self.record, expected_record=self.record)
        self.assertFalse(path.exists())

        path.write_text("{broken", encoding="utf-8")
        path.chmod(0o600)
        with self.assertRaises(RuntimeError):
            self.storage.save_session(self.project, self.record, expected_record=self.record)
        self.assertEqual(path.read_text(encoding="utf-8"), "{broken")


class NewSessionRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()

    def create(self, first_message: object) -> SessionRecord:
        return new_session(self.project, first_message)

    def test_new_record_has_project_metadata_and_empty_conversation_data(self) -> None:
        before = datetime.now(timezone.utc)
        record = self.create("Review the queue")
        after = datetime.now(timezone.utc)

        self.assertEqual(record["version"], 1)
        self.assertEqual(record["project_dir"], str(self.project.resolve()))
        self.assertEqual(record["name"], "Review the queue")
        identity = UUID(record["id"])
        self.assertEqual(identity.version, 4)
        self.assertEqual(str(identity), record["id"])
        created = datetime.fromisoformat(record["created_at"])
        offset = created.utcoffset()
        self.assertIsNotNone(offset)
        assert offset is not None
        self.assertEqual(offset.total_seconds(), 0)
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, after)
        self.assertEqual(record["updated_at"], record["created_at"])
        self.assertEqual(record["messages"], [])
        self.assertEqual(record["transcript"], [])
        self.assertIsNone(record["active_turn"])
        self.assertEqual(json.loads(json.dumps(record)), record)


    def test_auto_name_collapses_whitespace_and_limits_long_prompts(self) -> None:
        record = self.create("  Review\n\t the   queue  ")
        self.assertEqual(record["name"], "Review the queue")
        long_prompt = "Check " + "界" * 100
        record = self.create(long_prompt)
        self.assertEqual(record["name"], long_prompt[:80])


class SessionRecordValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.record = new_session(self.project, "Review the queue")

    def validate(self, record: object, project_dir: Pathish | None = None) -> None:
        return validate_session(record, self.project if project_dir is None else project_dir)

    def test_invalid_record_shape_or_unsupported_version_is_rejected(self) -> None:
        cases: list[object] = [None, [], "record", {}, dict(self.record, version=0),
                 dict(self.record, version=2), dict(self.record, version=True),
                 dict(self.record, version="1"), dict(self.record, version=1.0)]
        for record in cases:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    self.validate(record)


    def test_record_project_must_match_the_resolved_launch_directory(self) -> None:
        child = self.project / "child"
        child.mkdir()
        self.assertIsNone(self.validate(self.record, child / ".."))

        cases = [dict(self.record, project_dir=value) for value in (
            None, 42, str(self.project / "other"),
            self.record["project_dir"] + "/.", Path(self.record["project_dir"]),
        )]
        missing_project = dict(self.record)
        missing_project.pop("project_dir")
        cases.append(missing_project)
        for record in cases:
            with self.subTest(project_dir=record.get("project_dir")):
                with self.assertRaises(ValueError):
                    self.validate(record)


    def test_nested_runtime_objects_and_cycles_are_rejected_as_value_errors(self) -> None:
        from rich.text import Text

        values: list[object] = [self.project, Text("Not encoded"), {"a", "b"}, b"bytes",
                  object(), 1 + 2j, datetime.now(timezone.utc)]
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        values.append(cycle)
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                record = dict(self.record, messages=[{"metadata": {"nested": [value]}}])
                with self.assertRaises(ValueError):
                    self.validate(record)


class SessionSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.record = new_session(self.project, "Review the queue")

    def snapshot(self, messages: History, transcript: list[TranscriptEntry], active_turn: ActiveTurn | None) -> SessionRecord:
        return snapshot_session(self.record, messages, transcript, active_turn=active_turn)

    def test_snapshot_updates_conversation_and_timestamp_preserving_identity(self) -> None:
        from rich.text import Text
        from vin.sessions.records import validate_session
        from vin.sessions.transcript import decode_transcript

        messages: History = [{"type": "reasoning", "encrypted_content": "fake-reasoning"}]
        transcript: list[TranscriptEntry] = [
            {"role": "user", "content": "Review the queue"},
            {"role": "tool", "name": "read", "status": "running",
             "content": Text("Reading 界", style="dim")},
            {"role": "assistant", "content": "Checking", "streaming": True},
        ]
        active_turn: ActiveTurn = {"prompt": "Review the queue", "tools_used": True}
        original = deepcopy(self.record)
        before = datetime.now(timezone.utc)

        snapshot = self.snapshot(messages, transcript, active_turn)

        after = datetime.now(timezone.utc)
        self.assertEqual(self.record, original)
        for field in ("version", "id", "project_dir", "name", "created_at"):
            self.assertEqual(snapshot[field], original[field])
        updated = datetime.fromisoformat(snapshot["updated_at"])
        offset = updated.utcoffset()
        assert offset is not None
        self.assertEqual(offset.total_seconds(), 0)
        self.assertLessEqual(before, updated)
        self.assertLessEqual(updated, after)
        self.assertEqual(snapshot["messages"], messages)
        self.assertEqual(snapshot["active_turn"], active_turn)
        self.assertEqual(decode_transcript(snapshot["transcript"]), transcript)
        self.assertIsNone(validate_session(snapshot, self.project))
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)

    def test_snapshot_is_independent_and_omits_display_only_entries(self) -> None:
        from rich.text import Text

        messages: History = [{"metadata": {"nested": ["original"]}}]
        text = Text("Interrupted", style="yellow")
        transcript: list[TranscriptEntry] = [
            {"role": "notice", "content": Markdown("display only"), "persist": False},
            {"role": "notice", "content": "Thinking", "transient": True},
            {"role": "notice", "content": text},
        ]
        active: ActiveTurn = {"prompt": "Review the queue", "tools_used": False}
        snapshot = self.snapshot(messages, transcript, active)
        expected = deepcopy(snapshot)

        metadata = record_value(messages[0]["metadata"])
        nested = list_value(metadata["nested"])
        nested.append("later")
        text.append(" later")
        active["prompt"] = "Later prompt"
        self.record["name"] = "Later name"
        self.assertEqual(snapshot, expected)
        self.assertEqual(len(snapshot["transcript"]), 1)
        content = record_value(snapshot["transcript"][0]["content"])
        self.assertEqual(content["text"], "Interrupted")
        snapshot_metadata = record_value(snapshot["messages"][0]["metadata"])
        snapshot_nested = list_value(snapshot_metadata["nested"])
        snapshot_nested.append("snapshot only")
        self.assertEqual(nested, ["original", "later"])


class SavedSubagentRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.record = new_session(self.project, "Delegate it")

    def subagent(self) -> Subagent:
        from rich.text import Text

        return {"id": "call_task", "description": "Find sessions", "status": "finished", "started_at": 12.0,
                "duration": 4.5, "tokens": 100, "transcript": [
                    {"role": "user", "content": "Find where sessions are saved."},
                    {"role": "tool", "name": "read", "label": "read storage.py", "status": "finished",
                     "content": Text("● read storage.py · finished", style="dim")},
                ]}

    def test_snapshot_saves_subagents_without_their_clock_and_drops_the_key_when_empty(self) -> None:
        saved = snapshot_session(self.record, [], [], active_turn=None, subagents=[self.subagent()])

        item, = saved.get("subagents", [])
        self.assertEqual({key: value for key, value in item.items() if key != "transcript"},
                         {"id": "call_task", "description": "Find sessions", "status": "finished",
                          "duration": 4.5, "tokens": 100})
        self.assertEqual(item["transcript"][1]["content"], {"type": "rich_text", "text": "● read storage.py · finished",
                                                           "style": "dim", "spans": []})
        json.dumps(saved, allow_nan=False)

        emptied = snapshot_session(saved, [], [], active_turn=None)
        self.assertNotIn("subagents", emptied)
        self.assertNotIn("subagents", snapshot_session(self.record, [], [], active_turn=None))

    def test_sessions_without_subagents_stay_valid(self) -> None:
        validate_session(self.record, self.project)

    def test_invalid_saved_subagents_are_rejected(self) -> None:
        saved = snapshot_session(self.record, [], [], active_turn=None, subagents=[self.subagent()])
        good = record_value(cast(object, json.loads(json.dumps(saved.get("subagents", [])[0]))))
        cases: dict[str, object] = {
            "not a list": {"id": "x"},
            "not a record": ["x"],
            "missing id": [{key: value for key, value in good.items() if key != "id"}],
            "description not text": [dict(good, description=1)],
            "unknown status": [dict(good, status="paused")],
            "tokens are a bool": [dict(good, tokens=True)],
            "duration is text": [dict(good, duration="4s")],
            "broken transcript": [dict(good, transcript=[{"role": "robot", "content": "x"}])],
        }
        for label, subagents in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    validate_session(dict(deepcopy(self.record), subagents=subagents), self.project)
        validate_session(dict(deepcopy(self.record), subagents=[dict(good, duration=None), dict(good, duration=3)]),
                         self.project)


class MetadataToolEntry(ToolEntry):
    extra_metadata: dict[str, JSONValue]


class SessionRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.messages: History = [
            {"role": "user", "content": "Review the file"},
            {"type": "reasoning", "encrypted_content": "fake-reasoning"},
            {"type": "function_call", "name": "edit", "call_id": "call_saved", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_saved", "output": "Edited"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Finished", "annotations": []},
            ]},
        ]
        metadata_tool: MetadataToolEntry = {"role": "tool", "name": "edit", "label": "edit example.py",
             "status": "finished", "content": Text("Edited", style="dim"),
             "diff": "-old\n+new\n", "expanded": False,
             "extra_metadata": {"nested": ["original"]}}
        self.transcript: list[TranscriptEntry] = [
            {"role": "user", "content": "Review the file"},
            metadata_tool,
            {"role": "assistant", "content": "Finished", "streaming": False},
            {"role": "notice", "content": Text("Worked for 1s", style="dim")},
        ]
        snapshot = snapshot_session(new_session(self.project, "Review the file"),
                                    self.messages, self.transcript, active_turn=None)
        decoded = decoded_json(json.dumps(snapshot))
        validate_session(decoded, self.project)
        self.record = cast(SessionRecord, decoded)

    def restore(self, record: SessionRecord, project: Pathish | None = None) -> tuple[History, list[TranscriptEntry]]:
        return restore_session(record, self.project if project is None else project)

    def test_finished_session_restores_context_and_display_as_independent_data(self) -> None:
        original = deepcopy(self.record)

        messages, transcript = self.restore(self.record)

        self.assertEqual(messages, self.messages)
        self.assertEqual(encode_transcript(transcript), self.record["transcript"])
        self.assertIsInstance(transcript[1]["content"], Text)
        self.assertEqual(text_value(transcript[1]["content"]).style, "dim")
        self.assertEqual(self.record, original)
        record_value(list_value(messages[-1]["content"])[0])["text"] = "Changed in memory"
        text_value(transcript[1]["content"]).append(" changed")
        metadata = mapping_value(transcript[1]).get("extra_metadata")
        list_value(record_value(metadata)["nested"]).append("changed")
        self.assertEqual(self.record, original)


    def test_active_turn_normalizes_only_its_unfinished_display_entries(self) -> None:
        history: list[TranscriptEntry] = self.transcript + [
            {"role": "assistant", "content": "Earlier interrupted text", "streaming": True},
            {"role": "notice", "content": Text("Interrupted after 1s", style="yellow")},
        ]
        current: list[TranscriptEntry] = [
            {"role": "user", "content": "Continue review"},
            {"role": "assistant", "content": "Checking", "streaming": False, "intermediate": True},
            {"role": "tool", "name": "read", "status": "finished",
             "content": Text("Read earlier file", style="dim")},
            {"role": "tool", "name": "edit", "detail": "example.py", "label": "edit example.py",
             "status": "running", "content": Text("Editing", style="dim"),
             "diff": "-old\n+new\n"},
            {"role": "assistant", "content": "Partial current reply", "streaming": True},
        ]
        record = self.record.copy()
        record["transcript"] = encode_transcript(history + current)
        record["active_turn"] = {"prompt": "Continue review", "tools_used": True}
        original = deepcopy(record)

        messages, restored = self.restore(record)

        self.assertEqual(encode_transcript(restored[:len(history)]), record["transcript"][:len(history)])
        tail = restored[len(history):]
        self.assertEqual(tail[0], current[0])
        self.assertEqual(tail[1], current[1])
        self.assertEqual(tail[2], current[2])
        assert tail[3]["role"] == "tool"
        self.assertEqual(tail[3]["status"], "interrupted")
        self.assertEqual(tail[3].get("diff"), "-old\n+new\n")
        self.assertEqual(text_value(tail[3]["content"]).plain, "● edit example.py · interrupted")
        self.assertEqual(text_value(tail[3]["content"]).style, "yellow")
        self.assertEqual(tail[4]["content"], "Partial current reply")
        assert "streaming" in tail[4]
        self.assertFalse(tail[4]["streaming"])
        self.assertEqual(tail[5]["role"], "notice")
        self.assertIn("interrupted", text_value(tail[5]["content"]).plain.lower())
        self.assertEqual(messages[:len(self.messages)], self.messages)
        self.assertEqual(record, original)


    def interrupted_record(self, *, tools_used: bool) -> SessionRecord:
        entries: list[TranscriptEntry] = self.transcript + [{"role": "user", "content": "Continue review"}]
        if tools_used:
            entries.append({"role": "tool", "name": "edit", "status": "running",
                            "content": Text("Editing", style="dim"), "diff": "-old\n+new\n"})
        entries.append({"role": "assistant", "content": "Unfinished assistant fragment",
                        "streaming": True})
        record = self.record.copy()
        record["transcript"] = encode_transcript(entries)
        record["active_turn"] = {"prompt": "Continue review", "tools_used": tools_used}
        return record

    def test_tool_interruption_retains_pending_request_and_existing_caution_only(self) -> None:
        record = self.interrupted_record(tools_used=True)
        original = deepcopy(record)

        messages, transcript = self.restore(record)

        self.assertEqual(messages[:-2], self.messages)
        self.assertEqual(messages[-2], {"role": "user", "content": "Continue review"})
        self.assertEqual(messages[-1], {
            "role": "assistant",
            "content": ("Turn interrupted after tool use. "
                        "Changes may already exist; read relevant files before retrying."),
        })
        self.assertNotIn("Unfinished assistant fragment", json.dumps(messages))
        self.assertEqual(transcript[-2]["content"], "Unfinished assistant fragment")
        self.assertEqual(record, original)

    def test_text_only_interruption_keeps_previous_model_context_without_partial_reply(self) -> None:
        record = self.interrupted_record(tools_used=False)
        original = deepcopy(record)

        messages, transcript = self.restore(record)

        self.assertEqual(messages, self.messages)
        self.assertEqual(transcript[-2]["content"], "Unfinished assistant fragment")
        assert "streaming" in transcript[-2]
        self.assertFalse(transcript[-2]["streaming"])
        self.assertIn("interrupted", text_value(transcript[-1]["content"]).plain.lower())
        self.assertEqual(record, original)

    def test_repeated_restore_and_later_idle_checkpoint_do_not_duplicate_interruption(self) -> None:
        record = self.interrupted_record(tools_used=True)
        original = deepcopy(record)
        messages, transcript = self.restore(record)
        repeated_messages, repeated_transcript = self.restore(record)
        self.assertEqual(len(messages), len(self.messages) + 2)
        self.assertEqual(repeated_messages, messages)
        self.assertEqual(encode_transcript(repeated_transcript), encode_transcript(transcript))
        self.assertEqual(record, original)

        idle = snapshot_session(record, messages, transcript, active_turn=None)
        restored_messages, restored_transcript = self.restore(idle)
        self.assertEqual(restored_messages, messages)
        self.assertEqual(encode_transcript(restored_transcript), encode_transcript(transcript))


class SessionRenameTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.enterContext(patch.object(storage, "SESSIONS_DIR", self.root / "sessions"))
        self.owner = VinApp()
        self.owner.project_dir = self.project
        self.buffer = Buffer()
        self.buffer.text = "Unsent draft"

    def rename(self, name: object) -> str | None:
        return self.owner.rename_conversation(name)

    def test_invalid_name_preserves_current_conversation_and_saved_bytes(self) -> None:
        self.owner.messages = [{"role": "user", "content": "Original prompt"}]
        self.owner.transcript = [{"role": "user", "content": "Original prompt"}]
        record = snapshot_session(
            new_session(self.project, "Original prompt"), self.owner.messages,
            self.owner.transcript, active_turn=None,
        )
        path = storage.save_session(self.project, record)
        self.owner.session_record = deepcopy(record)
        self.owner.saved_session_record = deepcopy(record)
        previous = deepcopy((self.owner.session_record, self.owner.saved_session_record,
                             self.owner.messages, self.owner.transcript))
        previous_bytes = path.read_bytes()

        for name in (None, 42, "", " \t\n ", "x" * 81, "界" * 81):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.rename(name)
                self.assertEqual((self.owner.session_record, self.owner.saved_session_record,
                                  self.owner.messages, self.owner.transcript), previous)
                self.assertEqual(self.buffer.text, "Unsent draft")
                self.assertEqual(path.read_bytes(), previous_bytes)

    def prepare_saved_conversation(self) -> SessionRecord:
        self.owner.messages = [
            {"role": "user", "content": "Original prompt"},
            {"type": "reasoning", "encrypted_content": "fake-saved-reasoning"},
            {"role": "assistant", "content": "Original answer"},
        ]
        self.owner.transcript = [
            {"role": "user", "content": "Original prompt"},
            {"role": "assistant", "content": "Original answer"},
        ]
        record = snapshot_session(
            new_session(self.project, "Original prompt"), self.owner.messages,
            self.owner.transcript, active_turn=None,
        )
        storage.save_session(self.project, record)
        self.owner.session_record = deepcopy(record)
        self.owner.saved_session_record = deepcopy(record)
        self.owner.last_session_checkpoint = 10.0
        return record

    def test_rename_saves_normalized_name_immediately_without_changing_identity_or_history(self) -> None:
        record = self.prepare_saved_conversation()
        previous_messages = deepcopy(self.owner.messages)
        previous_transcript = deepcopy(self.owner.transcript)

        with patch.object(conversations, "monotonic", return_value=10.0):
            outcome = self.rename("  Fix\t cancellation\n behavior  ")

        self.assertEqual(outcome, "Renamed: Fix cancellation behavior")
        saved = present(storage.load_session(self.project, record["id"]))
        self.assertEqual(saved["name"], "Fix cancellation behavior")
        self.assertEqual(saved["id"], record["id"])
        self.assertEqual(saved["created_at"], record["created_at"])
        self.assertEqual(saved["project_dir"], record["project_dir"])
        self.assertEqual(saved["messages"], previous_messages)
        self.assertEqual(saved["transcript"], record["transcript"])
        self.assertEqual(self.owner.messages, previous_messages)
        self.assertEqual(self.owner.transcript, previous_transcript)
        self.assertEqual(self.buffer.text, "Unsent draft")
        self.assertEqual(self.owner.session_record, saved)
        self.assertEqual(self.owner.saved_session_record, saved)
        self.assertIsNot(self.owner.session_record, self.owner.saved_session_record)
        self.assertEqual([item["id"] for item in storage.list_sessions(self.project)], [record["id"]])
        self.assertEqual(self.rename("界" * 80), f"Renamed: {'界' * 80}")
        self.assertEqual(present(storage.load_session(self.project, record["id"]))["name"], "界" * 80)

    def test_failed_rename_keeps_previous_name_and_saved_bytes_then_can_retry(self) -> None:
        record = self.prepare_saved_conversation()
        path = storage.session_path(self.project, record["id"])
        previous_bytes = path.read_bytes()
        previous_transcript = deepcopy(self.owner.transcript)

        with patch.object(conversations, "monotonic", return_value=10.0):
            with patch.object(conversations, "save_session", side_effect=RuntimeError("Disk unavailable")):
                self.assertIsNone(self.rename("New title"))

            self.assertEqual(self.owner.session_record, record)
            self.assertEqual(self.owner.saved_session_record, record)
            self.assertEqual(path.read_bytes(), previous_bytes)
            self.assertEqual(self.owner.transcript[:-1], previous_transcript)
            notice = self.owner.transcript[-1]
            self.assertEqual(notice["role"], "notice")
            self.assertIn("Disk unavailable", str(notice["content"]))
            self.assertEqual(self.buffer.text, "Unsent draft")
            self.assertEqual(self.rename("New title"), "Renamed: New title")

        self.assertEqual(present(storage.load_session(self.project, record["id"]))["name"], "New title")
        self.assertIsNone(self.owner.session_save_error)


class SessionContentDecodingTests(unittest.TestCase):
    def decode(self, content: object) -> str | Text:
        return decode_content(content)

    def test_json_round_trip_restores_rich_content_and_its_rendered_appearance(self) -> None:
        from vin.sessions.transcript import encode_content

        styled = Text("● edit file.py\n界", style="dim")
        styled.stylize("bold cyan", 7, 14)
        styled.stylize("red", 9, 13)

        def render(content: RenderableType) -> str:
            output = StringIO()
            console = Console(file=output, width=60, force_terminal=True, color_system="truecolor")
            console.print(content)
            return output.getvalue()

        for source in (styled, Text("")):
            with self.subTest(text=source.plain):
                encoded: object = decoded_json(json.dumps(encode_content(source)))
                before = json.dumps(encoded, sort_keys=True)

                restored = self.decode(encoded)

                self.assertIsInstance(restored, Text)
                assert isinstance(restored, Text)
                self.assertEqual(restored.plain, source.plain)
                self.assertEqual(restored.style, source.style)
                self.assertEqual(restored.spans, source.spans)
                self.assertEqual(render(restored), render(source))
                self.assertEqual(json.dumps(encoded, sort_keys=True), before)


class SessionTranscriptDecodingTests(unittest.TestCase):
    def decode(self, entries: object) -> list[DecodedEntry]:
        return decode_transcript(entries)

    def test_whole_transcript_json_round_trip_restores_order_diffs_and_rendering(self) -> None:
        from prompt_toolkit.formatted_text import to_formatted_text
        from vin.sessions.transcript import encode_transcript
        from vin.tui.transcript import render_transcript

        source: list[TranscriptEntry] = [
            {"role": "user", "content": "Review\n[red]literal[/red]"},
            {"role": "assistant", "content": "Checking **the file**.", "intermediate": True},
            {"role": "tool", "name": "read", "detail": "same.py", "status": "finished",
             "content": Text("● read same.py · finished", style="dim"), "expanded": True},
            {"role": "tool", "name": "edit", "detail": "same.py", "status": "finished",
             "content": Text("● edit same.py · finished", style="dim"),
             "diff": "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"},
            {"role": "assistant", "content": "**Done.**", "streaming": False},
            {"role": "notice", "content": Text("Worked for 2s · done 09:30 PM", style="dim")},
        ]
        saved: object = decoded_json(json.dumps(encode_transcript(source)))
        original_saved = deepcopy(saved)

        restored = self.decode(saved)

        self.assertIsNot(restored, saved)
        self.assertEqual([entry["role"] for entry in restored],
                         ["user", "assistant", "tool", "tool", "assistant", "notice"])
        assert restored[3]["role"] == "tool"
        assert source[3]["role"] == "tool"
        self.assertEqual(restored[3].get("diff"), source[3].get("diff"))
        for width in (24, 90):
            with self.subTest(width=width):
                self.assertEqual(to_formatted_text(render_transcript(validated_transcript(restored), width, 100)),
                                 to_formatted_text(render_transcript(source, width, 100)))
        self.assertEqual(saved, original_saved)
        saved_items = list_value(saved)
        for entry, saved_entry in zip(restored, saved_items):
            self.assertIsNot(entry, saved_entry)
        empty: list[object] = []
        restored_empty = self.decode(empty)
        self.assertEqual(restored_empty, [])
        self.assertIsNot(restored_empty, empty)


if __name__ == "__main__":
    unittest.main()
