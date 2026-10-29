"""Initial records for named, resumable conversations."""

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from vin.sessions.records import new_session, validate_session, snapshot_session
from vin.types import ActiveTurn, History, Pathish, SessionRecord, TranscriptEntry
from session_typing_helpers import list_value, record_value
from rich.markdown import Markdown


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
