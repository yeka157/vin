"""Initial records for named, resumable conversations."""

from datetime import datetime, timezone
from copy import deepcopy
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID


class NewSessionRecordTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "project"
        self.project.mkdir()

    def create(self, first_message):
        try:
            records = importlib.import_module("vin.sessions.records")
        except ModuleNotFoundError as error:
            if error.name != "vin.sessions.records":
                raise
            self.fail("Create src/vin/sessions/records.py with new_session(project_dir, first_message).")
        factory = getattr(records, "new_session", None)
        self.assertTrue(callable(factory), "Add new_session(project_dir, first_message).")
        return factory(self.project, first_message)

    def test_new_record_has_project_metadata_and_empty_conversation_data(self):
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
        self.assertIsNotNone(created.utcoffset())
        self.assertEqual(created.utcoffset().total_seconds(), 0)
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, after)
        self.assertEqual(record["updated_at"], record["created_at"])
        self.assertEqual(record["messages"], [])
        self.assertEqual(record["transcript"], [])
        self.assertIsNone(record["active_turn"])
        self.assertEqual(json.loads(json.dumps(record)), record)


    def test_auto_name_collapses_whitespace_and_limits_long_prompts(self):
        record = self.create("  Review\n\t the   queue  ")
        self.assertEqual(record["name"], "Review the queue")
        long_prompt = "Check " + "界" * 100
        record = self.create(long_prompt)
        self.assertEqual(record["name"], long_prompt[:80])


class SessionRecordValidationTests(unittest.TestCase):
    def setUp(self):
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.record = new_session(self.project, "Review the queue")

    def validate(self, record, project_dir=None):
        from vin.sessions import records

        validator = getattr(records, "validate_session", None)
        self.assertTrue(callable(validator), "Add validate_session(record, project_dir) in sessions/records.py.")
        return validator(record, self.project if project_dir is None else project_dir)


    def test_invalid_record_shape_or_unsupported_version_is_rejected(self):
        cases = [None, [], "record", {}, dict(self.record, version=0),
                 dict(self.record, version=2), dict(self.record, version=True),
                 dict(self.record, version="1"), dict(self.record, version=1.0)]
        for record in cases:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    self.validate(record)


    def test_record_project_must_match_the_resolved_launch_directory(self):
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


    def test_nested_runtime_objects_and_cycles_are_rejected_as_value_errors(self):
        from rich.text import Text

        values = [self.project, Text("Not encoded"), {"a", "b"}, b"bytes",
                  object(), 1 + 2j, datetime.now(timezone.utc)]
        cycle = {}
        cycle["self"] = cycle
        values.append(cycle)
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                record = dict(self.record, messages=[{"metadata": {"nested": [value]}}])
                with self.assertRaises(ValueError):
                    self.validate(record)


class SessionSnapshotTests(unittest.TestCase):
    def setUp(self):
        from vin.sessions.records import new_session

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.record = new_session(self.project, "Review the queue")

    def snapshot(self, messages, transcript, active_turn):
        from vin.sessions import records

        helper = getattr(records, "snapshot_session", None)
        self.assertTrue(callable(helper),
                        "Add snapshot_session(record, messages, transcript, *, active_turn) in records.py.")
        return helper(self.record, messages, transcript, active_turn=active_turn)

    def test_snapshot_updates_conversation_and_timestamp_preserving_identity(self):
        from rich.text import Text
        from vin.sessions.records import validate_session
        from vin.sessions.transcript import decode_transcript

        messages = [{"type": "reasoning", "encrypted_content": "fake-reasoning"}]
        transcript = [
            {"role": "user", "content": "Review the queue"},
            {"role": "tool", "name": "read", "status": "running",
             "content": Text("Reading 界", style="dim")},
            {"role": "assistant", "content": "Checking", "streaming": True},
        ]
        active_turn = {"prompt": "Review the queue", "tools_used": True}
        original = deepcopy(self.record)
        before = datetime.now(timezone.utc)

        snapshot = self.snapshot(messages, transcript, active_turn)

        after = datetime.now(timezone.utc)
        self.assertEqual(self.record, original)
        for field in ("version", "id", "project_dir", "name", "created_at"):
            self.assertEqual(snapshot[field], original[field])
        updated = datetime.fromisoformat(snapshot["updated_at"])
        self.assertEqual(updated.utcoffset().total_seconds(), 0)
        self.assertLessEqual(before, updated)
        self.assertLessEqual(updated, after)
        self.assertEqual(snapshot["messages"], messages)
        self.assertEqual(snapshot["active_turn"], active_turn)
        self.assertEqual(decode_transcript(snapshot["transcript"]), transcript)
        self.assertIsNone(validate_session(snapshot, self.project))
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)

    def test_snapshot_is_independent_and_omits_display_only_entries(self):
        from rich.text import Text

        messages = [{"metadata": {"nested": ["original"]}}]
        text = Text("Interrupted", style="yellow")
        transcript = [
            {"role": "notice", "content": object(), "persist": False},
            {"role": "notice", "content": "Thinking", "transient": True},
            {"role": "notice", "content": text},
        ]
        active = {"prompt": "Review the queue", "tools_used": False}
        snapshot = self.snapshot(messages, transcript, active)
        expected = deepcopy(snapshot)

        messages[0]["metadata"]["nested"].append("later")
        text.append(" later")
        active["prompt"] = "Later prompt"
        self.record["name"] = "Later name"
        self.assertEqual(snapshot, expected)
        self.assertEqual(len(snapshot["transcript"]), 1)
        self.assertEqual(snapshot["transcript"][0]["content"]["text"], "Interrupted")
        snapshot["messages"][0]["metadata"]["nested"].append("snapshot only")
        self.assertEqual(messages[0]["metadata"]["nested"], ["original", "later"])
