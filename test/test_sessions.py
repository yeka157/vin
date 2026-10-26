"""Project identity for independently stored conversation sessions."""

import importlib
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4


def probe_file_lock(path):
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
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def key(self, directory):
        try:
            storage = importlib.import_module("vin.sessions.storage")
        except ModuleNotFoundError as error:
            if error.name not in ("vin.sessions", "vin.sessions.storage"):
                raise
            self.fail("Create src/vin/sessions/storage.py with project_key(project_dir).")
        helper = getattr(storage, "project_key", None)
        self.assertTrue(callable(helper), "Add project_key(project_dir) in vin.sessions.storage.")
        key = helper(directory)
        self.assertIsInstance(key, str, "project_key must return a hexadecimal string.")
        self.assertRegex(key, r"\A[0-9a-f]{64}\Z")
        return key

    def test_same_named_projects_in_different_locations_have_distinct_keys(self):
        first = self.root / "repository" / "vin"
        second = self.root / "experiments" / "vin"
        first.mkdir(parents=True)
        second.mkdir(parents=True)

        first_key = self.key(first)
        second_key = self.key(second)

        self.assertNotEqual(first_key, second_key)

    def test_equivalent_path_forms_keep_the_same_project_identity(self):
        project = self.root / "project with spaces"
        child = project / "child"
        child.mkdir(parents=True)

        expected = self.key(project.resolve())

        self.assertEqual(self.key(str(project)), expected)
        self.assertEqual(self.key(child / ".."), expected)
        self.assertEqual(self.key(project), expected)


class SessionFilePathTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "session-data"
        self.project = self.root / "project"
        self.project.mkdir()
        self.storage = importlib.import_module("vin.sessions.storage")
        location = patch.object(self.storage, "SESSIONS_DIR", self.destination, create=True)
        location.start()
        self.addCleanup(location.stop)

    def path(self, project, session_id):
        helper = getattr(self.storage, "session_path", None)
        self.assertTrue(callable(helper), "Add session_path(project_dir, session_id) in storage.py.")
        return helper(project, session_id)


    def test_invalid_or_noncanonical_ids_cannot_become_session_paths(self):
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
    def setUp(self):
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

    def save(self, record, **kwargs):
        helper = getattr(self.storage, "save_session", None)
        self.assertTrue(callable(helper), "Add save_session(project_dir, record) in storage.py.")
        return helper(self.project, record, **kwargs)

    def test_save_writes_private_json_at_the_session_path_without_mutation(self):
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


    def test_failed_atomic_replacement_preserves_previous_file_and_cleans_temporary(self):
        path = self.save(self.record)
        previous = path.read_bytes()
        changed = dict(self.record, name="Renamed session")

        with patch("os.replace", side_effect=OSError("Replacement failed")):
            with self.assertRaises(RuntimeError) as error:
                self.save(changed, expected_record=self.record)

        self.assertIn("session", str(error.exception).lower())
        self.assertNotIn("sign-in", str(error.exception).lower())
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(set(path.parent.iterdir()), {path, path.with_suffix(".lock")})
        self.assertEqual(probe_file_lock(path.with_suffix(".lock")), "acquired")


class SessionLoadTests(unittest.TestCase):
    def setUp(self):
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

    def load(self, session_id):
        helper = getattr(self.storage, "load_session", None)
        self.assertTrue(callable(helper), "Add load_session(project_dir, session_id) in storage.py.")
        return helper(self.project, session_id)


    def test_corrupt_invalid_or_mismatched_records_raise_session_errors(self):
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

    def test_nonprivate_or_foreign_owned_session_is_rejected(self):
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
    def setUp(self):
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

    def listing(self):
        helper = getattr(self.storage, "list_sessions", None)
        self.assertTrue(callable(helper), "Add list_sessions(project_dir) in storage.py.")
        return helper(self.project)

    def record(self, project=None):
        from vin.sessions.records import new_session

        return new_session(self.project if project is None else project, "Same name")


    def test_only_current_project_sessions_are_listed_newest_first_by_instant(self):
        records = []
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
    def setUp(self):
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

    def lock(self, session_id=None):
        helper = getattr(self.storage, "session_lock", None)
        self.assertTrue(callable(helper), "Add session_lock(project_dir, session_id) in storage.py.")
        return helper(self.project, self.session_id if session_id is None else session_id)

    def probe(self, path):
        return probe_file_lock(path)

    def test_private_lock_excludes_other_process_until_context_exits(self):
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
    def setUp(self):
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

    def test_update_requires_exact_previous_record_and_preserves_inputs(self):
        path = self.storage.save_session(self.project, self.record)
        baseline = deepcopy(self.record)
        updated = dict(self.record, name="Renamed session")
        original_updated = deepcopy(updated)

        result = self.storage.save_session(self.project, updated, expected_record=baseline)

        self.assertEqual(result, path)
        self.assertEqual(self.storage.load_session(self.project, self.record["id"]), updated)
        self.assertEqual(baseline, self.record)
        self.assertEqual(updated, original_updated)
        previous = path.read_bytes()
        for kwargs in ({}, {"expected_record": baseline}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RuntimeError) as error:
                    self.storage.save_session(self.project, dict(updated, name="Stale"), **kwargs)
                self.assertIn("changed", str(error.exception).lower())
                self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(probe_file_lock(path.with_suffix(".lock")), "acquired")

    def test_another_process_update_cannot_be_overwritten_by_stale_parent(self):
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
            self.storage.save_session(self.project, dict(self.record, name="Stale parent"),
                                      expected_record=self.record)
        self.assertIn("changed", str(error.exception).lower())
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(json.loads(previous)["name"], "Saved by another process")


    def test_expected_existing_record_cannot_recreate_missing_or_overwrite_corrupt_file(self):
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
