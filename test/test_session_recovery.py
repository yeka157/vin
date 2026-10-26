"""Restore saved conversation data without changing the saved checkpoint."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.text import Text

from vin.sessions.records import new_session, snapshot_session
from vin.sessions.transcript import encode_transcript


class SessionRestoreTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.messages = [
            {"role": "user", "content": "Review the file"},
            {"type": "reasoning", "encrypted_content": "fake-reasoning"},
            {"type": "function_call", "name": "edit", "call_id": "call_saved", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_saved", "output": "Edited"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Finished", "annotations": []},
            ]},
        ]
        self.transcript = [
            {"role": "user", "content": "Review the file"},
            {"role": "tool", "name": "edit", "label": "edit example.py",
             "status": "finished", "content": Text("Edited", style="dim"),
             "diff": "-old\n+new\n", "expanded": False,
             "extra_metadata": {"nested": ["original"]}},
            {"role": "assistant", "content": "Finished", "streaming": False},
            {"role": "notice", "content": Text("Worked for 1s", style="dim")},
        ]
        snapshot = snapshot_session(new_session(self.project, "Review the file"),
                                    self.messages, self.transcript, active_turn=None)
        self.record = json.loads(json.dumps(snapshot))

    def restore(self, record, project=None):
        try:
            recovery = importlib.import_module("vin.sessions.recovery")
        except ModuleNotFoundError as error:
            if error.name != "vin.sessions.recovery":
                raise
            self.fail("Create sessions/recovery.py with restore_session(record, project_dir).")
        helper = getattr(recovery, "restore_session", None)
        self.assertTrue(callable(helper), "Add restore_session(record, project_dir).")
        return helper(record, self.project if project is None else project)

    def test_finished_session_restores_context_and_display_as_independent_data(self):
        original = deepcopy(self.record)

        messages, transcript = self.restore(self.record)

        self.assertEqual(messages, self.messages)
        self.assertEqual(encode_transcript(transcript), self.record["transcript"])
        self.assertIsInstance(transcript[1]["content"], Text)
        self.assertEqual(transcript[1]["content"].style, "dim")
        self.assertEqual(self.record, original)
        messages[-1]["content"][0]["text"] = "Changed in memory"
        transcript[1]["content"].append(" changed")
        transcript[1]["extra_metadata"]["nested"].append("changed")
        self.assertEqual(self.record, original)


    def test_active_turn_normalizes_only_its_unfinished_display_entries(self):
        history = self.transcript + [
            {"role": "assistant", "content": "Earlier interrupted text", "streaming": True},
            {"role": "notice", "content": Text("Interrupted after 1s", style="yellow")},
        ]
        current = [
            {"role": "user", "content": "Continue review"},
            {"role": "assistant", "content": "Checking", "streaming": False, "intermediate": True},
            {"role": "tool", "name": "read", "status": "finished",
             "content": Text("Read earlier file", style="dim")},
            {"role": "tool", "name": "edit", "detail": "example.py", "label": "edit example.py",
             "status": "running", "content": Text("Editing", style="dim"),
             "diff": "-old\n+new\n"},
            {"role": "assistant", "content": "Partial current reply", "streaming": True},
        ]
        record = dict(self.record, transcript=encode_transcript(history + current),
                      active_turn={"prompt": "Continue review", "tools_used": True})
        original = deepcopy(record)

        messages, restored = self.restore(record)

        self.assertEqual(encode_transcript(restored[:len(history)]), record["transcript"][:len(history)])
        tail = restored[len(history):]
        self.assertEqual(tail[0], current[0])
        self.assertEqual(tail[1], current[1])
        self.assertEqual(tail[2], current[2])
        self.assertEqual(tail[3]["status"], "interrupted")
        self.assertEqual(tail[3]["diff"], "-old\n+new\n")
        self.assertEqual(tail[3]["content"].plain, "● edit example.py · interrupted")
        self.assertEqual(tail[3]["content"].style, "yellow")
        self.assertEqual(tail[4]["content"], "Partial current reply")
        self.assertFalse(tail[4]["streaming"])
        self.assertEqual(tail[5]["role"], "notice")
        self.assertIn("interrupted", tail[5]["content"].plain.lower())
        self.assertEqual(messages[:len(self.messages)], self.messages)
        self.assertEqual(record, original)


    def interrupted_record(self, *, tools_used):
        entries = self.transcript + [{"role": "user", "content": "Continue review"}]
        if tools_used:
            entries.append({"role": "tool", "name": "edit", "status": "running",
                            "content": Text("Editing", style="dim"), "diff": "-old\n+new\n"})
        entries.append({"role": "assistant", "content": "Unfinished assistant fragment",
                        "streaming": True})
        return dict(self.record, transcript=encode_transcript(entries),
                    active_turn={"prompt": "Continue review", "tools_used": tools_used})

    def test_tool_interruption_retains_pending_request_and_existing_caution_only(self):
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

    def test_text_only_interruption_keeps_previous_model_context_without_partial_reply(self):
        record = self.interrupted_record(tools_used=False)
        original = deepcopy(record)

        messages, transcript = self.restore(record)

        self.assertEqual(messages, self.messages)
        self.assertEqual(transcript[-2]["content"], "Unfinished assistant fragment")
        self.assertFalse(transcript[-2]["streaming"])
        self.assertIn("interrupted", transcript[-1]["content"].plain.lower())
        self.assertEqual(record, original)

    def test_repeated_restore_and_later_idle_checkpoint_do_not_duplicate_interruption(self):
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
