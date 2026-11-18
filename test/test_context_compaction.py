"""Retain native compaction checkpoints and all subsequent provider items."""

from copy import deepcopy
import unittest

from vin import agent
from vin.types import History, ProviderItem


class CompactedContextTests(unittest.TestCase):
    def test_context_without_native_checkpoint_is_preserved(self) -> None:
        fixtures: list[History] = [
            [],
            [
                {"role": "user", "content": "Keep the current goal"},
                {"type": "reasoning", "encrypted_content": "fake-reasoning"},
                {"type": "function_call", "call_id": "read-1", "name": "read", "arguments": '{"path":"README.md"}'},
                {"type": "function_call_output", "call_id": "read-1", "output": "Saved tool result"},
                {"type": "future_provider_item", "metadata": {"keep": True}},
            ],
        ]
        for history in fixtures:
            with self.subTest(empty=not history):
                before = deepcopy(history)
                result = agent.context_after_compaction(history)
                self.assertEqual(result, before)
                self.assertEqual(history, before)
                self.assertIsNot(result, history)

    def test_latest_checkpoint_keeps_opaque_fields_and_later_tool_exchange(self) -> None:
        checkpoint: ProviderItem = {
            "id": "cmp_latest", "type": "compaction", "encrypted_content": "fake-opaque-context",
            "created_by": "system", "future_metadata": {"keep": [1, True, None]},
        }
        call: ProviderItem = {
            "type": "function_call", "call_id": "read-2", "name": "read", "arguments": '{"path":"app.py"}',
        }
        output: ProviderItem = {"type": "function_call_output", "call_id": "read-2", "output": "Relevant current code"}
        reply: ProviderItem = {"role": "assistant", "content": "Next, check the failure path."}
        history: History = [
            {"role": "user", "content": "Old request"},
            {"id": "cmp_old", "type": "compaction", "encrypted_content": "fake-old-context"},
            {"role": "assistant", "content": "Superseded explanation"},
            checkpoint, call, output, reply,
        ]
        retained: History = [checkpoint, call, output, reply]
        cases: list[tuple[History, History]] = [
            (history, retained),
            (retained.copy(), retained),
            ([checkpoint], [checkpoint]),
        ]
        for original, expected in cases:
            with self.subTest(item_count=len(original)):
                before = deepcopy(original)
                result = agent.context_after_compaction(original)
                self.assertEqual(result, expected)
                self.assertEqual(original, before)
                self.assertIsNot(result, original)


if __name__ == "__main__":
    unittest.main()
