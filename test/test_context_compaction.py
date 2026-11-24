"""Vin-side compaction in the provider-neutral agent loop."""

from copy import deepcopy
from threading import Event
import unittest
from unittest.mock import patch

from vin import agent
from vin.agent import AgentError, SUMMARY_PREFIX
from vin.provider.events import EventSender
from vin.types import History, ProviderItem

type Reply = tuple[History, bool] | None


def answer(text: str) -> History:
    return [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]


def summary(text: str = "Goal, decisions and remaining work.") -> ProviderItem:
    return {"role": "user", "content": SUMMARY_PREFIX + text}


CALL: ProviderItem = {"type": "function_call", "call_id": "c1", "name": "read",
                      "arguments": '{"path":"f"}', "status": "completed"}


class CompactionLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancel = Event()
        self.calls: list[tuple[History, bool, bool]] = []
        self.sent: list[tuple[str, object]] = []
        self.history: History = [
            {"role": "user", "content": "Old request " + "x" * 3000},
            *answer("Old answer"),
            {"type": "reasoning", "encrypted_content": "opaque"},
            {"role": "user", "content": "Continue the task"},
        ]

    def send(self, kind: str, value: object) -> None:
        self.sent.append((kind, value))

    def run_turn(self, replies: list[Reply], *, threshold: int | None = 500, compact_only: bool = False,
                 history: History | None = None) -> History | None:
        queue = iter(replies)

        def request(history: History, sender: EventSender, subagent: bool, compact: bool) -> Reply:
            self.calls.append((deepcopy(history), subagent, compact))
            if compact:
                self.assertIs(sender, agent.quiet)
            return next(queue)

        inputs = self.history if history is None else history
        before = deepcopy(inputs)
        result = agent.run_agent_turn(request, inputs, self.send, self.cancel,
                                      compact_threshold=threshold, compact_only=compact_only)
        self.assertEqual(inputs, before)
        return result

    def test_over_the_threshold_compacts_first_and_keeps_the_current_request(self) -> None:
        result = self.run_turn([(answer("Summary text"), True), (answer("Done"), True)])
        self.assertEqual([compact for _, _, compact in self.calls], [True, False])
        self.assertEqual(self.calls[0][0], self.history)
        self.assertEqual(self.calls[1][0], [summary("Summary text"), {"role": "user", "content": "Continue the task"}])
        self.assertEqual(result, [summary("Summary text"), {"role": "user", "content": "Continue the task"}, *answer("Done")])
        self.assertTrue(agent.is_summary(summary()))

    def test_under_the_threshold_or_unknown_limit_never_compacts(self) -> None:
        for threshold in (10**9, None):
            with self.subTest(threshold=threshold):
                self.calls.clear()
                self.run_turn([(answer("Done"), True)], threshold=threshold)
                self.assertEqual([compact for _, _, compact in self.calls], [False])

    def test_growth_between_tool_rounds_triggers_compaction_mid_turn(self) -> None:
        history: History = [{"role": "user", "content": "Read the big file"}]
        big_output = "y" * 3000

        def execute(*_args: object, **_kwargs: object) -> str:
            return big_output
        with patch.object(agent, "execute_tool", execute):
            result = self.run_turn([([CALL], False), (answer("Summary"), True), (answer("Done"), True)],
                                   history=history)
        self.assertEqual([compact for _, _, compact in self.calls], [False, True, False])
        self.assertEqual(self.calls[1][0][-1], {"type": "function_call_output", "call_id": "c1", "output": big_output})
        self.assertEqual(self.calls[2][0], [summary("Summary"), {"role": "user", "content": "Read the big file"}])
        assert result is not None
        self.assertEqual(result[-1], answer("Done")[0])

    def test_a_compacted_history_is_not_compacted_again_in_a_loop(self) -> None:
        history: History = [summary("x" * 3000), {"role": "user", "content": "Continue"}]
        self.run_turn([(answer("Done"), True)], history=history)
        self.assertEqual([compact for _, _, compact in self.calls], [False])

    def test_manual_compaction_keeps_only_the_summary(self) -> None:
        result = self.run_turn([(answer("Summary text"), True)], compact_only=True, threshold=None)
        self.assertEqual(result, [summary("Summary text")])
        self.assertEqual([compact for _, _, compact in self.calls], [True])

    def test_summary_text_and_usage_never_reach_the_screen(self) -> None:
        self.run_turn([(answer("Summary text"), True)], compact_only=True)
        self.assertEqual(self.sent, [])

    def test_failures_raise_and_never_replace_the_history(self) -> None:
        cases: list[tuple[History, str]] = [
            ([], "Compaction returned no summary. Please try again."),
            (answer("   "), "Compaction returned no summary. Please try again."),
            ([CALL, *answer("Summary")], "Compaction returned unexpected tool calls."),
        ]
        for output, message in cases:
            for compact_only in (True, False):
                with self.subTest(message=message, compact_only=compact_only):
                    self.calls.clear()
                    with self.assertRaises(AgentError) as raised:
                        self.run_turn([(output, False), (answer("Done"), True)], compact_only=compact_only)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(len(self.calls), 1)

    def test_cancelled_compaction_returns_nothing(self) -> None:
        for compact_only in (True, False):
            with self.subTest(compact_only=compact_only):
                self.calls.clear()
                self.assertIsNone(self.run_turn([None], compact_only=compact_only))
                self.assertEqual(len(self.calls), 1)

    def test_cancel_before_compaction_sends_nothing(self) -> None:
        self.cancel.set()
        self.assertIsNone(self.run_turn([], compact_only=True))
        self.assertEqual(self.calls, [])

    def test_subagent_turns_tell_the_request_they_are_subagents(self) -> None:
        def request(history: History, sender: EventSender, subagent: bool, compact: bool) -> Reply:
            self.calls.append((history, subagent, compact))
            return (answer("Summary"), True) if compact else (answer("Done"), True)
        agent.run_agent_turn(request, self.history, self.send, self.cancel, subagent=True, compact_threshold=500)
        self.assertEqual([(subagent, compact) for _, subagent, compact in self.calls], [(True, True), (True, False)])

    def test_estimate_grows_with_the_history(self) -> None:
        small = agent.estimate_tokens([{"role": "user", "content": "hi"}])
        large = agent.estimate_tokens([{"role": "user", "content": "hi" * 3000}])
        self.assertGreater(large, small + 1500)


if __name__ == "__main__":
    unittest.main()
