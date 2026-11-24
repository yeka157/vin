"""The agent loop runs against a fake request function, with no OpenAI client."""

from collections.abc import Callable
from threading import Event
import unittest
from unittest.mock import patch

from vin import agent
from vin.provider.events import EventSender, TextEventKind, ToolEventKind, ToolStarted, TokenUsage
from vin.types import History


class ProviderNeutralLoopTests(unittest.TestCase):
    def test_fake_request_drives_one_tool_call_to_a_final_answer(self) -> None:
        user: History = [{"role": "user", "content": "Read f."}]
        call: History = [{"type": "function_call", "call_id": "call_1", "name": "read",
                          "arguments": '{"path":"f"}', "status": "completed"}]
        answer: History = [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "Done"}]}]
        seen: list[History] = []
        replies = iter([(call, False), (answer, True)])

        def request(history: History, _sender: EventSender, _subagent: bool, _compact: bool) -> tuple[History, bool] | None:
            seen.append(list(history))
            return next(replies)

        sent: list[tuple[str, object]] = []

        def send(kind: TextEventKind | ToolEventKind | str, value: str | ToolStarted | TokenUsage | History | None) -> None:
            sent.append((kind, value))

        with patch.object(agent, "execute_tool", return_value="1: hello") as execute:
            history = agent.run_agent_turn(request, user, send, Event())

        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[:2], ("read", '{"path":"f"}'))
        output: History = [{"type": "function_call_output", "call_id": "call_1", "output": "1: hello"}]
        self.assertEqual(seen, [user, user + call + output])
        self.assertEqual(history, user + call + output + answer)
        self.assertEqual(sent, [("tool_started", {"name": "read", "detail": "f"})])
        self.assertEqual(user, [{"role": "user", "content": "Read f."}])


TASK_ARGUMENTS = '{"description":"Find sessions","prompt":"Find where sessions are saved."}'


def task_call(arguments: str = TASK_ARGUMENTS) -> History:
    return [{"type": "function_call", "call_id": "call_task", "name": "task",
             "arguments": arguments, "status": "completed"}]


def answer(text: str) -> History:
    return [{"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": text}]}]


READ_CALL: History = [{"type": "function_call", "call_id": "call_read", "name": "read",
                       "arguments": '{"path":"f"}', "status": "completed"}]


STARTED: ToolStarted = {"name": "task", "detail": "Find sessions", "task_id": "call_task",
                        "prompt": "Find where sessions are saved."}


class TaskToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancel = Event()
        self.parent_seen: list[History] = []
        self.child_seen: list[History] = []
        self.sent: list[tuple[str, object]] = []

    def send(self, kind: TextEventKind | ToolEventKind | str, value: str | ToolStarted | TokenUsage | History | None) -> None:
        self.sent.append((kind, value))

    def run_turn(
        self,
        parent: list[tuple[History, bool]],
        child: Callable[[History, EventSender], tuple[History, bool] | None],
        subagent_budget: int | None = None,
        subagent_sender: agent.SubagentSender | None = None,
    ) -> History | None:
        replies = iter(parent)

        def request(history: History, sender: EventSender, subagent: bool, _compact: bool) -> tuple[History, bool] | None:
            if subagent:
                self.child_seen.append(list(history))
                return child(history, sender)
            self.parent_seen.append(list(history))
            return next(replies)

        user: History = [{"role": "user", "content": "Delegate it."}]
        if subagent_budget is None:
            return agent.run_agent_turn(request, user, self.send, self.cancel, subagent_sender=subagent_sender)
        return agent.run_agent_turn(request, user, self.send, self.cancel, subagent_budget=subagent_budget,
                                    subagent_sender=subagent_sender)

    def task_output(self) -> object:
        return self.parent_seen[-1][-1].get("output")

    def test_subagent_starts_fresh_and_its_final_answer_is_the_only_result(self) -> None:
        def child(_history: History, sender: EventSender) -> tuple[History, bool]:
            sender("text", "subagent text")
            return answer("Found it in storage.py"), True

        history = self.run_turn([(task_call(), False), (answer("Done"), True)], child)

        self.assertEqual(self.child_seen, [[{"role": "user", "content": "Find where sessions are saved."}]])
        self.assertEqual(self.task_output(), "Found it in storage.py")
        assert history is not None
        self.assertEqual([item.get("type") for item in history],
                         [None, "function_call", "function_call_output", "message"])
        self.assertEqual(self.sent, [
            ("tool_started", STARTED),
            ("tool_finished", "task"),
        ])

    def test_subagent_events_go_to_the_sender_made_for_its_task(self) -> None:
        tagged: list[tuple[str, str, object]] = []

        def subagent_sender(task_id: str) -> EventSender:
            def child_send(kind: str, value: object) -> None:
                tagged.append((task_id, kind, value))
            return child_send

        def child(_history: History, sender: EventSender) -> tuple[History, bool]:
            sender("text", "Found it")
            return answer("Found it"), True

        self.run_turn([(task_call(), False), (answer("Done"), True)], child, subagent_sender=subagent_sender)

        self.assertEqual(tagged, [("call_task", "text", "Found it")])
        self.assertEqual(self.sent, [("tool_started", STARTED), ("tool_finished", "task")])

    def test_request_budget_defaults_to_eight_and_failure_lets_the_parent_continue(self) -> None:
        for budget, expected in ((None, 8), (2, 2)):
            with self.subTest(budget=budget):
                self.setUp()
                with patch.object(agent, "execute_tool", return_value="1: hello"):
                    history = self.run_turn([(task_call(), False), (answer("Done"), True)],
                                            lambda _history, _sender: (READ_CALL, False), budget)

                self.assertEqual(len(self.child_seen), expected)
                self.assertEqual(self.task_output(),
                                 f"Error: subagent failed: Model request limit exceeded: maximum {expected}.")
                self.assertEqual(self.sent[-1], ("tool_failed", "task"))
                assert history is not None
                self.assertEqual(history[-1], answer("Done")[0])

    def test_invalid_arguments_fail_without_starting_a_subagent(self) -> None:
        for arguments in ("not json", "[]", '{"description":"x"}', '{"prompt":"p"}',
                          '{"description":"x","prompt":"   "}', '{"description":1,"prompt":"p"}'):
            with self.subTest(arguments=arguments):
                self.setUp()
                self.run_turn([(task_call(arguments), False), (answer("Done"), True)],
                              lambda _history, _sender: self.fail("Subagent must not start."))

                self.assertEqual(self.child_seen, [])
                self.assertEqual(self.task_output(), "Error: task needs a description and a prompt.")
                self.assertEqual([kind for kind, _ in self.sent], ["tool_started", "tool_failed"])
                self.assertEqual(self.sent[0], ("tool_started", {"name": "task", "detail": agent.tool_detail("task", arguments)}))

    def test_a_subagent_cannot_start_another_subagent(self) -> None:
        replies = iter([(task_call(), False), (answer("Could not delegate."), True)])

        self.run_turn([(task_call(), False), (answer("Done"), True)],
                      lambda _history, _sender: next(replies))

        self.assertEqual(len(self.child_seen), 2)
        self.assertEqual(self.child_seen[1][-1].get("output"), "Error (ValueError): Tool 'task' not registered.")
        self.assertEqual(self.task_output(), "Could not delegate.")

    def test_task_is_not_run_when_subagents_are_off(self) -> None:
        replies = iter([(task_call(), False), (answer("Done"), True)])

        def request(history: History, _sender: EventSender, subagent: bool, _compact: bool) -> tuple[History, bool] | None:
            self.assertFalse(subagent)
            self.parent_seen.append(list(history))
            return next(replies)

        agent.run_agent_turn(request, [{"role": "user", "content": "Delegate it."}], self.send, self.cancel,
                             subagents=False)

        self.assertEqual(self.task_output(), "Error (ValueError): Tool 'task' not registered.")

    def test_cancelling_inside_a_subagent_ends_the_parent_turn(self) -> None:
        def child(_history: History, _sender: EventSender) -> None:
            self.cancel.set()

        history = self.run_turn([(task_call(), False)], child)

        self.assertIsNone(history)
        self.assertEqual(len(self.parent_seen), 1)
        self.assertEqual(self.sent, [("tool_started", STARTED)])

    def test_provider_errors_inside_a_subagent_end_the_parent_turn(self) -> None:
        def child(_history: History, _sender: EventSender) -> None:
            raise RuntimeError("rate limited")

        with self.assertRaisesRegex(RuntimeError, "rate limited"):
            self.run_turn([(task_call(), False)], child)


if __name__ == "__main__":
    unittest.main()
