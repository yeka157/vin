"""The agent loop runs against a fake request function, with no OpenAI client."""

from threading import Event
import unittest
from unittest.mock import patch

from vin import agent
from vin.provider.events import TextEventKind, ToolEventKind, ToolStarted, TokenUsage
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

        def request(history: History) -> tuple[History, bool] | None:
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


if __name__ == "__main__":
    unittest.main()
