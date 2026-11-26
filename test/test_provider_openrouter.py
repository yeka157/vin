"""Offline OpenRouter Chat Completions contract tests.

The real openai SDK parses mocked HTTP streams, so SDK-level behavior such as
raising on a mid-stream error chunk is exercised too. No key, account or network.
"""

from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from threading import Event
from typing import TYPE_CHECKING, cast
import unittest
from unittest.mock import patch

from aisuite.utils.tools import Tools
import httpx2
import openai

from vin import agent

from vin.provider import prompts
from vin.provider.events import ReplyEvent, ReplyEvents
from vin.provider.openrouter import stream_openrouter_response, to_chat_messages
from vin.types import History, JSONValue, json_value

if TYPE_CHECKING:
    from aisuite.utils.tools import ToolCall

KEY = "sk-or-test-secret-never-print"
type Chunk = dict[str, JSONValue]
type Fixture = list[Chunk] | tuple[int, JSONValue] | Exception


def chunk(content: str | None = None, *, finish: str | None = None,
          tool_calls: list[JSONValue] | None = None) -> Chunk:
    delta: dict[str, JSONValue] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"id": "gen-1", "object": "chat.completion.chunk", "created": 0, "model": "test/model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def usage(prompt: int = 10, completion: int = 5) -> Chunk:
    return {"id": "gen-1", "object": "chat.completion.chunk", "created": 0, "model": "test/model",
            "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                     "total_tokens": prompt + completion}}


def tool_delta(index: int, *, call_id: str | None = None, name: str | None = None,
               arguments: str | None = None) -> JSONValue:
    function: dict[str, JSONValue] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    entry: dict[str, JSONValue] = {"index": index, "function": function}
    if call_id is not None:
        entry["id"] = call_id
        entry["type"] = "function"
    return entry


def sse(chunks: Sequence[Chunk]) -> bytes:
    return ("".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n").encode()


def called(entry: object) -> tuple[str, str]:
    function = cast(dict[str, dict[str, str]], entry)["function"]
    return function["name"], function["arguments"]


def record(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def records(value: JSONValue) -> list[dict[str, JSONValue]]:
    assert isinstance(value, list)
    return [record(item) for item in value]


class OpenRouterStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancel = Event()
        self.events = ReplyEvents()
        self.bodies: list[dict[str, JSONValue]] = []
        self.headers: list[httpx2.Headers] = []
        self.urls: list[str] = []
        self.clients: list[httpx2.Client] = []
        self.constructed: list[dict[str, object]] = []
        self.executions: list[ToolCall] = []
        self.input: History = [{"role": "user", "content": "Inspect the workspace."}]
        self.sleeps: list[float] = []
        self.enterContext(patch("openai._base_client.time.sleep", side_effect=self.sleeps.append))

    def run_stream(self, responses: Sequence[Fixture], *, account: object = ..., model: object = "test/model",
                   compact_only: bool = False, subagents: bool = True,
                   execute: Callable[[list[ToolCall]], list[str]] | None = None) -> list[ReplyEvent]:
        real_openai = openai.OpenAI

        def handler(request: httpx2.Request) -> httpx2.Response:
            self.bodies.append(record(json_value(cast(object, json.loads(request.content)))))
            self.headers.append(request.headers)
            self.urls.append(str(request.url))
            index = len(self.bodies) - 1
            self.assertLess(index, len(responses), "Unexpected extra model request.")
            fixture = responses[index]
            if isinstance(fixture, Exception):
                raise fixture
            if isinstance(fixture, tuple):
                status, body = fixture
                return httpx2.Response(status, json=body)
            return httpx2.Response(200, content=sse(fixture), headers={"content-type": "text/event-stream"})

        def create_client(*, api_key: str, base_url: str, default_headers: dict[str, str],
                          timeout: float, max_retries: int) -> openai.OpenAI:
            self.constructed.append({"api_key": api_key, "base_url": base_url, "default_headers": default_headers,
                                     "timeout": timeout, "max_retries": max_retries})
            client = httpx2.Client(transport=httpx2.MockTransport(handler))
            self.clients.append(client)
            return real_openai(api_key=api_key, base_url=base_url, default_headers=default_headers,
                               timeout=timeout, max_retries=max_retries, http_client=client)

        def tool_execute(tool_calls: ToolCall | list[ToolCall]) -> list[str]:
            entries = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            self.executions.extend(deepcopy(entries))
            return execute(entries) if execute else ["tool result"] * len(entries)

        output = io.StringIO()
        with (patch("openai.OpenAI", side_effect=create_client),
              patch.object(Tools, "execute", side_effect=tool_execute),
              redirect_stdout(output), redirect_stderr(output)):
            stream_openrouter_response({"key": KEY, "label": "test"} if account is ... else account, model,
                                       list(self.input), self.events, self.cancel,
                                       compact_only=compact_only, subagents=subagents)
        self.assertNotIn(KEY, output.getvalue())
        self.assertTrue(all(client.is_closed for client in self.clients), "Close every SDK/HTTP client.")
        result: list[ReplyEvent] = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
        self.assertNotIn(KEY, str(result))
        return result

    def assert_failed(self, events: list[ReplyEvent], kind: str, message: str) -> None:
        self.assertEqual(events[-1], (kind, message))
        self.assertNotIn("complete", [event[0] for event in events])

    def test_text_streams_with_usage_and_completes_with_a_message_item(self) -> None:
        events = self.run_stream([[chunk("Hel"), chunk("lo"), chunk(finish="stop"), usage()]])
        self.assertEqual(events[:3], [("text", "Hel"), ("text", "lo"),
                                      ("usage", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})])
        self.assertEqual(events[3], ("complete", self.input + [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}]))
        self.assertEqual(len(events), 4)

    def test_request_targets_openrouter_with_prompt_tools_reasoning_and_usage(self) -> None:
        self.run_stream([[chunk("Done", finish="stop")]])
        self.assertEqual(self.constructed, [{"api_key": KEY, "base_url": "https://openrouter.ai/api/v1",
                                             "default_headers": {"X-OpenRouter-Title": "Vin"},
                                             "timeout": 60, "max_retries": 2}])
        self.assertEqual(self.urls, ["https://openrouter.ai/api/v1/chat/completions"])
        self.assertEqual(self.headers[0]["authorization"], f"Bearer {KEY}")
        self.assertEqual(self.headers[0]["x-openrouter-title"], "Vin")
        self.assertNotIn("http-referer", self.headers[0])
        body, = self.bodies
        self.assertEqual(body["model"], "test/model")
        self.assertEqual(body["messages"], [
            {"role": "system", "content": prompts.SYSTEM_PROMPT + prompts.TASK_PROMPT},
            {"role": "user", "content": "Inspect the workspace."}])
        self.assertIs(body["stream"], True)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["reasoning"], {"effort": "medium"})
        self.assertNotIn("context_management", body)
        tools = records(body["tools"])
        self.assertEqual({str(record(tool["function"])["name"]) for tool in tools}, {"read", "write", "edit", "bash", "task"})
        for tool in tools:
            self.assertEqual(set(tool), {"type", "function"})
            self.assertEqual(tool["type"], "function")
            self.assertEqual(set(record(tool["function"])), {"name", "description", "parameters"})

    def test_subagents_off_drops_the_task_tool_and_its_prompt(self) -> None:
        self.run_stream([[chunk("Done", finish="stop")]], subagents=False)
        body, = self.bodies
        self.assertEqual(records(body["messages"])[0], {"role": "system", "content": prompts.SYSTEM_PROMPT})
        self.assertEqual({str(record(tool["function"])["name"]) for tool in records(body["tools"])},
                         {"read", "write", "edit", "bash"})

    def test_compaction_request_sends_no_tools_and_no_text(self) -> None:
        events = self.run_stream([[chunk("Summary", finish="stop")]], compact_only=True)
        body, = self.bodies
        self.assertNotIn("tools", body)
        self.assertEqual(records(body["messages"])[0], {"role": "system", "content": prompts.COMPACTION_PROMPT})
        self.assertNotIn("text", [event[0] for event in events])

    def test_tool_calls_split_across_chunks_run_and_return_to_the_model(self) -> None:
        events = self.run_stream([
            [chunk("Checking."),
             chunk(tool_calls=[tool_delta(0, call_id="call_a", name="read", arguments='{"pa')]),
             chunk(tool_calls=[tool_delta(1, call_id="call_b", name="bash", arguments="")]),
             chunk(tool_calls=[tool_delta(0, arguments='th":"f"}'), tool_delta(1, arguments='{"command":"ls"}')]),
             chunk(finish="tool_calls")],
            [chunk("All done", finish="stop")],
        ], execute=lambda calls: [{"read": "file text", "bash": "listing"}[called(call)[0]] for call in calls])
        self.assertEqual([called(call) for call in self.executions],
                         [("read", '{"path":"f"}'), ("bash", '{"command":"ls"}')])
        self.assertEqual(events[-1][0], "complete")
        follow_up = records(self.bodies[1]["messages"])
        self.assertEqual(follow_up[2:], [
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "read", "arguments": '{"path":"f"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "file text"},
            {"role": "tool", "tool_call_id": "call_b", "content": "listing"},
        ])

    def test_tool_call_without_arguments_sends_an_empty_object(self) -> None:
        self.run_stream([
            [chunk(tool_calls=[tool_delta(0, call_id="call_a", name="bash")]), chunk(finish="tool_calls")],
            [chunk("ok", finish="stop")],
        ])
        self.assertEqual(called(self.executions[0]), ("bash", "{}"))

    def test_tool_call_without_an_id_is_rejected(self) -> None:
        events = self.run_stream([[chunk(tool_calls=[tool_delta(0, name="read", arguments="{}")]),
                                   chunk(finish="tool_calls")]])
        self.assert_failed(events, "error", "OpenRouter returned an invalid tool call.")
        self.assertEqual(self.executions, [])

    def test_length_finish_is_an_incomplete_response(self) -> None:
        events = self.run_stream([[chunk("Partial"), chunk(finish="length")]])
        self.assert_failed(events, "error", "OpenRouter returned an incomplete response.")

    def test_stream_without_a_finish_reason_did_not_complete(self) -> None:
        events = self.run_stream([[chunk("Partial")]])
        self.assert_failed(events, "error", "OpenRouter stream ended before completion.")

    def test_mid_stream_error_chunk_is_reported_without_completing(self) -> None:
        failure = chunk("", finish="error")
        failure["error"] = {"code": 502, "message": f"provider down {KEY}",
                            "metadata": {"error_type": "provider_unavailable"}}
        events = self.run_stream([[chunk("Hel"), failure]])
        self.assert_failed(events, "error", "OpenRouter response did not complete. Please try again.")

    def test_empty_reply_is_rejected_by_the_agent_loop(self) -> None:
        events = self.run_stream([[chunk(finish="stop")]])
        self.assertEqual(events[-1][0], "error")
        self.assertIn("The model returned an empty response", str(events[-1][1]))

    def test_http_errors_map_to_openrouter_messages(self) -> None:
        cases = [
            (401, 1, "error", "OpenRouter key was rejected. Use /login to enter a new one."),
            (402, 1, "error", "OpenRouter credits are used up. Add credits and try again."),
            (403, 1, "error", "OpenRouter blocked this request (HTTP 403). Check your key's limits or try another model."),
            (429, 3, "rate_limit", "OpenRouter rate limit reached (HTTP 429). Please wait and try again."),
            (502, 3, "error", "OpenRouter request failed (HTTP 502)."),
        ]
        for status, attempts, kind, message in cases:
            with self.subTest(status=status):
                self.setUp()
                failure: Fixture = (status, {"error": {"code": status, "message": f"bad {KEY}"}})
                events = self.run_stream([failure] * attempts)
                self.assertEqual(events, [(kind, message)])
                self.assertEqual(len(self.bodies), attempts)
                self.assertEqual(len(self.sleeps), attempts - 1)

    def test_a_too_long_conversation_says_how_to_recover(self) -> None:
        too_long = ("This endpoint's maximum context length is 131072 tokens. "
                    "However, you requested about 150000 tokens.")
        cases: list[Fixture] = [
            (400, {"error": {"code": 400, "message": too_long}}),
            (413, {"error": {"code": 413, "message": "Prompt exceeds the context length of this model"}}),
        ]
        for failure in cases:
            with self.subTest(failure=failure):
                self.setUp()
                self.assertEqual(self.run_stream([failure]), [("error", agent.CONTEXT_FULL)])

    def test_a_plain_bad_request_keeps_the_generic_message(self) -> None:
        events = self.run_stream([(400, {"error": {"code": 400, "message": "Invalid tool schema"}})])
        self.assertEqual(events, [("error", "OpenRouter request failed (HTTP 400).")])

    def test_a_transient_failure_is_retried_and_the_turn_completes(self) -> None:
        failures: list[Fixture] = [
            (503, {"error": {"code": 503, "message": "busy"}}),
            httpx2.ConnectError("reset", request=httpx2.Request("POST", "https://openrouter.ai")),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.setUp()
                events = self.run_stream([failure, [chunk("Done", finish="stop")]])
                self.assertEqual(events[-1][0], "complete")
                self.assertIn(("text", "Done"), events)
                self.assertEqual(len(self.bodies), 2)
                self.assertEqual(self.bodies[0], self.bodies[1])
                self.assertEqual(len(self.sleeps), 1)

    def test_network_failures_map_to_openrouter_messages(self) -> None:
        request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        cases: list[tuple[Exception, str]] = [
            (httpx2.ConnectError(KEY, request=request), "Could not connect to OpenRouter. Please try again."),
            (httpx2.ReadTimeout(KEY, request=request), "OpenRouter request timed out. Please try again."),
        ]
        for error, message in cases:
            with self.subTest(error=type(error).__name__):
                self.setUp()
                self.assertEqual(self.run_stream([error] * 3), [("error", message)])
                self.assertEqual(len(self.sleeps), 2)

    def test_unexpected_failure_uses_the_generic_message(self) -> None:
        with patch("vin.provider.openrouter._request_completion", side_effect=ValueError(KEY)):
            events = self.run_stream([])
        self.assertEqual(events, [("error", "OpenRouter request failed. Please try again.")])

    def test_cleanup_failure_cannot_publish_a_success(self) -> None:
        def failing_close() -> None:
            for client in self.clients:
                client.close()
            raise OSError(KEY)
        with patch.object(openai.OpenAI, "close", side_effect=failing_close):
            events = self.run_stream([[chunk("Hello", finish="stop")]])
        self.assertEqual(events[0], ("text", "Hello"))
        self.assert_failed(events, "error", "OpenRouter request failed. Please try again.")

    def test_missing_key_or_model_stops_before_any_request(self) -> None:
        cases: list[tuple[object, object, str]] = [
            (None, "test/model", "Sign in with /login before sending a message."),
            ({"key": "  "}, "test/model", "Sign in with /login before sending a message."),
            ({"key": KEY}, None, "Choose a model with /model before sending a message."),
            ({"key": KEY}, " ", "Choose a model with /model before sending a message."),
        ]
        for account, model, message in cases:
            with self.subTest(account=account, model=model):
                self.setUp()
                self.assertEqual(self.run_stream([], account=account, model=model), [("error", message)])
                self.assertEqual(self.constructed, [])

    def test_cancel_before_start_sends_nothing(self) -> None:
        self.cancel.set()
        self.assertEqual(self.run_stream([]), [])
        self.assertEqual(self.bodies, [])

    def test_cancel_mid_stream_stops_without_completing(self) -> None:
        def execute(calls: list[ToolCall]) -> list[str]:
            self.cancel.set()
            return ["x"] * len(calls)
        events = self.run_stream([
            [chunk(tool_calls=[tool_delta(0, call_id="call_a", name="read", arguments="{}")]), chunk(finish="tool_calls")],
            [chunk("never", finish="stop")],
        ], execute=execute)
        self.assertNotIn("complete", [event[0] for event in events])
        self.assertNotIn(("text", "never"), events)


class ToChatMessagesTests(unittest.TestCase):
    def test_history_items_convert_to_chat_messages(self) -> None:
        history: History = [
            {"role": "user", "content": "Fix it."},
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Look"}, {"type": "output_text", "text": "ing."}]},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}", "status": "completed"},
            {"type": "function_call_output", "call_id": "c1", "output": "body"},
            {"type": "compaction", "id": "cmp", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "c2", "name": "bash", "arguments": '{"command":"ls"}'},
            {"type": "function_call", "call_id": "c3", "name": "read", "arguments": '{"path":"a"}'},
            {"type": "function_call_output", "call_id": "c2", "output": "a"},
            {"type": "function_call_output", "call_id": "c3", "output": "text"},
            {"role": "user", "content": [{"type": "input_text", "text": "Thanks"}]},
        ]
        original = deepcopy(history)
        self.assertEqual(to_chat_messages(history, "PROMPT"), [
            {"role": "system", "content": "PROMPT"},
            {"role": "user", "content": "Fix it."},
            {"role": "assistant", "content": "Looking.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "body"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}},
                {"id": "c3", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "a"},
            {"role": "tool", "tool_call_id": "c3", "content": "text"},
            {"role": "user", "content": "Thanks"},
        ])
        self.assertEqual(history, original)

    def test_assistant_text_after_a_user_message_does_not_reuse_an_old_assistant(self) -> None:
        history: History = [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "One"}]},
            {"role": "user", "content": "Next"},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        ]
        messages = to_chat_messages(history, "P")
        self.assertEqual(messages[1], {"role": "assistant", "content": "One"})
        self.assertEqual(messages[3], {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})


if __name__ == "__main__":
    unittest.main()
