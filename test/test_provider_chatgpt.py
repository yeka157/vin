import io
import json
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from collections.abc import Generator, Iterable, Iterator
from email.message import Message
from typing import Literal, TypedDict, cast
from urllib.request import Request
from threading import Event
from queue import SimpleQueue
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

import httpx2
import openai
from openai.types.responses import (
    Response, ResponseStatus, ResponseStreamEvent, ResponseOutputTextParam, ResponseOutputRefusalParam,
    ResponseCompletedEvent, ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)

from vin.provider import chatgpt
from vin.provider.events import ReplyEvent, ReplyEvents
from vin.types import History, JSONValue
from provider_typing_helpers import ClientDouble, RequestSnapshot, REQUEST_ADAPTER, credentials as fixture_credentials


class DeltaFields(TypedDict):
    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int

type Lifecycle = ResponseCompletedEvent | ResponseCreatedEvent | ResponseFailedEvent | ResponseIncompleteEvent


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials = fixture_credentials("fake-access-token")

    def test_visible_models_keep_server_order(self) -> None:
        first = {"slug": "model-b", "display_name": "B", "visibility": "list"}
        second = {"slug": "model-a", "display_name": "A", "visibility": "list"}
        response = io.BytesIO(
            json.dumps({"models": [first, {"visibility": "hidden"}, second]}).encode()
        )

        with patch.object(chatgpt, "urlopen", return_value=response) as request:
            result = chatgpt.list_chatgpt_models(self.credentials)

        self.assertEqual(result, [first, second])
        self.assertTrue(response.closed)
        request.assert_called_once()
        sent: object = cast(object, request.call_args.args[0])
        assert isinstance(sent, Request)
        self.assertEqual(sent.full_url, "https://api.openai.com/v1/models")
        self.assertEqual(sent.get_method(), "GET")
        self.assertEqual(sent.get_header("Authorization"), "Bearer fake-access-token")
        self.assertEqual(sent.get_header("Accept"), "application/json")
        self.assertEqual(request.call_args.kwargs, {"timeout": 60})

    def test_empty_or_hidden_catalog_returns_no_choices(self) -> None:
        for models in ([], [{"visibility": "hidden"}]):
            with self.subTest(models=models):
                response = io.BytesIO(json.dumps({"models": models}).encode())
                with patch.object(chatgpt, "urlopen", return_value=response):
                    self.assertEqual(chatgpt.list_chatgpt_models(self.credentials), [])
                self.assertTrue(response.closed)

    def test_missing_credentials_or_permission_prevents_request(self) -> None:
        cases: list[object] = [None, {}]
        cases.extend(
            {**self.credentials, "access_token": token}
            for token in (None, "", " ", 123)
        )
        cases.extend([
            {"access_token": "fake-access-token"},
            {**self.credentials, "scopes": ["openid"]},
        ])
        for credentials in cases:
            with self.subTest(credentials=credentials):
                with patch.object(chatgpt, "urlopen") as request:
                    with self.assertRaises(RuntimeError):
                        chatgpt.list_chatgpt_models(credentials)
                    request.assert_not_called()

    def test_invalid_json_is_rejected_and_response_closed(self) -> None:
        for body in (b"not json", b"\xff"):
            with self.subTest(body=body):
                response = io.BytesIO(body)
                with patch.object(chatgpt, "urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                        chatgpt.list_chatgpt_models(self.credentials)
                self.assertTrue(response.closed)

    def test_invalid_catalog_is_rejected(self) -> None:
        valid = {"slug": "test-model", "display_name": "Test", "visibility": "list"}
        payloads: list[JSONValue] = [None, [], {}, {"models": None}, {"models": {}}, {"models": [None]}]
        for field in ("slug", "display_name"):
            for value in (None, "", " ", 123):
                payloads.append({"models": [{**valid, field: value}]})

        for payload in payloads:
            with self.subTest(payload=payload):
                response = io.BytesIO(json.dumps(payload).encode())
                with patch.object(chatgpt, "urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "invalid"):
                        chatgpt.list_chatgpt_models(self.credentials)
                self.assertTrue(response.closed)

    def test_http_failure_hides_body_closes_response_and_does_not_retry(self) -> None:
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                body = io.BytesIO(b"fake-access-token")
                error = HTTPError(
                    "https://api.openai.com/v1/models", status, "failure", Message(), body
                )
                with patch.object(chatgpt, "urlopen", side_effect=error) as request:
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as caught:
                        chatgpt.list_chatgpt_models(self.credentials)
                request.assert_called_once()
                self.assertTrue(body.closed)
                self.assertNotIn("fake-access-token", str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_connection_failure_and_timeout_do_not_retry(self) -> None:
        for error in (URLError("fake-access-token"), TimeoutError("fake-access-token")):
            with self.subTest(error=type(error).__name__):
                with patch.object(chatgpt, "urlopen", side_effect=error) as request:
                    with self.assertRaisesRegex(RuntimeError, "Could not connect") as caught:
                        chatgpt.list_chatgpt_models(self.credentials)
                request.assert_called_once()
                self.assertNotIn("fake-access-token", str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)


class StreamingResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials = fixture_credentials("fake-access-token")
        self.messages: History = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Next question"},
        ]
        self.events: SimpleQueue[ReplyEvent] = ReplyEvents()
        self.cancel = Event()
        self.closed: list[str] = []
        self.client = ClientDouble(lambda: self.closed.append("client"))

    def delta(self, text: str, *, refusal: bool = False) -> ResponseRefusalDeltaEvent | ResponseTextDeltaEvent:
        values: DeltaFields = {
            "content_index": 0, "delta": text, "item_id": "msg_test",
            "output_index": 0, "sequence_number": 1,
        }
        if refusal:
            return ResponseRefusalDeltaEvent(type="response.refusal.delta", **values)
        return ResponseTextDeltaEvent(
            type="response.output_text.delta", logprobs=[], **values
        )

    def lifecycle(self, event_type: Literal["created", "completed", "failed", "incomplete"] = "completed", *, status: ResponseStatus | None = None, text: str = "Hello", refusal: bool = False) -> Lifecycle:
        status = status or ("in_progress" if event_type == "created" else event_type)
        content: ResponseOutputRefusalParam | ResponseOutputTextParam = (
            {"type": "refusal", "refusal": text}
            if refusal else
            {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        )
        response = Response.model_validate(dict(
            id="resp_test", created_at=0, model="selected-model", object="response",
            status=status, parallel_tool_calls=False, tool_choice="none", tools=[],
            output=[{
                "id": "msg_test", "type": "message", "role": "assistant",
                "status": "completed" if status == "completed" else "incomplete",
                "content": [content],
            }],
        )
        )
        event_class = {
            "created": ResponseCreatedEvent,
            "completed": ResponseCompletedEvent,
            "failed": ResponseFailedEvent,
            "incomplete": ResponseIncompleteEvent,
        }[event_type]
        return event_class.model_validate(dict(
            type=f"response.{event_type}", response=response, sequence_number=2
        ))

    @contextmanager
    def stream(self, source: Iterable[ResponseStreamEvent]) -> Generator[Iterator[ResponseStreamEvent]]:
        try:
            yield iter(source)
        finally:
            self.closed.append("stream")

    def invoke(self, source: Iterable[ResponseStreamEvent] = (), *, credentials: object = ..., model: object = "selected-model") -> Mock:
        self.client.responses.create.return_value = self.stream(source)
        if credentials is ...:
            credentials = self.credentials
        output = io.StringIO()
        with (
            patch("openai.OpenAI", return_value=self.client) as constructor,
            redirect_stdout(output), redirect_stderr(output),
        ):
            chatgpt.stream_chatgpt_response(
                credentials, model, self.messages, self.events, self.cancel
            )
        self.assertEqual(output.getvalue(), "")
        return constructor

    def drain(self) -> list[ReplyEvent]:
        result: list[ReplyEvent] = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
        return result

    def test_chunks_arrive_before_completion_with_selected_model_and_context(self) -> None:
        original = deepcopy(self.messages)
        submitted: RequestSnapshot | None = None

        def source() -> Iterator[ResponseStreamEvent]:
            nonlocal submitted
            submitted = REQUEST_ADAPTER.validate_python(cast(object, self.client.responses.create.call_args.kwargs))
            yield self.lifecycle("created", status="in_progress")
            yield self.delta("")
            yield self.delta("Hel")
            self.assertEqual(self.events.get_nowait(), ("text", "Hel"))
            yield self.delta("lo")
            self.assertEqual(self.events.get_nowait(), ("text", "lo"))
            yield self.lifecycle()
            self.fail("The worker read past the completion event.")

        constructor = self.invoke(source())
        output = self.lifecycle().response.output[0].model_dump(exclude_none=True)
        self.assertEqual(self.drain(), [("complete", original + [output])])
        self.assertEqual(self.closed, ["stream", "client"])
        self.assertEqual(self.messages, original)
        constructor.assert_called_once_with(
            api_key="fake-access-token", base_url="https://api.openai.com/v1",
            timeout=60, max_retries=0,
        )
        from vin.tools import _response_tools
        self.client.responses.create.assert_called_once()
        self.assertEqual(submitted, {
            "model": "selected-model", "input": original,
            "instructions": chatgpt.SYSTEM_PROMPT,
            "tools": _response_tools(), "reasoning": {"effort": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False, "stream": True,
        })

    def test_added_null_item_does_not_block_completed_done_output(self) -> None:
        completed = self.lifecycle(text="Done")
        item = completed.response.output[0]
        missing = ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=None,
            output_index=0, sequence_number=1,
        )
        done = ResponseOutputItemDoneEvent(
            type="response.output_item.done", item=item,
            output_index=1, sequence_number=2,
        )
        completed.response.output = []

        self.invoke([missing, done, completed])

        self.assertEqual(self.drain(), [
            ("text", "Done"),
            ("complete", self.messages + [item.model_dump(exclude_none=True)]),
        ])
        self.assertEqual(self.closed, ["stream", "client"])

    def test_tool_use_sets_custom_queue_marker(self) -> None:
        class Marker:
            def __init__(self) -> None:
                self.calls = 0

            def set(self) -> None:
                self.calls += 1

        class MarkedEvents(SimpleQueue[ReplyEvent]):
            def __init__(self) -> None:
                super().__init__()
                self.tools_used = Marker()

        events = MarkedEvents()
        self.events = events
        tool_response = self.lifecycle()
        tool_response.response.output = [ResponseFunctionToolCall(
            id="tool_test", type="function_call", status="completed",
            call_id="call_test", name="read", arguments='{"path":"sample.txt"}',
        )]
        self.client.responses.create.side_effect = [
            self.stream([tool_response]),
            self.stream([self.lifecycle(text="Done")]),
        ]
        with patch("vin.tools.TOOL_REGISTRY.execute", return_value=["contents"]):
            self.invoke()

        self.assertEqual(events.tools_used.calls, 1)
        emitted = self.drain()
        self.assertEqual(emitted[:2], [
            ("tool_started", {"name": "read", "detail": "sample.txt"}),
            ("tool_finished", "read"),
        ])
        self.assertEqual(emitted[-1][0], "complete")
        self.assertEqual(self.closed, ["stream", "stream", "client"])

    def test_refusal_text_is_streamed_and_can_complete(self) -> None:
        constructor = self.invoke([
            self.delta("I cannot help.", refusal=True),
            self.lifecycle(text="I cannot help.", refusal=True),
        ])
        output = self.lifecycle(text="I cannot help.", refusal=True).response.output[0]
        self.assertEqual(self.drain(), [
            ("text", "I cannot help."),
            ("complete", self.messages + [output.model_dump(exclude_none=True)]),
        ])
        constructor.assert_called_once()
        self.assertEqual(self.closed, ["stream", "client"])

    def test_missing_token_permission_or_model_prevents_request(self) -> None:
        cases: list[tuple[object, object, str]] = [(None, "selected-model", "Sign in"), ({}, "selected-model", "Sign in")]
        cases.extend(
            ({**self.credentials, "access_token": token}, "selected-model", "Sign in")
            for token in (None, "", " ", 123)
        )
        cases.extend([
            ({"access_token": "fake-access-token"}, "selected-model", "plan usage"),
            ({**self.credentials, "scopes": []}, "selected-model", "plan usage"),
            ({**self.credentials, "scopes": None}, "selected-model", "plan usage"),
            (self.credentials, None, "/model"),
            (self.credentials, "", "/model"),
            (self.credentials, " ", "/model"),
        ])
        for credentials, model, hint in cases:
            with self.subTest(credentials=credentials, model=model):
                constructor = self.invoke(credentials=credentials, model=model)
                constructor.assert_not_called()
                outcome, = self.drain()
                self.assertEqual(outcome[0], "error")
                assert outcome[0] == "error"
                self.assertIn(hint, outcome[1])
                self.assertNotIn("fake-access-token", outcome[1])

    def test_eof_failure_incomplete_and_empty_output_never_complete(self) -> None:
        error_event = ResponseErrorEvent(
            type="error", code="server_error", message="fake-access-token",
            param=None, sequence_number=2,
        )
        cases: list[list[ResponseStreamEvent]] = [
            [], [self.delta("partial")],
            [self.delta("partial"), self.lifecycle("failed")],
            [self.delta("partial"), self.lifecycle("incomplete")],
            [self.delta("partial"), error_event],
            [self.delta("partial"), self.lifecycle(status="incomplete")],
            [self.lifecycle(text="")],
            [self.delta(" \n"), self.lifecycle(text=" \n")],
        ]
        for source in cases:
            with self.subTest(events=[event.type for event in source]):
                self.closed.clear()
                self.invoke(source)
                result = self.drain()
                self.assertEqual(result[-1][0], "error")
                self.assertNotIn("complete", [kind for kind, _ in result])
                self.assertNotIn("fake-access-token", str(result))
                if source and source[0].type == "response.output_text.delta":
                    self.assertEqual(result[0], ("text", source[0].delta))
                self.assertEqual(self.closed, ["stream", "client"])

    def test_api_and_network_errors_preserve_partial_without_leaking_secrets(self) -> None:
        request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        failures: list[tuple[Exception, str, str]] = [
            (openai.APIStatusError(
                "fake-access-token", response=httpx2.Response(status, request=request),
                body={"message": "fake-access-token"},
            ), f"HTTP {status}", kind)
            for status, kind in ((401, "error"), (403, "error"), (429, "rate_limit"), (500, "error"))
        ]
        failures.extend([
            (openai.APITimeoutError(request), "timed out", "error"),
            (openai.APIConnectionError(request=request, message="fake-access-token"), "connect", "error"),
            (openai.APIError("fake-access-token", request, body=None), "failed", "error"),
            (RuntimeError("fake-access-token"), "failed", "error"),
        ])
        for error, hint, kind in failures:
            for after_partial in (False, True):
                with self.subTest(error=type(error).__name__, partial=after_partial):
                    self.closed.clear()
                    self.client.responses.create.reset_mock(side_effect=True)

                    def source() -> Iterator[ResponseStreamEvent]:
                        yield self.delta("partial")
                        raise error

                    if not after_partial:
                        self.client.responses.create.side_effect = error
                    constructor = self.invoke(source())
                    result = self.drain()
                    self.assertEqual(result[-1][0], kind)
                    terminal = result[-1]
                    assert isinstance(terminal[1], str)
                    self.assertIn(hint, terminal[1])
                    self.assertNotIn("fake-access-token", str(result))
                    self.assertEqual(result[:-1], [("text", "partial")] if after_partial else [])
                    self.assertEqual(self.closed, ["stream", "client"] if after_partial else ["client"])
                    constructor.assert_called_once()
                    self.client.responses.create.assert_called_once()

    def test_cancel_before_request_or_while_waiting_for_headers_suppresses_results(self) -> None:
        self.cancel.set()
        constructor = self.invoke()
        constructor.assert_not_called()
        self.assertEqual(self.drain(), [])
        self.cancel.clear()
        def cancel_headers(**kwargs: object) -> object:
            self.cancel.set()
            return cast(object, self.client.responses.create.return_value)
        self.client.responses.create.side_effect = cancel_headers

        def source() -> Iterator[ResponseStreamEvent]:
            self.fail("A cancelled request should close without reading more events.")
            yield

        self.invoke(source())
        self.assertEqual(self.drain(), [])
        self.assertEqual(self.closed, ["stream", "client"])

    def test_cancellation_discards_late_text_completion_and_error(self) -> None:
        for late_error in (False, True):
            with self.subTest(late_error=late_error):
                self.cancel.clear()
                self.closed.clear()

                def source() -> Iterator[ResponseStreamEvent]:
                    yield self.delta("partial")
                    self.cancel.set()
                    if late_error:
                        raise RuntimeError("fake-access-token")
                    yield self.delta("late")
                    yield self.lifecycle()

                self.invoke(source())
                self.assertEqual(self.drain(), [("text", "partial")])
                self.assertEqual(self.closed, ["stream", "client"])

    def test_cleanup_failure_cannot_publish_a_success(self) -> None:
        self.client.exit_error = OSError("fake-access-token")
        self.invoke([self.delta("Hello"), self.lifecycle()])
        result = self.drain()
        self.assertEqual(result[0], ("text", "Hello"))
        self.assertEqual(result[1][0], "error")
        self.assertNotIn("fake-access-token", str(result))

    def test_api_diagnostics_hide_credentials_in_code_and_parameter_fields(self) -> None:
        request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        for code, param in (("fake-access-token", "reasoning.effort"),
                            ("unsupported_value", "fake-access-token"),
                            ("bad\nprovider output", "bad\x1b[31mparameter")):
            with self.subTest(code=code, param=param):
                self.client.responses.create.side_effect = openai.APIStatusError(
                    "fake-access-token",
                    response=httpx2.Response(400, request=request),
                    body={"message": "fake-access-token", "code": code, "param": param},
                )
                self.invoke()
                event = self.drain()[-1]
                assert event[0] == "error"
                kind, diagnostic = event
                self.assertEqual(kind, "error")
                self.assertIn("400", diagnostic)
                self.assertNotIn("fake-access-token", diagnostic)
                self.assertNotIn("\n", diagnostic)
                self.assertNotIn("\x1b", diagnostic)
