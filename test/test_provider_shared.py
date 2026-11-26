"""The shared provider flow: checks before sending, plugging in any client, and error wording."""

from contextlib import AbstractContextManager
from threading import Event
from types import TracebackType
import unittest

import httpx2
import openai

from vin import agent
from vin.provider.events import EventSender, ReplyEvent, ReplyEvents
from vin.provider.shared import Outcome, error_message, stream_reply
from vin.types import History

SECRET = "sk-secret-value"


def status_error(status: int, code: object = None, message: str = "bad") -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://example.test/v1")
    return openai.APIStatusError(message, response=httpx2.Response(status, request=request),
                                 body={"message": message, "code": code})


class FakeClient(AbstractContextManager[object]):
    def __init__(self) -> None:
        self.closed = False

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.closed = True


class StreamReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = ReplyEvents()
        self.cancel = Event()
        self.clients: list[FakeClient] = []
        self.keys: list[str] = []
        self.searches: list[bool] = []

    def connect(self, key: str) -> FakeClient:
        self.keys.append(key)
        client = FakeClient()
        self.clients.append(client)
        return client

    def run_reply(self, key: str | None = SECRET, model: object = "any/model", problem: str | None = None,
                  search: bool | None = None) -> list[ReplyEvent]:
        def complete(client: FakeClient, model: str, history: History, send: EventSender, cancel_event: Event, *,
                     compact_only: bool, subagent: bool, task: bool, search: bool) -> tuple[History, bool]:
            self.searches.append(search)
            send("text", f"answer from {model}")
            return [{"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "answer"}]}], True

        if search is None:
            stream_reply("Fake", key, model, [{"role": "user", "content": "hi"}], self.events, self.cancel,
                         connect=self.connect, complete=complete, problem=problem)
        else:
            stream_reply("Fake", key, model, [{"role": "user", "content": "hi"}], self.events, self.cancel,
                         connect=self.connect, complete=complete, problem=problem, search=search)
        result: list[ReplyEvent] = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
        return result

    def test_any_client_and_wire_format_can_plug_in(self) -> None:
        events = self.run_reply()
        self.assertEqual(events[0], ("text", "answer from any/model"))
        self.assertEqual(events[-1][0], "complete")
        self.assertEqual(self.keys, [SECRET])
        self.assertTrue(self.clients[0].closed)

    def test_web_search_is_off_unless_asked_for_and_reaches_the_request(self) -> None:
        _ = self.run_reply()
        _ = self.run_reply(search=True)
        _ = self.run_reply(search=False)
        self.assertEqual(self.searches, [False, True, False])

    def test_checks_run_in_order_before_connecting(self) -> None:
        cases: list[tuple[str | None, object, str | None, str]] = [
            (None, "m", None, "Sign in with /login before sending a message."),
            ("  ", "m", "ignored", "Sign in with /login before sending a message."),
            (SECRET, "m", "Plan is not enabled.", "Plan is not enabled."),
            (SECRET, None, None, "Choose a model with /model before sending a message."),
            (SECRET, " ", None, "Choose a model with /model before sending a message."),
        ]
        for key, model, problem, message in cases:
            with self.subTest(message=message, key=key, model=model):
                self.setUp()
                self.assertEqual(self.run_reply(key, model, problem), [("error", message)])
                self.assertEqual(self.clients, [])

    def test_cancel_before_start_sends_nothing(self) -> None:
        self.cancel.set()
        self.assertEqual(self.run_reply(), [])
        self.assertEqual(self.clients, [])


class ErrorMessageTests(unittest.TestCase):
    def test_wording_uses_the_provider_name(self) -> None:
        request = httpx2.Request("POST", "https://example.test/v1")
        cases: list[tuple[Exception, str]] = [
            (openai.APITimeoutError(request), "Fake request timed out. Please try again."),
            (openai.APIConnectionError(request=request), "Could not connect to Fake. Please try again."),
            (openai.APIError("stream broke", request, body=None), "Fake response did not complete. Please try again."),
            (RuntimeError(SECRET), "Fake request failed. Please try again."),
            (status_error(500), "Fake request failed (HTTP 500)."),
        ]
        for error, message in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(error_message(error, "Fake", SECRET, {}), ("error", message))

    def test_provider_statuses_win_and_429_is_a_rate_limit(self) -> None:
        statuses: dict[int, Outcome] = {402: ("error", "Out of credits.")}
        self.assertEqual(error_message(status_error(402), "Fake", SECRET, statuses), ("error", "Out of credits."))
        self.assertEqual(error_message(status_error(429), "Fake", SECRET, {}),
                         ("rate_limit", "Fake request failed (HTTP 429)."))

    def test_too_long_conversation_beats_provider_statuses(self) -> None:
        error = status_error(400, "context_length_exceeded")
        self.assertEqual(error_message(error, "Fake", SECRET, {400: ("error", "Other.")}), ("error", agent.CONTEXT_FULL))

    def test_safe_codes_are_shown_but_secrets_and_repeated_statuses_are_not(self) -> None:
        self.assertEqual(error_message(status_error(400, "invalid_value"), "Fake", SECRET, {})[1],
                         "Fake request failed (HTTP 400). code=invalid_value.")
        for code in (SECRET, "400", "bad\ncode", 400):
            with self.subTest(code=code):
                self.assertEqual(error_message(status_error(400, code), "Fake", SECRET, {})[1],
                                 "Fake request failed (HTTP 400).")


if __name__ == "__main__":
    unittest.main()
