"""Web search (#17): the Tavily request, result formatting, errors, cancellation and the saved key."""

from email.message import Message
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import cast
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from vin import tools
from vin.agent import execute_tool, tool_detail
from vin.auth import search
from vin.tools import response_tools, tool_cancellation, web_search
from vin.types.data import is_record

KEY = "tvly-secret"


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def reply(results: object) -> Response:
    return Response(json.dumps({"query": "q", "results": results}).encode())


def result(number: int) -> dict[str, object]:
    return {
        "title": f"Title {number}",
        "url": f"https://example.com/{number}",
        "content": f"Snippet {number}",
        "score": 0.5,
    }


def http_error(code: int, body: object = None) -> HTTPError:
    data = b"not json" if body is None else json.dumps(body).encode()
    return HTTPError(tools.TAVILY_SEARCH_URL, code, "error", Message(), io.BytesIO(data))


class WebSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.load_key = self.enterContext(patch.object(search, "load_search_key", return_value=KEY))

    def search(self, response: object, query: str = "vin", limit: int = 5) -> tuple[str, Mock]:
        side_effect = response if isinstance(response, BaseException) else None
        return_value = None if side_effect is not None else response
        with patch.object(tools, "urlopen", side_effect=side_effect, return_value=return_value) as urlopen:
            output = web_search(query, limit)
        return output, urlopen

    def failure(self, response: object) -> str:
        with self.assertRaises(RuntimeError) as raised:
            _ = self.search(response)
        message = str(raised.exception)
        self.assertNotIn(KEY, message)
        self.assertIsNone(raised.exception.__cause__)
        return message

    def sent_body(self, urlopen: Mock) -> dict[str, object]:
        request = cast(Request, urlopen.call_args.args[0])
        data = cast(bytes, request.data)
        body = cast(object, json.loads(data))
        assert is_record(body)
        return body

    def test_posts_query_to_tavily_with_bearer_key_and_formats_results(self) -> None:
        output, urlopen = self.search(reply([result(1), result(2)]), query="  latest prompt-toolkit  ")

        request = cast(Request, urlopen.call_args.args[0])
        self.assertEqual(request.full_url, "https://api.tavily.com/search")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], tools.SEARCH_TIMEOUT)
        self.assertEqual(
            self.sent_body(urlopen),
            {"query": "latest prompt-toolkit", "max_results": 5, "search_depth": "basic"},
        )
        self.assertEqual(
            output,
            'Search results for "latest prompt-toolkit":\n\n'
            "1. Title 1\n   https://example.com/1\n   Snippet 1\n\n"
            "2. Title 2\n   https://example.com/2\n   Snippet 2",
        )
        self.assertNotIn(KEY, output)

    def test_limit_is_clamped_and_caps_the_entries_shown(self) -> None:
        for limit, sent in ((0, 1), (-3, 1), (1, 1), (10, 10), (11, 10), (50, 10)):
            with self.subTest(limit=limit):
                _, urlopen = self.search(reply([]), limit=limit)
                self.assertEqual(self.sent_body(urlopen)["max_results"], sent)

        output, _ = self.search(reply([result(number) for number in range(1, 6)]), limit=2)
        self.assertIn("2. Title 2", output)
        self.assertNotIn("Title 3", output)

    def test_snippets_are_decoded_collapsed_and_cut(self) -> None:
        messy = {"title": "T", "url": "https://a.dev", "content": "a &amp; b\n\n\t c"}
        output, _ = self.search(reply([messy]))
        self.assertIn("   a & b c", output)

        long = {"title": "T", "url": "https://a.dev", "content": "x" * 500}
        output, _ = self.search(reply([long]))
        snippet = output.splitlines()[-1].strip()
        self.assertEqual(len(snippet), tools.MAX_SNIPPET_CHARS)
        self.assertTrue(snippet.endswith("…"))

    def test_malformed_items_are_skipped_without_gaps_in_numbering(self) -> None:
        items: list[object] = [
            "not a record",
            {"title": "No URL", "content": "x"},
            {"title": "Empty URL", "url": "", "content": "x"},
            {"title": 7, "url": "https://untitled.dev", "content": None},
            result(2),
        ]
        output, _ = self.search(reply(items))
        self.assertEqual(
            output,
            'Search results for "vin":\n\n'
            "1. https://untitled.dev\n   https://untitled.dev\n   \n\n"
            "2. Title 2\n   https://example.com/2\n   Snippet 2",
        )

    def test_no_results(self) -> None:
        for items in ([], ["junk", {"title": "no url"}]):
            with self.subTest(items=items):
                output, _ = self.search(reply(items))
                self.assertEqual(output, 'No results for "vin".')

    def test_empty_query_is_rejected_before_any_request(self) -> None:
        for query in ("", "   \n"):
            with self.subTest(query=query), patch.object(tools, "urlopen") as urlopen:
                with self.assertRaisesRegex(ValueError, "Query must not be empty"):
                    _ = web_search(query)
                urlopen.assert_not_called()

    def test_missing_key_asks_for_setup_without_a_request(self) -> None:
        self.load_key.return_value = None
        with patch.object(tools, "urlopen") as urlopen:
            with self.assertRaisesRegex(RuntimeError, r"not set up\. Ask the user to turn it on in /settings"):
                _ = web_search("vin")
        urlopen.assert_not_called()

    def test_http_errors_have_their_own_messages(self) -> None:
        cases = {
            401: "Web search key was rejected. Ask the user to replace the key in /settings.",
            403: "Web search key was rejected. Ask the user to replace the key in /settings.",
            429: "Web search limit reached. Try again later.",
            432: "Web search limit reached. Try again later.",
            433: "Web search limit reached. Try again later.",
            500: "Web search failed (HTTP 500).",
            502: "Web search failed (HTTP 502).",
        }
        for code, message in cases.items():
            with self.subTest(code=code):
                self.assertEqual(self.failure(http_error(code)), message)

    def test_rejected_request_shows_tavilys_reason(self) -> None:
        cases: list[tuple[int, object, str]] = [
            (400, {"detail": {"error": "Invalid topic."}}, "Invalid topic."),
            (422, {"detail": [{"loc": ["body", "query"], "msg": "Field required"}]}, "Field required"),
            (422, {"detail": []}, "no details"),
            (400, {"detail": {"error": 5}}, "no details"),
            (400, ["not", "a", "record"], "no details"),
            (400, None, "no details"),
        ]
        for code, body, reason in cases:
            with self.subTest(code=code, body=body):
                self.assertEqual(
                    self.failure(http_error(code, body)),
                    f"Web search request was rejected: {reason}.",
                )

    def test_network_failures_and_bad_responses(self) -> None:
        self.assertEqual(
            self.failure(URLError("nodename nor servname provided")),
            "Web search failed: nodename nor servname provided.",
        )
        self.assertEqual(self.failure(TimeoutError("timed out")), "Web search failed: timed out.")
        for body in (b"<html>", b'{"results": "nope"}', b"[]"):
            with self.subTest(body=body):
                self.assertEqual(self.failure(Response(body)), "Web search returned an invalid response.")

    def test_cancelled_before_the_request_sends_nothing(self) -> None:
        cancel = Event()
        cancel.set()
        with tool_cancellation(cancel), patch.object(tools, "urlopen") as urlopen:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                _ = web_search("vin")
        urlopen.assert_not_called()

    def test_cancelled_during_the_request_drops_the_results(self) -> None:
        cancel = Event()

        def answer(*_args: object, **_kwargs: object) -> Response:
            cancel.set()
            return reply([result(1)])

        with tool_cancellation(cancel), patch.object(tools, "urlopen", side_effect=answer):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                _ = web_search("vin")

    def test_model_sees_the_error_text_and_the_row_shows_the_query(self) -> None:
        with patch.object(tools, "urlopen", side_effect=http_error(429)):
            output = execute_tool("web_search", json.dumps({"query": "vin"}), Event())
        self.assertEqual(output, "Error (RuntimeError): Web search limit reached. Try again later.")
        self.assertEqual(tool_detail("web_search", json.dumps({"query": "latest vin"})), "latest vin")


class ToolListTests(unittest.TestCase):
    def test_web_search_is_offered_only_when_search_is_on(self) -> None:
        names = [tool["name"] for tool in response_tools(task=True)]
        self.assertNotIn("web_search", names)

        names = [tool["name"] for tool in response_tools(task=True, search=True)]
        self.assertIn("web_search", names)
        self.assertIn("bash", names)

    def test_description_marks_results_as_untrusted(self) -> None:
        (tool,) = [tool for tool in response_tools(search=True) if tool["name"] == "web_search"]
        description = tool.get("description") or ""
        self.assertTrue(description.startswith("Search the web"))
        self.assertIn("untrusted", description)


class SearchKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "search.json"
        self.enterContext(patch.object(search, "SEARCH_KEY_PATH", self.path))

    def test_missing_file_means_no_key(self) -> None:
        self.assertIsNone(search.load_search_key())

    def test_saved_key_is_trimmed(self) -> None:
        _ = self.path.write_text(json.dumps({"key": f"  {KEY}\n", "enabled": False}), encoding="utf-8")
        self.assertEqual(search.load_search_key(), KEY)

    def test_unreadable_or_invalid_files_are_errors(self) -> None:
        _ = self.path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Could not read the saved web search key"):
            _ = search.load_search_key()

        for content in ([KEY], {}, {"key": ""}, {"key": "   ", "enabled": True}, {"key": 5, "enabled": True}):
            with self.subTest(content=content):
                _ = self.path.write_text(json.dumps(content), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Saved web search key is invalid"):
                    _ = search.load_search_key()


if __name__ == "__main__":
    _ = unittest.main()
