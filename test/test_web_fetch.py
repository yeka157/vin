"""Web fetch (#18): URL and address checks, redirects, content types, paging, limits, errors and cancellation."""

from http.client import HTTPMessage
import io
import json
import socket
from threading import Event
from typing import cast
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from vin import tools
from vin.agent import execute_tool, tool_detail
from vin.tools import FETCH_CHARS, response_tools, tool_cancellation, web_fetch

URL = "https://docs.example.com/guide"
PUBLIC = "93.184.216.34"

type AddressInfo = tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int] | tuple[str, int, int, int]]


def address(host: str) -> AddressInfo:
    if ":" in host:
        return (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (host, 443, 0, 0))
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 443))


class Page:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8", url: str = URL,
                 on_read: Event | None = None) -> None:
        self.stream = io.BytesIO(body)
        self.headers = HTTPMessage()
        self.headers["Content-Type"] = content_type
        self.url = url
        self.reads = 0
        self.taken = 0
        self.on_read = on_read

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.on_read is not None:
            self.on_read.set()
        data = self.stream.read(size)
        self.taken += len(data)
        return data

    def __enter__(self) -> "Page":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stream.close()


def http_error(code: int, reason: str) -> HTTPError:
    return HTTPError(URL, code, reason, HTTPMessage(), io.BytesIO(b""))


class FetchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dns = self.enterContext(patch.object(tools.socket, "getaddrinfo", return_value=[address(PUBLIC)]))
        self.open = Mock()
        self.build = self.enterContext(patch.object(tools, "build_opener", return_value=Mock(open=self.open)))

    def serve(self, page: object) -> None:
        if isinstance(page, BaseException):
            self.open.side_effect = page
        else:
            self.open.return_value = page

    def fetch(self, page: object, url: str = URL, offset: int = 0, limit: int = FETCH_CHARS) -> str:
        self.serve(page)
        return web_fetch(url, offset, limit)

    def failure(self, page: object, url: str = URL, offset: int = 0, limit: int = FETCH_CHARS) -> str:
        self.serve(page)
        with self.assertRaises((RuntimeError, ValueError)) as raised:
            _ = web_fetch(url, offset, limit)
        self.assertIsNone(raised.exception.__cause__)
        return str(raised.exception)


class WebFetchTests(FetchTestCase):
    def test_html_page_comes_back_as_text_with_title_and_url(self) -> None:
        output = self.fetch(Page(b"<title>Guide</title><nav>Menu</nav><h1>Start</h1><p>Read <a href='/api'>the API</a>.</p>"),
                            url=f"  {URL}  ")

        self.assertEqual(output, "Title: Guide\nURL: https://docs.example.com/guide\n\n"
                                 "# Start\n\nRead the API (https://docs.example.com/api).")
        self.build.assert_called_once_with(tools._CheckedRedirects)  # pyright: ignore[reportPrivateUsage]
        request = cast(Request, self.open.call_args.args[0])
        self.assertEqual(request.full_url, URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("User-agent"), "Vin")
        self.assertIn("text/html", str(request.get_header("Accept")))
        self.assertEqual(self.open.call_args.kwargs["timeout"], tools.FETCH_TIMEOUT)

    def test_final_url_after_redirects_is_shown_and_used_for_links(self) -> None:
        output = self.fetch(Page(b"<a href='next'>Next</a>", url="https://moved.example.org/docs/"))
        self.assertEqual(output, "URL: https://moved.example.org/docs/\n\nNext (https://moved.example.org/docs/next)")

    def test_text_and_json_come_back_unchanged(self) -> None:
        cases = {
            "text/plain": b"  line one\n\n\n<b>not html</b>\n",
            "text/markdown; charset=utf-8": b"# Title\n\n- item",
            "application/json": json.dumps({"version": "3.0.53"}).encode(),
            "application/vnd.api+json": b'{"data": []}',
            "text/csv": b"a,b\n1,2",
        }
        for content_type, body in cases.items():
            with self.subTest(content_type=content_type):
                self.assertEqual(self.fetch(Page(body, content_type)), f"URL: {URL}\n\n{body.decode()}")

    def test_binary_types_are_refused_before_downloading(self) -> None:
        for content_type in ("application/pdf", "image/png", "application/octet-stream", "video/mp4"):
            with self.subTest(content_type=content_type):
                page = Page(b"%PDF-1.7", content_type)
                self.assertEqual(self.failure(page),
                                 f"Fetch failed: content type {content_type} is not supported.")
                self.assertEqual(page.reads, 0)

    def test_long_pages_are_read_in_parts(self) -> None:
        text = "".join(f"{index:05d}\n" for index in range(10_000))
        body = text.encode()

        first = self.fetch(Page(body, "text/plain"))
        self.assertEqual(first, f"URL: {URL}\n\n{text[:FETCH_CHARS]}\n\n"
                                f"[Characters 0-19,000 of 60,000. Call web_fetch with offset=19000 to read more.]")
        self.assertLessEqual(len(first), tools.MAX_OUTPUT_CHARS)

        middle = self.fetch(Page(body, "text/plain"), offset=19_000, limit=100)
        self.assertTrue(middle.endswith(f"{text[19_000:19_100]}\n\n"
                                        "[Characters 19,000-19,100 of 60,000. Call web_fetch with offset=19100 to read more.]"))

        last = self.fetch(Page(body, "text/plain"), offset=59_990)
        self.assertEqual(last, f"URL: {URL}\n\n{text[59_990:]}")

    def test_offset_and_limit_out_of_range(self) -> None:
        self.assertEqual(self.failure(Page(b"short", "text/plain"), offset=5),
                         "Offset 5 is past the end of the page (5 characters).")
        for offset, limit, message in ((-1, FETCH_CHARS, "Offset must be 0 or more."),
                                       (0, 0, f"Limit must be between 1 and {FETCH_CHARS}."),
                                       (0, FETCH_CHARS + 1, f"Limit must be between 1 and {FETCH_CHARS}.")):
            with self.subTest(offset=offset, limit=limit):
                self.open.reset_mock()
                self.assertEqual(self.failure(Page(b"x"), offset=offset, limit=limit), message)
                self.open.assert_not_called()

    def test_download_stops_at_five_megabytes(self) -> None:
        exact = self.fetch(Page(b"a" * tools.MAX_FETCH_BYTES, "text/plain"))
        self.assertNotIn("larger than 5 MB", exact)

        page = Page(b"a" * (tools.MAX_FETCH_BYTES + 10), "text/plain")
        output = self.fetch(page)
        self.assertTrue(output.endswith("\n\n[The page is larger than 5 MB; only the first 5 MB was read.]"))
        self.assertIn("of 5,000,000.", output)
        self.assertEqual(page.taken, tools.MAX_FETCH_BYTES + 1)

    def test_charsets_are_honored_and_unknown_ones_fall_back_to_utf8(self) -> None:
        latin = self.fetch(Page("café".encode("latin-1"), "text/plain; charset=iso-8859-1"))
        self.assertTrue(latin.endswith("café"))
        made_up = self.fetch(Page("naïve".encode(), "text/plain; charset=x-made-up"))
        self.assertTrue(made_up.endswith("naïve"))
        broken = self.fetch(Page(b"ok \xff\xfe end", "text/plain"))
        self.assertTrue(broken.endswith("ok �� end"))

    def test_page_without_readable_text_says_it_may_need_javascript(self) -> None:
        for body in (b"<title>App</title><div id='root'></div><script>render()</script>", b"", b"   \n"):
            with self.subTest(body=body):
                output = self.fetch(Page(body))
                self.assertTrue(output.endswith("The page has no readable text. It may need JavaScript to render."))

    def test_http_and_network_errors_have_their_own_messages(self) -> None:
        cases: list[tuple[BaseException, str]] = [
            (http_error(404, "Not Found"), "Fetch failed: HTTP 404 Not Found."),
            (http_error(500, "Internal Server Error"), "Fetch failed: HTTP 500 Internal Server Error."),
            (http_error(302, "Found"), "Fetch failed: too many redirects."),
            (URLError("[Errno 61] Connection refused"), "Fetch failed: [Errno 61] Connection refused."),
            (TimeoutError("timed out"), "Fetch failed: timed out."),
            (ConnectionResetError("reset by peer"), "Fetch failed: reset by peer."),
        ]
        for error, message in cases:
            with self.subTest(message=message):
                self.assertEqual(self.failure(error), message)

    def test_cancelled_before_the_request_sends_nothing(self) -> None:
        cancel = Event()
        cancel.set()
        with tool_cancellation(cancel):
            self.assertRegex(self.failure(Page(b"x")), "cancelled")
        self.open.assert_not_called()

    def test_cancelled_during_the_download_stops_reading(self) -> None:
        cancel = Event()
        page = Page(b"a" * (tools.FETCH_CHUNK * 10), "text/plain", on_read=cancel)
        with tool_cancellation(cancel):
            self.assertRegex(self.failure(page), "cancelled")
        self.assertEqual(page.reads, 1)


class AddressTests(FetchTestCase):
    def test_only_http_and_https_urls_with_a_host_and_valid_port(self) -> None:
        for url in ("ftp://example.com/file", "file:///etc/passwd", "example.com", "http://", "https:///path",
                    "javascript:alert(1)", "http://example.com:99999/", "http://example.com:port/", ""):
            with self.subTest(url=url):
                self.assertEqual(self.failure(Page(b"x"), url=url), "URL must start with http:// or https://.")
        self.dns.assert_not_called()
        self.open.assert_not_called()

    def test_local_and_private_addresses_are_refused(self) -> None:
        for host in ("127.0.0.1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254", "100.64.0.1",
                     "0.0.0.0", "::1", "fe80::1%en0", "fc00::1", "::ffff:127.0.0.1"):
            with self.subTest(host=host):
                self.dns.return_value = [address(host)]
                self.assertEqual(self.failure(Page(b"x"), url="http://internal.example/"),
                                 "web_fetch does not fetch local or private network addresses.")
        self.open.assert_not_called()

    def test_one_private_answer_among_public_ones_is_enough_to_refuse(self) -> None:
        self.dns.return_value = [address(PUBLIC), address("10.1.2.3")]
        self.assertEqual(self.failure(Page(b"x")), "web_fetch does not fetch local or private network addresses.")

    def test_dns_is_checked_with_the_url_host_and_port(self) -> None:
        _ = self.fetch(Page(b"x", "text/plain"), url="https://Docs.Example.com:8443/x")
        self.assertEqual(self.dns.call_args.args[:2], ("docs.example.com", 8443))

    def test_unresolvable_host(self) -> None:
        self.dns.side_effect = socket.gaierror(8, "nodename nor servname provided")
        self.assertEqual(self.failure(Page(b"x"), url="https://no-such-host.invalid/"),
                         "Fetch failed: could not resolve no-such-host.invalid.")

    def test_redirects_are_checked_and_limited(self) -> None:
        handler = tools._CheckedRedirects()  # pyright: ignore[reportPrivateUsage]
        self.assertEqual(handler.max_redirections, 5)
        original = Request(URL)

        followed = handler.redirect_request(original, io.BytesIO(), 302, "Found", HTTPMessage(), "https://docs.example.com/new")
        assert followed is not None
        self.assertEqual(followed.full_url, "https://docs.example.com/new")

        self.dns.return_value = [address("127.0.0.1")]
        with self.assertRaisesRegex(RuntimeError, "does not fetch local or private"):
            _ = handler.redirect_request(original, io.BytesIO(), 302, "Found", HTTPMessage(), "http://localhost/admin")
        with self.assertRaisesRegex(ValueError, "must start with http"):
            _ = handler.redirect_request(original, io.BytesIO(), 302, "Found", HTTPMessage(), "file:///etc/passwd")


class FetchToolListTests(FetchTestCase):
    def test_web_fetch_is_offered_only_when_fetch_is_on(self) -> None:
        self.assertNotIn("web_fetch", [tool["name"] for tool in response_tools(task=True, search=True)])
        (tool,) = [tool for tool in response_tools(fetch=True) if tool["name"] == "web_fetch"]
        description = tool.get("description") or ""
        self.assertTrue(description.startswith("Fetch a web page"))
        self.assertIn("untrusted", description)

    def test_model_sees_the_error_and_the_row_shows_the_url(self) -> None:
        self.serve(http_error(404, "Not Found"))
        output = execute_tool("web_fetch", json.dumps({"url": URL}), Event())
        self.assertEqual(output, "Error (RuntimeError): Fetch failed: HTTP 404 Not Found.")
        self.assertEqual(tool_detail("web_fetch", json.dumps({"url": URL})), URL)


if __name__ == "__main__":
    _ = unittest.main()
