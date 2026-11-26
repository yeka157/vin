"""HTML to readable text for web fetch (#18): structure, links, skipped parts, code and messy input."""

import unittest

from vin.html_text import html_to_text

BASE = "https://docs.example.com/guide/start.html"


def text(source: str, base: str = BASE) -> str:
    return html_to_text(source, base)[1]


class HtmlToTextTests(unittest.TestCase):
    def test_page_structure_becomes_readable_lines(self) -> None:
        title, body = html_to_text(
            """<!doctype html><html><head><title>  Getting
            started  </title><meta charset="utf-8"></head>
            <body><h1>Install</h1><p>Run   the
            installer.</p><h3>Options</h3><ul><li>fast</li><li>safe <b>mode</b></li></ul>
            <p>line one<br>line two</p></body></html>""",
            BASE,
        )
        self.assertEqual(title, "Getting started")
        self.assertEqual(
            body,
            "# Install\n\nRun the installer.\n\n### Options\n\n- fast\n- safe mode\n\nline one\nline two",
        )

    def test_links_show_text_and_absolute_url(self) -> None:
        cases = {
            '<a href="/api">API docs</a>': "API docs (https://docs.example.com/api)",
            '<a href="next.html">Next</a>': "Next (https://docs.example.com/guide/next.html)",
            '<a href="https://pypi.org/">https://pypi.org/</a>': "https://pypi.org/",
            '<a href="https://pypi.org/"><img alt=""></a>': "https://pypi.org/",
            '<a href="#install">Jump</a>': "Jump",
            '<a href="javascript:void(0)">Open</a>': "Open",
            "<a>No target</a>": "No target",
            '<a href="">Empty</a>': "Empty",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(text(f"<p>{source}</p>"), expected)

    def test_link_inside_a_sentence_keeps_its_spaces(self) -> None:
        self.assertEqual(
            text('<p>See the <a href="/faq">  frequently\n asked questions </a> first.</p>'),
            "See the frequently asked questions (https://docs.example.com/faq) first.",
        )

    def test_scripts_styles_and_page_chrome_are_dropped(self) -> None:
        body = text(
            """<header><a href="/">Home</a></header><nav><ul><li>Menu</li></ul></nav>
            <style>p { color: red }</style><script>var x = "<p>fake</p>";</script>
            <main><p>Content</p><aside>Ads</aside><noscript>Enable JS</noscript>
            <svg><title>icon</title><text>chart</text></svg><template><p>hidden</p></template>
            <iframe src="/x"></iframe></main><footer>© 2026</footer>"""
        )
        self.assertEqual(body, "Content")

    def test_nested_skipped_parts_stay_skipped_until_the_outer_one_closes(self) -> None:
        self.assertEqual(text("<nav><aside>a</aside>b</nav><p>c</p>"), "c")

    def test_svg_title_does_not_replace_the_page_title(self) -> None:
        title, body = html_to_text("<title>Real</title><svg><title>Logo</title></svg><p>x</p>", BASE)
        self.assertEqual((title, body), ("Real", "x"))

    def test_unclosed_script_hides_only_what_follows_it(self) -> None:
        self.assertEqual(text("<p>before</p><script>while(1){}"), "before")

    def test_preformatted_code_keeps_its_indentation(self) -> None:
        body = text("<p>Example:</p><pre><code>def f():\n    return  1\n</code></pre><p>After</p>")
        self.assertEqual(body, "Example:\n\ndef f():\n    return  1\n\nAfter")

    def test_entities_are_decoded(self) -> None:
        self.assertEqual(text("<p>a &amp; b &lt;tag&gt; &quot;q&quot; caf&eacute; &#8212;</p>"),
                         'a & b <tag> "q" café —')

    def test_hyphenated_text_split_across_tags_keeps_its_space(self) -> None:
        self.assertEqual(text("<p>well-<b>known</b> fact</p>"), "well-known fact")
        self.assertEqual(text("<p>a -<i>b</i></p>"), "a -b")

    def test_blank_line_runs_collapse_to_one(self) -> None:
        self.assertEqual(text("<p>a</p><div></div><div><p></p></div><hr><p>b</p>"), "a\n\nb")

    def test_empty_plain_and_broken_input(self) -> None:
        self.assertEqual(html_to_text("", BASE), ("", ""))
        self.assertEqual(html_to_text("   \n ", BASE), ("", ""))
        self.assertEqual(text("just words, no tags"), "just words, no tags")
        self.assertEqual(text("<p>open <b>never closed <div>still here"), "open never closed\nstill here")
        self.assertEqual(text("</p></div>stray closers<p"), "stray closers")


if __name__ == "__main__":
    _ = unittest.main()
