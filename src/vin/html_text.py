import re
from html.parser import HTMLParser
from urllib.parse import urljoin

SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "iframe",
        "nav",
        "header",
        "footer",
        "aside",
    }
)

BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "table",
        "tr",
        "blockquote",
        "pre",
        "hr",
        "figure",
        "figcaption",
        "form",
    }
)

HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


class _TextParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.title: list[str] = []
        self.skip = 0
        self.in_title = False
        self.pre = 0
        self.href: str | None = None
        self.link: list[str] = []

    def emit(self, text: str) -> None:
        if self.href is not None:
            self.link.append(text)
        else:
            self.parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "title":
            self.in_title = True
        if tag == "br":
            self.parts.append("\n")
        if tag in HEADINGS:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        if tag == "li":
            self.parts.append("\n- ")
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "pre":
            self.pre += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not (href.startswith("#") or href.startswith("javascript:")):
                self.href = urljoin(self.base_url, href)
                self.link = []

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.href is not None:
            text = " ".join("".join(self.link).split())
            href = self.href
            self.href = None
            if not text or text == href:
                self.emit(href)
            else:
                self.emit(f"{text} ({href})")
        if tag in HEADINGS or tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "pre":
            self.pre = max(0, self.pre - 1)

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.in_title:
            self.title.append(data)
            return
        if self.pre:
            self.emit(data)
            return
        text = re.sub(r"\s+", " ", data)
        if not self.parts or self.parts[-1].endswith(("\n", "- ", "# ")):
            text = text.lstrip()
        if text:
            self.emit(text)


def html_to_text(source: str, base_url: str) -> tuple[str, str]:
    parser = _TextParser(base_url)
    parser.feed(source)
    parser.close()
    title = " ".join("".join(parser.title).split())
    text = "\n".join(line.rstrip() for line in "".join(parser.parts).split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text
