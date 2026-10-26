from rich.text import Span, Text


def encode_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, Text):
        return {
            "type": "rich_text",
            "text": content.plain,
            "style": str(content.style),
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "style": str(span.style),
                }
                for span in content.spans
            ],
        }

    raise ValueError("Unsupported transcript content.")


def decode_content(content):
    if isinstance(content, str):
        return content

    if not isinstance(content, dict) or content.get("type") != "rich_text":
        raise ValueError("Invalid saved transcript content.")

    text = content.get("text")
    style = content.get("style")
    spans = content.get("spans")

    if (
        not isinstance(text, str)
        or not isinstance(style, str)
        or not isinstance(spans, list)
    ):
        raise ValueError("Invalid saved text fields.")

    restored = Text(text, style=style)

    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("Invalid saved text span.")

        start = span.get("start")
        end = span.get("end")
        span_style = span.get("style")

        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(span_style, str)
            or not 0 <= start <= end <= len(text)
        ):
            raise ValueError("Invalid saved text span.")

        restored.spans.append(Span(start, end, span_style))

    return restored


def encode_entry(entry):
    if (
        not isinstance(entry, dict)
        or entry.get("role") not in ("user", "assistant", "tool", "notice")
        or "content" not in entry
    ):
        raise ValueError("Invalid transcript entry.")

    encoded = entry.copy()
    encoded["content"] = encode_content(entry["content"])
    return encoded


def decode_entry(entry):
    if (
        not isinstance(entry, dict)
        or entry.get("role") not in ("user", "assistant", "tool", "notice")
        or "content" not in entry
    ):
        raise ValueError("Invalid transcript entry.")

    decoded = entry.copy()
    decoded["content"] = decode_content(entry["content"])
    return decoded


def encode_transcript(entries):
    if not isinstance(entries, list):
        raise ValueError("Invalid transcript.")

    encoded = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid transcript entry.")

        if entry.get("persist") is False or entry.get("transient") is True:
            continue

        encoded.append(encode_entry(entry))

    return encoded


def decode_transcript(entries):
    if not isinstance(entries, list):
        raise ValueError("Invalid saved transcript.")

    return [decode_entry(entry) for entry in entries]
