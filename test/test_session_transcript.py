"""JSON-compatible transcript content without losing visible text or styling."""

from io import StringIO
import json
from copy import deepcopy
import unittest

from rich.console import Console
from rich.text import Text


class SessionContentDecodingTests(unittest.TestCase):
    def decode(self, content):
        from vin.sessions import transcript

        decoder = getattr(transcript, "decode_content", None)
        self.assertTrue(callable(decoder), "Add decode_content(content) in sessions/transcript.py.")
        return decoder(content)


    def test_json_round_trip_restores_rich_content_and_its_rendered_appearance(self):
        from vin.sessions.transcript import encode_content

        styled = Text("● edit file.py\n界", style="dim")
        styled.stylize("bold cyan", 7, 14)
        styled.stylize("red", 9, 13)

        def render(content):
            output = StringIO()
            console = Console(file=output, width=60, force_terminal=True, color_system="truecolor")
            console.print(content)
            return output.getvalue()

        for source in (styled, Text("")):
            with self.subTest(text=source.plain):
                encoded = json.loads(json.dumps(encode_content(source)))
                before = json.dumps(encoded, sort_keys=True)

                restored = self.decode(encoded)

                self.assertIsInstance(restored, Text)
                self.assertEqual(restored.plain, source.plain)
                self.assertEqual(restored.style, source.style)
                self.assertEqual(restored.spans, source.spans)
                self.assertEqual(render(restored), render(source))
                self.assertEqual(json.dumps(encoded, sort_keys=True), before)


class SessionTranscriptDecodingTests(unittest.TestCase):
    def decode(self, entries):
        from vin.sessions import transcript

        decoder = getattr(transcript, "decode_transcript", None)
        self.assertTrue(callable(decoder), "Add decode_transcript(entries) in sessions/transcript.py.")
        return decoder(entries)

    def test_whole_transcript_json_round_trip_restores_order_diffs_and_rendering(self):
        from prompt_toolkit.formatted_text import to_formatted_text
        from vin.sessions.transcript import encode_transcript
        from vin.tui.transcript import render_transcript

        source = [
            {"role": "user", "content": "Review\n[red]literal[/red]"},
            {"role": "assistant", "content": "Checking **the file**.", "intermediate": True},
            {"role": "tool", "name": "read", "detail": "same.py", "status": "finished",
             "content": Text("● read same.py · finished", style="dim"), "expanded": True},
            {"role": "tool", "name": "edit", "detail": "same.py", "status": "finished",
             "content": Text("● edit same.py · finished", style="dim"),
             "diff": "--- f\n+++ f\n@@ -1 +1 @@\n-old\n+new\n"},
            {"role": "assistant", "content": "**Done.**", "streaming": False},
            {"role": "notice", "content": Text("Worked for 2s · done 09:30 PM", style="dim")},
        ]
        saved = json.loads(json.dumps(encode_transcript(source)))
        original_saved = deepcopy(saved)

        restored = self.decode(saved)

        self.assertIsNot(restored, saved)
        self.assertEqual([entry["role"] for entry in restored],
                         ["user", "assistant", "tool", "tool", "assistant", "notice"])
        self.assertEqual(restored[3]["diff"], source[3]["diff"])
        for width in (24, 90):
            with self.subTest(width=width):
                self.assertEqual(to_formatted_text(render_transcript(restored, width, 100)),
                                 to_formatted_text(render_transcript(source, width, 100)))
        self.assertEqual(saved, original_saved)
        for entry, saved_entry in zip(restored, saved):
            self.assertIsNot(entry, saved_entry)
        empty = []
        restored_empty = self.decode(empty)
        self.assertEqual(restored_empty, [])
        self.assertIsNot(restored_empty, empty)
