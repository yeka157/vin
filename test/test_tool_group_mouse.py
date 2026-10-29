"""Tool-group header geometry, click routing, and composer preservation."""

import asyncio
import unittest
from types import NotImplementedType

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.text import Text

from vin.tui.transcript import TranscriptControl
from vin.types import ToolEntry, TranscriptEntry


class ToolGroupClickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first: ToolEntry = {
            "role": "tool", "name": "read", "detail": "src/first.py",
            "status": "finished", "content": Text("● read src/first.py · finished"),
        }
        self.entries: list[TranscriptEntry] = [self.first]
        self.control = TranscriptControl(lambda: self.entries)

    def render(self, width: int = 100, height: int = 100) -> list[str]:
        content = self.control.create_content(width, height)
        return ["".join(part[1] for part in content.get_line(line))
                for line in range(content.line_count)]

    def mouse(self, y: int, event_type: MouseEventType = MouseEventType.MOUSE_UP, button: MouseButton = MouseButton.LEFT) -> NotImplementedType | None:
        return self.control.mouse_handler(MouseEvent(
            position=Point(x=1, y=y), event_type=event_type,
            button=button, modifiers=frozenset(),
        ))

    def test_left_release_opens_then_closes_without_double_toggling_on_press(self) -> None:
        self.render()
        self.assertIs(self.mouse(0, MouseEventType.MOUSE_DOWN), NotImplemented)
        self.assertFalse(self.first.get("expanded", False))
        self.assertIsNone(self.mouse(0))
        self.assertIs(self.first.get("expanded"), True)
        self.assertIn("src/first.py", "\n".join(self.render()))
        self.assertIsNone(self.mouse(0))
        self.assertIs(self.first.get("expanded"), False)
        self.assertNotIn("src/first.py", "\n".join(self.render()))


    def test_window_routes_clicks_without_changing_composer_draft_or_focus(self) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.current import create_app_session, set_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.layout import BufferControl, HSplit, Layout, Window
        from prompt_toolkit.layout.mouse_handlers import MouseHandlers
        from prompt_toolkit.layout.screen import Screen, WritePosition
        from prompt_toolkit.output import DummyOutput

        async def exercise() -> None:
            with create_app_session(input=DummyInput(), output=DummyOutput()):
                draft = Document("next unsent\ndraft", cursor_position=4)
                buffer = Buffer(document=draft)
                composer = BufferControl(buffer=buffer)
                pane = Window(content=self.control)
                root = HSplit([pane, Window(content=composer, height=3)])
                app: Application[None] = Application(layout=Layout(root, focused_element=composer), mouse_support=True)

                with set_app(app):
                    try:
                        def render_screen() -> tuple[Screen, MouseHandlers, WritePosition]:
                            app.render_counter += 1
                            screen, handlers = Screen(), MouseHandlers()
                            root.write_to_screen(
                                screen, handlers, WritePosition(0, 0, 60, 14), "", True, None,
                            )
                            # Application._redraw refreshes these after painting;
                            # mouse routing uses them to find the active modal area.
                            app.layout.update_parents_relations()
                            bounds = screen.visible_windows_to_write_positions[pane]
                            return screen, handlers, bounds

                        for marker, expected in (("▸", True), ("▾", False)):
                            screen, handlers, bounds = render_screen()
                            y = next(
                                y for y in range(bounds.ypos, bounds.ypos + bounds.height)
                                if screen.data_buffer[y][bounds.xpos].char == marker
                            )
                            position = Point(x=bounds.xpos + 1, y=y)
                            result = handlers.mouse_handlers[y][position.x](MouseEvent(
                                position=position, event_type=MouseEventType.MOUSE_UP,
                                button=MouseButton.LEFT, modifiers=frozenset(),
                            ))
                            self.assertIsNone(result)
                            self.assertIs(self.first.get("expanded"), expected)
                            self.assertEqual(buffer.document, draft)
                            self.assertIs(app.layout.current_control, composer)
                    finally:
                        await app.cancel_and_wait_for_background_tasks()

        asyncio.run(exercise())
