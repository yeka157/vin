"""Real keyboard events for focused UI fixtures."""

from weakref import ref

from prompt_toolkit import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys


def key_event(app: Application[str], key: Keys) -> KeyPressEvent:
    with set_app(app):
        return KeyPressEvent(ref(app.key_processor), None, [KeyPress(key)], [], False)
