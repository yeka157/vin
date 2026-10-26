from queue import SimpleQueue
from threading import Event

class ReplyEvents(SimpleQueue):
    def __init__(self) -> None:
        super().__init__()
        self.tools_used = Event()
