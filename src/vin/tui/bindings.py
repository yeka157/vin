"""Keyboard bindings for the terminal workspace."""

from time import monotonic
from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.text import Text

from vin.tui.constants import is_command
from vin.tui.transcript import format_duration

if TYPE_CHECKING:
    from vin.tui.app import VinApp


def create_bindings(app: "VinApp") -> KeyBindings:
    bindings = KeyBindings()

    scroll_keys = Condition(lambda: app.choice_future is None)

    @bindings.add("pageup", filter=scroll_keys)
    def scroll_page_up(event: KeyPressEvent) -> None:
        app.transcript_control.scroll(
            -max(1, app.transcript_control.viewport_height - 1)
        )
        event.app.invalidate()

    @bindings.add("pagedown", filter=scroll_keys)
    def scroll_page_down(event: KeyPressEvent) -> None:
        app.transcript_control.scroll(
            max(1, app.transcript_control.viewport_height - 1)
        )
        event.app.invalidate()

    @bindings.add("c-end", filter=scroll_keys)
    def follow_latest(event: KeyPressEvent) -> None:
        app.transcript_control.scroll_to_bottom()
        event.app.invalidate()

    def completing_session() -> bool:
        if app.choice_future is not None:
            return False
        state = app.session.default_buffer.complete_state
        if state is None or not state.completions:
            return False
        parts = state.original_document.text_before_cursor.split(maxsplit=1)
        return parts[:1] == ["/resume"]

    @bindings.add("tab", filter=Condition(completing_session), eager=True)
    def accept_session_completion(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        state = buffer.complete_state
        assert state is not None
        completion = state.current_completion or state.completions[0]
        buffer.apply_completion(completion)

    @bindings.add("enter", eager=True)
    def submit(event: KeyPressEvent) -> None:
        if app.choice_future is not None:
            if not app.choice_future.done():
                if app.choices is not None:
                    picker_bindings = app.choices.control.get_key_bindings()
                    assert picker_bindings is not None
                    picker_bindings.get_bindings_for_keys((Keys.ControlM,))[-1].handler(
                        event
                    )
                app.choice_future.set_result(
                    app.choices.current_value if app.choices is not None else None
                )
        elif not app.loading_model and (
            not is_command(event.current_buffer.text)
            or (
                app.pending_reply is None
                and app.submitted.empty()
                and (app.queue_paused or not app.queued_prompts)
            )
        ):
            event.current_buffer.validate_and_handle()

    @bindings.add(
        "escape",
        filter=Condition(lambda: app.choice_future is not None),
        eager=True,
    )
    @bindings.add("c-c")
    @bindings.add("c-d", filter=Condition(lambda: app.choice_future is not None))
    def cancel_or_exit(event: KeyPressEvent) -> None:
        if app.choice_future is not None:
            if not app.choice_future.done():
                app.choice_future.cancel()
        elif app.pending_model_load is not None:
            app.pending_model_load.cancel()
            event.app.invalidate()
        elif app.pending_auth is not None:
            if not app.pending_auth.cancelling():
                app.pending_auth.cancel()
            event.app.invalidate()
        elif app.pending_reply is not None:
            if app.tool_status is not None and app.tool_status[1] == "running":
                app.tool_status = (app.tool_status[0], "failed")
            task = app.pending_reply
            app.last_request_duration = (
                monotonic() - app.request_started_at
                if app.request_started_at is not None
                else 0.0
            )
            app.request_started_at = None
            app.pending_reply = None
            app.reply_entry = None
            task.cancel()
            app.transcript.append(
                {
                    "role": "notice",
                    "content": Text(
                        f"Interrupted after {format_duration(app.last_request_duration)}",
                        style="yellow",
                    ),
                }
            )
            event.app.invalidate()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    @bindings.add(
        "escape",
        "enter",
        filter=Condition(lambda: app.choice_future is None),
    )
    def insert_new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings
