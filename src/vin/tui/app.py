"""Application state, conversation flow, commands, and account/model setup."""

import asyncio
from collections import deque
from queue import Empty
from threading import Event, Thread
from time import monotonic
from datetime import datetime
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Container, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.widgets import RadioList
from rich.markdown import Markdown
from rich.console import RenderableType
from rich.padding import Padding
from rich.text import Text

from vin.auth.chatgpt import (
    TokenExchangeError,
    login_with_chatgpt,
    logout_chatgpt,
    refresh_credentials,
)
from vin.auth.credentials import load_active_session
from vin.provider.chatgpt import list_chatgpt_models, stream_chatgpt_response
from vin.provider.model_cache import (
    load_cached_chatgpt_models,
    save_cached_chatgpt_models,
)
from vin.provider.preferences import load_model_preference, save_model_preference
from vin.sessions.records import new_session
from vin.tools import LAUNCH_DIR
from vin.tui import conversations
from vin.tui.constants import PROVIDER_LABELS, is_command
from vin.tui.layout import build_workspace
from vin.tui.transcript import TranscriptControl, format_duration, render_transcript
from vin.provider.events import ReplyEvents
from vin.types import (
    AccountRecord, ActiveTurn, AssistantEntry, Credentials, Registration, History, ModelMetadata,
    ModelPreference, SessionRecord, ToolEntry, TranscriptEntry,
)


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and bool(task.cancelling())


def create_choice_menu(
    message: str,
    options: Sequence[tuple[str, str]],
    default: str | None = None,
) -> tuple[RadioList[str], HSplit]:
    if not options:
        raise RuntimeError("No choices are available.")

    choices: RadioList[str] = RadioList(
        values=options,
        default=default,
        select_on_focus=True,
        show_numbers=True,
        open_character="",
        select_character="❯",
        close_character="",
        show_cursor=False,
        show_scrollbar=False,
        container_style="class:completion-menu.completion",
        default_style="",
        selected_style="fg:#ffffff bg:#303b59 bold",
        checked_style="class:completion-menu.completion.current",
        number_style="",
    )

    container = HSplit(
        [
            Window(FormattedTextControl(f" {message}"), height=1),
            choices,
        ],
        height=Dimension(max=6),
    )

    return choices, container


def choose_login_method() -> str:
    bindings = KeyBindings()

    @bindings.add("c-d")
    def exit_menu(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    return choice(
        message="Sign in to Vin",
        options=[("chatgpt", "Sign in with ChatGPT")],
        symbol="❯",
        key_bindings=bindings,
    )


def choose_chatgpt_model(models: Sequence[ModelMetadata]) -> str:
    if not models:
        raise RuntimeError("No ChatGPT models are available for this account.")

    bindings = KeyBindings()

    @bindings.add("c-d")
    def exit_menu(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    return choice(
        message="Choose a ChatGPT model",
        options=[(model["slug"], model["display_name"]) for model in models],
        symbol="❯",
        key_bindings=bindings,
    )


def model_display_name(models: Sequence[ModelMetadata], slug: str) -> str:
    return next(
        (model["display_name"] for model in models if model["slug"] == slug),
        slug,
    )


def restore_chatgpt_model(
    transcript: list[TranscriptEntry], models: Sequence[ModelMetadata]
) -> ModelPreference:
    preference: ModelPreference | None = get_chatgpt_model_preference(transcript, models)
    if preference is None:
        selected_model = choose_chatgpt_model(models)
        preference = {"provider": "chatgpt", "model": selected_model}
        save_model_preference(preference["provider"], preference["model"])
    return preference


def get_chatgpt_model_preference(
    transcript: list[TranscriptEntry], models: Sequence[ModelMetadata]
) -> ModelPreference | None:
    preference = load_model_preference()
    if preference is None:
        return None

    if any(model["slug"] == preference["model"] for model in models):
        return preference

    transcript.append(
        {
            "role": "notice",
            "content": Text(
                "Current model is now unavailable. Please choose another model.",
                style="yellow",
            ),
        }
    )
    return None


class VinApp:
    session: PromptSession[str]
    app: Application[str]
    input_window: Window
    header: Markdown

    def __init__(self) -> None:
        self.credentials: AccountRecord | None = None
        self.models: list[ModelMetadata] = []
        self.selected_provider: Literal["chatgpt"] | None = None
        self.selected_model: str | None = None
        self.messages: History = []
        self.transcript: list[TranscriptEntry] = []
        self.transcript_control = TranscriptControl(self.get_transcript_entries)
        self.pending_reply: asyncio.Task[Literal["error", "rate_limit"] | None] | None = None
        self.loading_model = False
        self.submitted: asyncio.Queue[str | None] = asyncio.Queue()
        self.choice_menu: Container | None = None
        self.choices: RadioList[str] | None = None
        self.choice_future: asyncio.Future[str | None] | None = None
        self.pending_model_load: asyncio.Task[list[ModelMetadata]] | None = None
        self.auth_status: str | None = None
        self.pending_auth: asyncio.Task[None] | None = None
        self.reply_entry: AssistantEntry | None = None
        self.queued_prompts: deque[str] = deque()
        self.queue_paused = False
        self.request_started_at: float | None = None
        self.last_request_duration: float | None = None
        self.tool_status: tuple[str, Literal["running", "finished", "failed"]] | None = None
        self._credential_refresh: asyncio.Task[Credentials] | None = None
        self.project_dir: Path = LAUNCH_DIR
        self.session_record: SessionRecord | None = None
        self.saved_session_record: SessionRecord | None = None
        self.active_turn: ActiveTurn | None = None
        self.last_session_checkpoint: float | None = None
        self.session_save_error: str | None = None

    def checkpoint_session(self, *, force: bool = False) -> bool:
        return conversations.checkpoint_session(self, force=force)

    def get_transcript_entries(self) -> list[TranscriptEntry]:
        entries = list(self.transcript)
        if self.request_started_at is not None:
            now = monotonic()
            elapsed = now - self.request_started_at
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            frame = frames[int(now * 10) % len(frames)]
            entries.append(
                {
                    "role": "notice",
                    "transient": True,
                    "content": Text(
                        f"{frame} Thinking… ({format_duration(elapsed)})", style="dim"
                    ),
                }
            )
        return entries

    async def prepare_request_credentials(self) -> AccountRecord | None:
        if not self.credentials or not self.credentials.get("access_token"):
            return self.credentials

        assert "access_token" in self.credentials
        if self._credential_refresh is None:
            original = self.credentials

            async def refresh() -> Credentials:
                updated = await asyncio.to_thread(refresh_credentials, original)
                if self.credentials is not original:
                    raise RuntimeError("ChatGPT session changed during token refresh.")
                self.credentials = updated
                return updated

            task = asyncio.create_task(refresh())
            self._credential_refresh = task

            def finish(done: asyncio.Task[Credentials]) -> None:
                # A cancelled waiter must not cancel a rotating-token exchange.
                if not done.cancelled():
                    done.exception()
                if self._credential_refresh is done:
                    self._credential_refresh = None

            task.add_done_callback(finish)

        return await asyncio.shield(self._credential_refresh)

    async def choose_inline(
        self,
        message: str,
        options: Sequence[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        if self.choice_future is not None:
            raise RuntimeError("A choice is already open")

        self.choices, self.choice_menu = create_choice_menu(message, options, default)
        self.choice_future = asyncio.get_running_loop().create_future()

        try:
            self.session.layout.focus(self.choices)
            self.session.app.invalidate()
            selected = await self.choice_future
            assert selected is not None
            return selected
        finally:
            self.choices = None
            self.choice_menu = None
            self.choice_future = None
            self.session.layout.focus(self.input_window)
            self.session.app.invalidate()

    async def show_panel(self, content: RenderableType) -> None:
        if self.choice_future is not None:
            raise RuntimeError("A panel is already open")

        self.choices = None
        self.choice_menu = Window(
            FormattedTextControl(
                lambda: render_transcript(
                    [
                        {
                            "role": "notice",
                            "content": Padding(
                                content,
                                (
                                    0,
                                    0,
                                    0,
                                    2,
                                ),
                            ),
                        }
                    ],
                    self.app.output.get_size().columns,
                    self.app.output.get_size().rows,
                ),
                focusable=True,
                show_cursor=False,
            ),
            wrap_lines=True,
            dont_extend_height=True,
        )
        self.choice_future = asyncio.get_running_loop().create_future()

        try:
            self.session.layout.focus(self.choice_menu)
            self.app.invalidate()
            await self.choice_future
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
        finally:
            self.choice_menu = None
            self.choice_future = None
            self.session.layout.focus(self.input_window)
            self.app.invalidate()

    def accept_input(self, buffer: Buffer) -> bool:
        text = buffer.text
        if is_command(text):
            self.submitted.put_nowait(text)
        elif text.strip() or (self.queue_paused and self.queued_prompts):
            if text.strip():
                self.queued_prompts.append(text)
            self.queue_paused = False
            self.submitted.put_nowait(None)
        return False

    async def stream_reply(self, text: str) -> Literal["error", "rate_limit"] | None:
        events = ReplyEvents()
        cancel_event = Event()
        entry: AssistantEntry = {"role": "assistant", "content": "", "streaming": True}
        request = [message.copy() for message in self.messages]
        request.append({"role": "user", "content": text})
        self.reply_entry = entry
        self.request_started_at = monotonic()
        self.last_request_duration = None
        completed = False
        text_dirty = False
        tool_entry: ToolEntry | None = None
        self.tool_status = None

        try:
            try:
                credentials = await self.prepare_request_credentials()
            except Exception as error:
                message = "Could not refresh ChatGPT sign-in. Please try again."
                if isinstance(error, TokenExchangeError) and error.requires_login:
                    message = "ChatGPT sign-in has expired. Sign in again with /login."
                self.transcript.append(
                    {"role": "notice", "content": Text(message, style="red")}
                )
                return "error"

            Thread(
                target=stream_chatgpt_response,
                args=(
                    (credentials or {}).copy(),
                    self.selected_model,
                    request,
                    events,
                    cancel_event,
                ),
                daemon=True,
            ).start()

            while True:
                try:
                    event = events.get_nowait()
                    kind = event[0]
                except Empty:
                    if text_dirty and self.checkpoint_session():
                        text_dirty = False
                    await asyncio.sleep(0.03)
                    continue

                if event[0] == "text":
                    if not entry["content"]:
                        self.transcript.append(entry)
                    entry["content"] += event[1]
                    text_dirty = True
                elif event[0] == "tool_diff":
                    if tool_entry is not None and tool_entry["name"] == "edit":
                        tool_entry["diff"] = event[1]
                elif (
                    event[0] == "tool_started"
                    or event[0] == "tool_finished"
                    or event[0] == "tool_failed"
                ):
                    value = event[1]
                    name = value["name"] if isinstance(value, dict) else value
                    detail = value.get("detail", "") if isinstance(value, dict) else ""
                    state: Literal["running", "finished", "failed"] = (
                        "running" if kind == "tool_started"
                        else "finished" if kind == "tool_finished"
                        else "failed"
                    )
                    self.tool_status = (name, state)
                    if kind == "tool_started":
                        entry["intermediate"] = True
                        entry["streaming"] = False
                        entry = {"role": "assistant", "content": "", "streaming": True}
                        self.reply_entry = entry
                        label = f"{name} {detail}" if detail else name
                        tool_entry = {
                            "role": "tool",
                            "name": name,
                            "detail": detail,
                            "label": label,
                            "status": "running",
                            "content": Text(f"● {label} · running", style="dim"),
                        }
                        self.transcript.append(tool_entry)
                    elif tool_entry is not None:
                        assert "label" in tool_entry
                        tool_entry["status"] = state
                        tool_entry["content"] = Text(
                            f"● {tool_entry['label']} · {state}",
                            style="red" if kind == "tool_failed" else "dim",
                        )
                        tool_entry = None
                elif event[0] == "complete":
                    value = event[1]
                    entry["streaming"] = False
                    if value is None:
                        self.messages.extend(
                            [
                                {"role": "user", "content": text},
                                {"role": "assistant", "content": entry["content"]},
                            ]
                        )
                    else:
                        self.messages[:] = value
                    completed = True
                    return
                elif event[0] == "error" or event[0] == "rate_limit":
                    self.transcript.append(
                        {"role": "notice", "content": Text(event[1], style="red")}
                    )
                    return event[0]

                if kind in (
                    "tool_started",
                    "tool_finished",
                    "tool_failed",
                    "tool_diff",
                ):
                    if self.active_turn is not None:
                        self.active_turn["tools_used"] = True
                    if self.checkpoint_session(force=True):
                        text_dirty = False
                elif kind == "text":
                    if self.checkpoint_session():
                        text_dirty = False

                self.session.app.invalidate()
                await asyncio.sleep(0)
        finally:
            cancel_event.set()
            if not completed and tool_entry is not None:
                assert "label" in tool_entry
                interrupted = _task_is_cancelling()
                interrupted_state: Literal["interrupted", "failed"] = (
                    "interrupted" if interrupted else "failed"
                )
                tool_entry["status"] = interrupted_state
                tool_entry["content"] = Text(
                    f"● {tool_entry['label']} · {interrupted_state}",
                    style="yellow" if interrupted else "red",
                )
            if not completed and events.tools_used.is_set():
                self.messages.extend(
                    [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": (
                                "Turn interrupted after tool use. "
                                "Changes may already exist; read relevant "
                                "files before retrying."
                            ),
                        },
                    ]
                )
            if self.pending_reply is asyncio.current_task():
                if (
                    not completed
                    and self.tool_status is not None
                    and self.tool_status[1] == "running"
                ):
                    self.tool_status = (self.tool_status[0], "failed")
                assert self.request_started_at is not None
                self.last_request_duration = monotonic() - self.request_started_at
                if completed:
                    finished_at = datetime.now().astimezone().strftime("%I:%M %p")
                    self.transcript.append(
                        {
                            "role": "notice",
                            "content": Text(
                                f"* Worked for {format_duration(self.last_request_duration)}"
                                f" · done {finished_at}",
                                style="dim",
                            ),
                        }
                    )
                else:
                    interrupted = _task_is_cancelling()
                    label = "Interrupted" if interrupted else "Failed"
                    self.transcript.append(
                        {
                            "role": "notice",
                            "content": Text(
                                f"{label} after {format_duration(self.last_request_duration)}",
                                style="yellow" if interrupted else "red",
                            ),
                        }
                    )
                self.active_turn = None
                self.checkpoint_session(force=True)
                self.request_started_at = None
                self.pending_reply = None
                self.reply_entry = None
            self.session.app.invalidate()

    async def resume_conversation(self, query: str | None = None) -> str | None:
        return await conversations.resume_conversation(self, query)

    def rename_conversation(self, name: object) -> str | None:
        return conversations.rename_conversation(self, name)

    async def change_model(self) -> str | None:
        if not self.credentials or not self.credentials.get("access_token"):
            return "Sign in with /login before choosing a model."

        if self.selected_provider is None:
            raise RuntimeError("Model setup is incomplete. Use /login to try again.")

        models = self.models
        provider_name = PROVIDER_LABELS[self.selected_provider]

        if not models:
            self.loading_model = True
            task = asyncio.create_task(self.discover_chatgpt_models())
            self.pending_model_load = task
            self.app.invalidate()
            try:
                models = await task
                self.models = models
            except asyncio.CancelledError:
                if _task_is_cancelling():
                    raise
                return
            finally:
                self.pending_model_load = None
                self.loading_model = False
                self.app.invalidate()

        if not models:
            raise RuntimeError(
                f"No {provider_name} models are available for this account."
            )

        try:
            new_model = await self.choose_inline(
                f"Choose a {provider_name} model",
                [(model["slug"], model["display_name"]) for model in models],
                default=self.selected_model,
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return

        if new_model == self.selected_model:
            return

        save_model_preference(self.selected_provider, new_model)
        self.selected_model = new_model
        model_name = model_display_name(self.models, self.selected_model)
        return f"Model: {provider_name} · {model_name}"

    async def run_auth_command(self, text: str) -> None:
        self.loading_model = True
        task = asyncio.create_task(
            self.initialize_session(sign_in=True)
            if text == "/login"
            else self.sign_out()
        )
        self.pending_auth = task
        self.app.invalidate()

        try:
            await task
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
        finally:
            self.pending_auth = None
            self.auth_status = None
            self.loading_model = False
            self.app.invalidate()

    async def sign_out(self, *, announce: bool = True) -> None:
        credentials: AccountRecord | None = None
        notices: list[str] = []
        self.auth_status = "Signing out…"
        self.app.invalidate()

        async def finish_sign_out() -> Registration | None:
            nonlocal credentials
            if self._credential_refresh is not None:
                try:
                    await asyncio.shield(self._credential_refresh)
                except Exception:
                    pass
            credentials = (
                self.credentials.copy() if self.credentials is not None else None
            )
            return await asyncio.to_thread(
                logout_chatgpt,
                credentials,
                notify=notices.append,
            )

        task = asyncio.create_task(finish_sign_out())
        cancelled = False
        try:
            try:
                registration = await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                self.auth_status = "Finishing sign-out…"
                self.app.invalidate()
                registration = await asyncio.shield(task)
            self.credentials = registration
        except (RuntimeError, OSError, ValueError):
            if credentials is not None and not credentials.get("access_token"):
                self.credentials = credentials
            raise
        finally:
            if not self.credentials or not self.credentials.get("access_token"):
                self.models = []
                self.selected_provider = None
                self.selected_model = None
            self.auth_status = None
            self.app.invalidate()

        for message in notices:
            self.transcript.append(
                {
                    "role": "notice",
                    "content": Text(message),
                }
            )

        if announce:
            self.transcript.append(
                {
                    "role": "notice",
                    "content": "Signed out of Vin.",
                }
            )

        if cancelled:
            raise asyncio.CancelledError

    async def chat_loop(self) -> str:

        self.session.message = "❯ "
        self.session.default_buffer.accept_handler = self.accept_input

        async def receive_messages() -> None:
            if self.loading_model:
                try:
                    await self.initialize_session()
                except asyncio.CancelledError:
                    if _task_is_cancelling():
                        raise
                    self.app.exit(result="/quit")
                    return
                except (RuntimeError, OSError, ValueError) as error:
                    self.transcript.append(
                        {
                            "role": "notice",
                            "content": Text(
                                str(error),
                                style="red",
                            ),
                        }
                    )
                finally:
                    self.loading_model = False
                    self.app.invalidate()
            while not self.app.is_done:
                if self.queued_prompts and not self.queue_paused:
                    text = self.queued_prompts.popleft()
                else:
                    text = await self.submitted.get()
                    if text is None:
                        continue
                if not text.strip():
                    continue
                if text in (
                    "/exit",
                    "/quit",
                ):
                    self.app.exit(result=text)
                    return
                if is_command(text):
                    parts = text.split(maxsplit=1)
                    try:
                        if text in ("/login", "/logout"):
                            await self.run_auth_command(text)
                        elif parts[0] == "/resume":
                            query = parts[1] if len(parts) > 1 else None
                            outcome = await self.resume_conversation(query)
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        elif parts[0] == "/rename":
                            name = parts[1] if len(parts) > 1 else ""
                            outcome = self.rename_conversation(name)
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        elif text == "/model":
                            try:
                                outcome = await self.change_model()
                            except (RuntimeError, OSError, ValueError) as error:
                                outcome = Text(str(error), style="red")
                            self.transcript.append({"role": "user", "content": text})
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        else:
                            content = self.handle_command(text)
                            if content is not None:
                                await self.show_panel(content)
                    except (RuntimeError, OSError, ValueError) as error:
                        self.transcript.append(
                            {
                                "role": "notice",
                                "content": Text(
                                    str(error),
                                    style="red",
                                ),
                            }
                        )
                    self.app.invalidate()
                    continue

                self.transcript.append({"role": "user", "content": text})
                if self.session_record is None:
                    self.session_record = new_session(self.project_dir, text)

                self.active_turn = {"prompt": text, "tools_used": False}
                self.checkpoint_session(force=True)
                self.reply_entry = None
                self.pending_reply = asyncio.create_task(self.stream_reply(text))
                self.app.invalidate()

                try:
                    if await self.pending_reply == "rate_limit":
                        self.queue_paused = True
                        self.app.invalidate()
                except asyncio.CancelledError:
                    if _task_is_cancelling():
                        raise
                    self.active_turn = None
                    self.checkpoint_session(force=True)

        with patch_stdout(raw=True):
            def start_receiver() -> None:
                self.app.create_background_task(receive_messages())

            try:
                return await self.app.run_async(
                    pre_run=start_receiver
                )
            finally:
                task = self.pending_reply
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self.pending_reply = None

    def run_conversation(self) -> None:
        try:
            asyncio.run(self.chat_loop())
        except (KeyboardInterrupt, EOFError):
            pass

    def handle_command(self, text: str) -> Markdown | Text | None:
        if text == "/clear":
            self.session_record = None
            self.saved_session_record = None
            self.active_turn = None
            self.last_session_checkpoint = None
            self.session_save_error = None
            self.messages = []
            self.transcript.clear()
            self.transcript_control.scroll_to_bottom()
            self.transcript.append(
                {
                    "role": "notice",
                    "content": self.header,
                    "persist": False,
                }
            )
        elif text == "/help":
            return Markdown(
                "**Commands**\n\n"
                "- `/help` - Show commands\n"
                "- `/user` - Show current account\n"
                "- `/clear` - Start new session with empty context\n"
                "- `/resume [title-or-id]` - Resume a saved conversation\n"
                "- `/rename <name>` - Rename the current conversation\n"
                "- `/model` - Set the AI model\n"
                "- `/login` - Sign in with ChatGPT Subscription\n"
                "- `/logout` - Sign out\n"
                "- `/quit` or `/exit` - leave Vin"
            )
        elif text == "/user":
            if self.credentials and self.credentials.get("access_token"):
                account = self.credentials.get("email") or self.credentials["subject"]
                notice = f"{PROVIDER_LABELS['chatgpt']}: {account}"
            else:
                notice = "Signed out. Use /login to sign in"
            return Text(notice)

    async def discover_chatgpt_models(self) -> list[ModelMetadata]:
        credentials = await self.prepare_request_credentials()

        models = await asyncio.to_thread(list_chatgpt_models, credentials)

        save_cached_chatgpt_models(credentials, models)
        return models

    async def load_initial_model(self) -> None:
        models = await self.discover_chatgpt_models()
        provider_name = PROVIDER_LABELS["chatgpt"]
        if not models:
            raise RuntimeError(
                f"No {provider_name} models are available for this account."
            )

        preference: ModelPreference | None = get_chatgpt_model_preference(self.transcript, models)
        if preference is None:
            selected_model = await self.choose_inline(
                f"Choose a {provider_name} model",
                [(model["slug"], model["display_name"]) for model in models],
            )
            preference = {"provider": "chatgpt", "model": selected_model}
            save_model_preference(preference["provider"], preference["model"])
        self.models = models
        self.selected_provider = preference["provider"]
        self.selected_model = preference["model"]

    async def initialize_session(self, *, sign_in: bool = False) -> None:
        if self.credentials is None and not sign_in:
            self.credentials = await asyncio.to_thread(load_active_session)

        if not sign_in and self.credentials and self.credentials.get("access_token"):
            preference = load_model_preference()
            if preference is not None:
                self.models = load_cached_chatgpt_models(self.credentials)
                self.selected_provider = preference["provider"]
                self.selected_model = preference["model"]
                return

        if sign_in or not self.credentials or not self.credentials.get("access_token"):
            await self.choose_inline(
                "Sign in to Vin", [("chatgpt", "Sign in with ChatGPT")]
            )
            if sign_in:
                await self.sign_out(announce=False)

            loop = asyncio.get_running_loop()
            cancel_event = Event()

            def append_notice(message: str) -> None:
                if not cancel_event.is_set() and not self.app.is_done:
                    self.transcript.append({"role": "notice", "content": Text(message)})
                    self.app.invalidate()

            def notify(message: str) -> None:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(append_notice, message)

            try:
                signed_in = await asyncio.to_thread(
                    login_with_chatgpt,
                    self.credentials,
                    cancel_event=cancel_event,
                    notify=notify,
                )
            finally:
                cancel_event.set()
            if signed_in is None or self.app.is_done:
                return
            self.credentials = signed_in
        await self.load_initial_model()

    def model_indicator_text(self) -> str:
        if self.selected_model:
            assert self.selected_provider is not None
            return (
                f" {PROVIDER_LABELS[self.selected_provider]} · "
                + model_display_name(self.models, self.selected_model)
            )
        return (
            " No model selected · /login"
            if self.credentials and self.credentials.get("access_token")
            else " Not signed in · /login"
        )

    def run(self) -> None:
        self.header = Markdown(
            "**Vin**\n\n" "Type a message · `/help` for commands · `/exit` to quit"
        )
        self.transcript.append(
            {
                "role": "notice",
                "content": self.header,
                "persist": False,
            }
        )
        build_workspace(self)
        self.loading_model = True
        self.run_conversation()
