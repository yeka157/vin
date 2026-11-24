"""Application state, conversation flow, commands, and account/model setup."""

import asyncio
from collections import deque
from queue import Empty
from threading import Event, Thread
from time import monotonic
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict
from collections.abc import Coroutine, Sequence

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
from vin.auth.openrouter import (
    check_openrouter_key,
    delete_openrouter_account,
    load_openrouter_account,
    save_openrouter_account,
)
from vin.provider.chatgpt import list_chatgpt_models, stream_chatgpt_response
from vin.provider.openrouter import (
    list_openrouter_models,
    load_openrouter_whitelist,
    save_openrouter_whitelist,
    stream_openrouter_response,
)
from vin.provider.model_cache import (
    load_cached_chatgpt_models,
    save_cached_chatgpt_models,
)
from vin.provider.preferences import (
    DEFAULT_SUBAGENT_SETTINGS,
    load_model_preference,
    load_subagent_settings,
    save_model_preference,
    save_subagent_settings,
)
from vin.sessions.records import new_session
from vin.tools import LAUNCH_DIR
from vin.tui import conversations
from vin.tui.constants import PROVIDER_LABELS, is_command
from vin.tui.layout import build_workspace
from vin.tui.subagents import apply_subagent_event, finish_subagent
from vin.tui.transcript import TranscriptControl, format_duration, render_transcript
from vin.provider.events import ReplyEvents, TokenUsage
from vin.types import (
    AccountRecord,
    ActiveTurn,
    AssistantEntry,
    Credentials,
    OpenRouterAccount,
    Provider,
    Registration,
    History,
    ModelMetadata,
    ModelPreference,
    SessionRecord,
    Subagent,
    SubagentSettings,
    ToolEntry,
    TranscriptEntry,
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
    preference: ModelPreference | None = get_chatgpt_model_preference(
        transcript, models
    )
    if preference is None:
        selected_model = choose_chatgpt_model(models)
        preference = {"provider": "chatgpt", "model": selected_model}
        save_model_preference(preference["provider"], preference["model"])
    return preference


def get_chatgpt_model_preference(
    transcript: list[TranscriptEntry], models: Sequence[ModelMetadata]
) -> ModelPreference | None:
    preference = load_model_preference()
    if preference is None or preference["provider"] != "chatgpt":
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


class ReplyOptions(TypedDict, total=False):
    compact_threshold: int
    compact_only: bool
    subagents: bool
    subagent_budget: int


class VinApp:
    session: PromptSession[str]
    app: Application[str]
    input_window: Window
    header: Markdown

    def __init__(self) -> None:
        self.credentials: AccountRecord | None = None
        self.models: list[ModelMetadata] = []
        self.selected_provider: Provider | None = None
        self.selected_model: str | None = None
        self.messages: History = []
        self.transcript: list[TranscriptEntry] = []
        self.transcript_control = TranscriptControl(self.get_transcript_entries)
        self.pending_reply: (
            asyncio.Task[Literal["error", "rate_limit"] | None] | None
        ) = None
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
        self.last_usage: TokenUsage | None = None
        self.tool_status: (
            tuple[str, Literal["running", "finished", "failed"]] | None
        ) = None
        self._credential_refresh: asyncio.Task[Credentials] | None = None
        self.project_dir: Path = LAUNCH_DIR
        self.session_record: SessionRecord | None = None
        self.saved_session_record: SessionRecord | None = None
        self.active_turn: ActiveTurn | None = None
        self.last_session_checkpoint: float | None = None
        self.session_save_error: str | None = None
        self.subagents: list[Subagent] = []
        self.viewed: str | None = None
        self.panel_start = 0
        self.panel_index: int | None = None
        self.subagent_settings: SubagentSettings = DEFAULT_SUBAGENT_SETTINGS.copy()
        self.openrouter: OpenRouterAccount | None = None
        self.secret_future: asyncio.Future[str] | None = None

    async def ask_secret(self, message: str) -> str:
        if self.choice_future is not None or self.secret_future is not None:
            raise RuntimeError("A choice is already open")
        buffer = self.session.default_buffer
        draft = buffer.document
        buffer.reset()
        self.auth_status = message
        self.secret_future = asyncio.get_running_loop().create_future()
        self.app.invalidate()
        try:
            return await self.secret_future
        finally:
            self.secret_future = None
            buffer.document = draft
            self.auth_status = None
            self.app.invalidate()

    async def sign_in_openrouter(self) -> None:
        try:
            key = await self.ask_secret(
                "Paste your OpenRouter API key · Enter to confirm · Esc to cancel"
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return
        self.auth_status = "Checking key…"
        self.app.invalidate()
        account = await asyncio.to_thread(check_openrouter_key, key)
        save_openrouter_account(account)
        self.openrouter = account
        self.selected_provider = "openrouter"
        self.selected_model = None
        self.last_usage = None
        self.transcript.append(
            {
                "role": "notice",
                "content": f"Signed in to OpenRouter{' · ' + account['label'] if account['label'] else ''}.",
            }
        )
        self.set_models("openrouter", load_openrouter_whitelist())
        if not any(model["provider"] == "openrouter" for model in self.models):
            added = await self.add_openrouter_model("")
            if added:
                self.transcript.append({"role": "notice", "content": added})
        if self.models:
            chosen = await self.change_model()
            if chosen:
                self.transcript.append({"role": "notice", "content": chosen})

    def panel_agents(self) -> list[Subagent]:
        return self.subagents[self.panel_start :]

    def viewed_subagent(self) -> Subagent | None:
        return next(
            (item for item in self.subagents if item["id"] == self.viewed), None
        )

    def checkpoint_session(self, *, force: bool = False) -> bool:
        return conversations.checkpoint_session(self, force=force)

    def get_transcript_entries(self) -> list[TranscriptEntry]:
        subagent = self.viewed_subagent()
        if subagent is not None:
            return list(subagent["transcript"])
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

    async def stream_reply(
        self, text: str, *, compact_only: bool = False
    ) -> Literal["error", "rate_limit"] | None:
        events = ReplyEvents()
        cancel_event = Event()
        entry: AssistantEntry = {"role": "assistant", "content": "", "streaming": True}
        request = [message.copy() for message in self.messages]
        if not compact_only:
            request.append({"role": "user", "content": text})
        self.reply_entry = entry
        self.request_started_at = monotonic()
        self.last_request_duration = None
        completed = False
        text_dirty = False
        tool_entry: ToolEntry | None = None
        subagent: Subagent | None = None
        self.tool_status = None

        if not compact_only:
            self.panel_start = len(self.subagents)
            self.panel_index = None

        try:
            if compact_only and not request:
                self.transcript.append(
                    {"role": "notice", "content": "No conversation context to compact."}
                )
                completed = True
                return
            credentials: AccountRecord | OpenRouterAccount | None
            if self.selected_provider == "openrouter":
                credentials = self.openrouter
            else:
                try:
                    credentials = await self.prepare_request_credentials()
                except Exception as error:
                    message = "Could not refresh ChatGPT sign-in. Please try again."
                    if isinstance(error, TokenExchangeError) and error.requires_login:
                        message = (
                            "ChatGPT sign-in has expired. Sign in again with /login."
                        )
                    self.transcript.append(
                        {"role": "notice", "content": Text(message, style="red")}
                    )
                    return "error"

            compact_threshold: int | None = None
            for model in self.models:
                if model["slug"] == self.selected_model:
                    window = model.get("context_window")
                    if window is not None:
                        compact_threshold = window * 4 // 5
                    break

            if compact_only:
                compact_threshold = 1024

            if compact_threshold is None and self.selected_model is not None:
                notice = (
                    "Automatic compaction is off because this model's "
                    "context limit is unknown."
                )
                if not any(
                    row["role"] == "notice" and row["content"] == notice
                    for row in self.transcript
                ):
                    self.transcript.append({"role": "notice", "content": notice})

            options: ReplyOptions = {}
            if compact_threshold is not None:
                options["compact_threshold"] = compact_threshold
            if compact_only:
                options["compact_only"] = True

            if not compact_only:
                options["subagents"] = self.subagent_settings["enabled"]
                options["subagent_budget"] = self.subagent_settings["budget"]

            Thread(
                target=(
                    stream_openrouter_response
                    if self.selected_provider == "openrouter"
                    else stream_chatgpt_response
                ),
                args=(
                    (credentials or {}).copy(),
                    self.selected_model,
                    request,
                    events,
                    cancel_event,
                ),
                kwargs=options,
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
                elif event[0] == "usage":
                    self.last_usage = event[1].copy()
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
                        "running"
                        if kind == "tool_started"
                        else "finished" if kind == "tool_finished" else "failed"
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
                        if isinstance(value, dict) and "task_id" in value:
                            subagent = {
                                "id": value["task_id"],
                                "description": detail,
                                "status": "running",
                                "started_at": monotonic(),
                                "duration": None,
                                "tokens": 0,
                                "transcript": [
                                    {"role": "user", "content": value.get("prompt", "")}
                                ],
                            }
                            self.subagents.append(subagent)
                    elif tool_entry is not None:
                        assert "label" in tool_entry
                        tool_entry["status"] = state
                        tool_entry["content"] = Text(
                            f"● {tool_entry['label']} · {state}",
                            style="red" if kind == "tool_failed" else "dim",
                        )
                        tool_entry = None
                        if subagent is not None:
                            finish_subagent(
                                subagent,
                                "failed" if kind == "tool_failed" else "finished",
                            )
                            subagent = None
                elif event[0] == "subagent":
                    target = next(
                        (
                            item
                            for item in self.subagents
                            if item["id"] == event[1]["task_id"]
                        ),
                        None,
                    )
                    if target is not None:
                        apply_subagent_event(target, event[1]["event"])
                elif event[0] == "complete":
                    value = event[1]
                    entry["streaming"] = False
                    if value is None and compact_only:
                        self.transcript.append(
                            {"role": "notice", "content": "Nothing to compact yet."}
                        )
                    elif value is None:
                        self.messages.extend(
                            [
                                {"role": "user", "content": text},
                                {"role": "assistant", "content": entry["content"]},
                            ]
                        )
                    else:
                        newly_compacted = any(
                            item.get("type") == "compaction"
                            and item not in self.messages
                            for item in value
                        )
                        self.messages[:] = value
                        if newly_compacted:
                            self.transcript.append(
                                {
                                    "role": "notice",
                                    "content": "Conversation compacted.",
                                }
                            )
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
                elif kind in ("text", "subagent"):
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
                if subagent is not None:
                    finish_subagent(subagent, interrupted_state)
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
                    if not compact_only:
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

    async def wait_for_models(
        self, load: Coroutine[object, object, list[ModelMetadata]]
    ) -> list[ModelMetadata] | None:
        self.loading_model = True
        task = asyncio.create_task(load)
        self.pending_model_load = task
        self.app.invalidate()
        try:
            return await task
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None
        finally:
            self.pending_model_load = None
            self.loading_model = False
            self.app.invalidate()

    def select_model(self, model: ModelMetadata) -> str:
        save_model_preference(model["provider"], model["slug"])
        self.selected_provider = model["provider"]
        self.selected_model = model["slug"]
        self.last_usage = None
        return f"Model: {PROVIDER_LABELS[model['provider']]} · {model['display_name']}"

    def save_whitelist(self, models: list[ModelMetadata]) -> None:
        save_openrouter_whitelist(models)
        self.set_models("openrouter", models)

    async def change_model(self) -> str | None:
        chatgpt = bool(self.credentials and self.credentials.get("access_token"))
        if not chatgpt and self.openrouter is None:
            return "Sign in with /login before choosing a model."
        if chatgpt and not any(model["provider"] == "chatgpt" for model in self.models):
            models = await self.wait_for_models(self.discover_chatgpt_models())
            if models is None:
                return None
            self.set_models("chatgpt", models)
        if not self.models:
            return "No models yet. Use /model add to add an OpenRouter model."
        options = {
            f"{model['provider']}:{model['slug']}": model for model in self.models
        }
        try:
            key = await self.choose_inline(
                "Choose a model",
                [
                    (
                        key,
                        f"{model['display_name']} · {PROVIDER_LABELS[model['provider']]}",
                    )
                    for key, model in options.items()
                ],
                default=f"{self.selected_provider}:{self.selected_model}",
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None
        chosen = options[key]
        if (
            chosen["provider"] == self.selected_provider
            and chosen["slug"] == self.selected_model
        ):
            return None
        return self.select_model(chosen)

    async def add_openrouter_model(self, query: str) -> str | None:
        if self.openrouter is None:
            return "Sign in to OpenRouter with /login before adding a model."
        catalog = await self.wait_for_models(asyncio.to_thread(list_openrouter_models))
        if catalog is None:
            return None
        whitelist = [
            model for model in self.models if model["provider"] == "openrouter"
        ]
        added = {model["slug"] for model in whitelist}
        search = query.casefold()
        matches = [
            model
            for model in catalog
            if model["slug"] not in added
            and (
                search in model["slug"].casefold()
                or search in model["display_name"].casefold()
            )
        ]
        if not matches:
            return (
                f'No OpenRouter models match "{query}".'
                if query
                else "Every OpenRouter model with tool support is already added."
            )
        try:
            slug = await self.choose_inline(
                "Add an OpenRouter model",
                [(model["slug"], model["display_name"]) for model in matches],
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None
        model = next(model for model in matches if model["slug"] == slug)
        whitelist.append(model)
        self.save_whitelist(whitelist)
        return (
            f"Added OpenRouter · {model['display_name']}. Use /model to switch to it."
        )

    async def remove_openrouter_model(self) -> str | None:
        whitelist = [
            model for model in self.models if model["provider"] == "openrouter"
        ]
        if not whitelist:
            return "No OpenRouter models to remove."
        try:
            slug = await self.choose_inline(
                "Remove an OpenRouter model",
                [(model["slug"], model["display_name"]) for model in whitelist],
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None
        removed = next(model for model in whitelist if model["slug"] == slug)
        remaining = [model for model in whitelist if model["slug"] != slug]
        self.save_whitelist(remaining)
        message = f"Removed OpenRouter · {removed['display_name']}."
        if self.selected_provider == "openrouter" and self.selected_model == slug:
            self.selected_model = None
            self.last_usage = None
            if self.models:
                outcome = await self.change_model()
                if outcome:
                    return f"{message} {outcome}"
        return message

    async def run_auth_command(self, text: str) -> None:
        self.loading_model = True
        task = asyncio.create_task(
            self.initialize_session(sign_in=True)
            if text == "/login"
            else (
                self.sign_out_openrouter()
                if self.selected_provider == "openrouter"
                else self.sign_out()
            )
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
                self.set_models("chatgpt", [])
                if self.selected_provider != "openrouter":
                    self.selected_provider = (
                        "openrouter" if self.openrouter is not None else None
                    )
                    self.selected_model = None
                    self.last_usage = None
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

    async def sign_out_openrouter(self, *, announce: bool = True) -> None:
        delete_openrouter_account()
        self.openrouter = None
        self.set_models("openrouter", [])
        if self.selected_provider == "openrouter":
            self.selected_model = None
            self.last_usage = None
            self.selected_provider = (
                "chatgpt"
                if self.credentials and self.credentials.get("access_token")
                else None
            )
        if announce:
            self.transcript.append(
                {"role": "notice", "content": "Signed out of OpenRouter."}
            )
        if self.selected_provider == "chatgpt":
            await self.load_initial_model()

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
                        elif parts[0] == "/model":
                            try:
                                action, _, query = (
                                    parts[1] if len(parts) > 1 else ""
                                ).partition(" ")
                                if not action:
                                    outcome = await self.change_model()
                                elif action == "add":
                                    outcome = await self.add_openrouter_model(
                                        query.strip()
                                    )
                                elif action == "remove" and not query.strip():
                                    outcome = await self.remove_openrouter_model()
                                else:
                                    outcome = Text(
                                        "Use /model, /model add [search] or /model remove.",
                                        style="red",
                                    )
                            except (RuntimeError, OSError, ValueError) as error:
                                outcome = Text(str(error), style="red")
                            self.transcript.append({"role": "user", "content": text})
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        elif text == "/settings":
                            try:
                                outcome = await self.change_settings()
                            except (RuntimeError, OSError, ValueError) as error:
                                outcome = Text(str(error), style="red")
                            self.transcript.append({"role": "user", "content": text})
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        elif text == "/tasks":
                            try:
                                outcome = await self.open_task()
                            except (RuntimeError, OSError, ValueError) as error:
                                outcome = Text(str(error), style="red")
                            self.transcript.append({"role": "user", "content": text})
                            if outcome is not None:
                                self.transcript.append(
                                    {"role": "notice", "content": outcome}
                                )
                        elif text == "/compact":
                            self.reply_entry = None
                            self.pending_reply = asyncio.create_task(
                                self.stream_reply("", compact_only=True)
                            )
                            self.app.invalidate()
                            try:
                                result = await self.pending_reply
                                if result == "rate_limit":
                                    self.queue_paused = True
                            except asyncio.CancelledError:
                                if _task_is_cancelling():
                                    raise
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
                return await self.app.run_async(pre_run=start_receiver)
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
            self.last_usage = None
            self.transcript.clear()
            self.transcript_control.scroll_to_bottom()
            self.transcript.append(
                {
                    "role": "notice",
                    "content": self.header,
                    "persist": False,
                }
            )
            self.subagents = []
            self.panel_start = 0
            self.viewed = None
            self.panel_index = None
        elif text == "/help":
            return Markdown(
                "**Commands**\n\n"
                "- `/help` - Show commands\n"
                "- `/user` - Show current account\n"
                "- `/clear` - Start new session with empty context\n"
                "- `/resume [title-or-id]` - Resume a saved conversation\n"
                "- `/rename <name>` - Rename the current conversation\n"
                "- `/model` - Set the AI model\n"
                "- `/model add [search]` - Add an OpenRouter model\n"
                "- `/model remove` - Remove an OpenRouter model\n"
                "- `/compact` - Compact the conversation context\n"
                "- `/settings` - Manage settings\n"
                "- `/tasks` - View tasks from this conversation\n"
                "- `/login` - Sign in with ChatGPT or an OpenRouter API key\n"
                "- `/logout` - Sign out\n"
                "- `/quit` or `/exit` - leave Vin"
            )
        elif text == "/user":
            lines: list[str] = []
            if self.credentials and self.credentials.get("access_token"):
                account = self.credentials.get("email") or self.credentials["subject"]
                lines.append(f"{PROVIDER_LABELS['chatgpt']}: {account}")
            if self.openrouter is not None:
                lines.append(f"OpenRouter: {self.openrouter['label'] or 'API key'}")
            notice = "\n".join(lines) if lines else "Signed out. Use /login to sign in"
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

        preference: ModelPreference | None = get_chatgpt_model_preference(
            self.transcript, models
        )
        if preference is None:
            selected_model = await self.choose_inline(
                f"Choose a {provider_name} model",
                [(model["slug"], model["display_name"]) for model in models],
            )
            preference = {"provider": "chatgpt", "model": selected_model}
            save_model_preference(preference["provider"], preference["model"])
        self.set_models("chatgpt", models)
        self.selected_provider = preference["provider"]
        self.selected_model = preference["model"]

    def set_models(self, provider: Provider, models: list[ModelMetadata]) -> None:
        self.models = [
            model for model in self.models if model["provider"] != provider
        ] + models

    async def initialize_session(self, *, sign_in: bool = False) -> None:
        if not sign_in:
            try:
                self.subagent_settings = load_subagent_settings()
            except RuntimeError as error:
                self.transcript.append(
                    {"role": "notice", "content": Text(str(error), style="red")}
                )
            try:
                self.openrouter = load_openrouter_account()
                if self.openrouter is not None:
                    self.set_models("openrouter", load_openrouter_whitelist())
            except RuntimeError as error:
                self.transcript.append(
                    {"role": "notice", "content": Text(str(error), style="red")}
                )
        if self.credentials is None and not sign_in:
            self.credentials = await asyncio.to_thread(load_active_session)

        if not sign_in and self.credentials and self.credentials.get("access_token"):
            preference = load_model_preference()
            if preference is not None and preference["provider"] == "chatgpt":
                self.set_models("chatgpt", load_cached_chatgpt_models(self.credentials))
                self.selected_provider = preference["provider"]
                self.selected_model = preference["model"]
                return

        if not sign_in and self.openrouter is not None:
            preference = load_model_preference()
            self.selected_provider = "openrouter"
            self.selected_model = (
                preference["model"]
                if preference is not None
                and preference["provider"] == "openrouter"
                and any(model["slug"] == preference["model"] for model in self.models)
                else None
            )
            return

        if sign_in or not self.credentials or not self.credentials.get("access_token"):
            method = await self.choose_inline(
                "Sign in to Vin",
                [
                    ("chatgpt", "Sign in with ChatGPT"),
                    ("openrouter", "Use an OpenRouter API key"),
                ],
            )
            if method == "openrouter":
                await self.sign_in_openrouter()
                return

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
        subagent = self.viewed_subagent()
        if subagent is not None:
            return (
                f" Viewing task · {subagent['description']} · read-only · Esc to return"
            )
        if self.selected_model:
            assert self.selected_provider is not None
            return (
                f" {PROVIDER_LABELS[self.selected_provider]} · "
                + model_display_name(self.models, self.selected_model)
            )
        if self.selected_provider == "openrouter":
            return " OpenRouter · No model selected"
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

    async def change_settings(self) -> str | None:
        settings = self.subagent_settings
        try:
            choice = await self.choose_inline(
                "Settings",
                [
                    (
                        "enabled",
                        f"Subagents: {'on' if settings['enabled'] else 'off'}",
                    ),
                    ("budget", f"Subagent request limit: {settings['budget']}"),
                ],
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None

        if choice == "enabled":
            updated: SubagentSettings = {
                "enabled": not settings["enabled"],
                "budget": settings["budget"],
            }
        else:
            try:
                value = await self.choose_inline(
                    "Subagent request limit",
                    [(str(number), str(number)) for number in range(1, 33)],
                    default=str(settings["budget"]),
                )
            except asyncio.CancelledError:
                if _task_is_cancelling():
                    raise
                return None

            updated = {"enabled": settings["enabled"], "budget": int(value)}

        save_subagent_settings(updated)
        self.subagent_settings = updated
        return f"Subagents {'on' if updated['enabled'] else 'off'} · request limit {updated['budget']}"

    async def open_task(self) -> str | None:
        if not self.subagents:
            return "No subagents in this conversation."

        try:
            choice = await self.choose_inline(
                "Open a subagent",
                [
                    (item["id"], f"{item['description']} · {item['status']}")
                    for item in reversed(self.subagents)
                ],
            )
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            return None
        self.viewed = choice
        self.panel_index = None
        self.transcript_control.scroll_to_bottom()
        return None
