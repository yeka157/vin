"""VinApp: the application state, the input loop and commands.

Each area's code lives in its own module (menus, settings_menu, accounts, model_menu, turns,
conversations); VinApp keeps a one-line method per action so key bindings, the layout and
tests reach everything through the app.
"""

import asyncio
from collections import deque
from collections.abc import Coroutine, Sequence
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Literal, Unpack

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Container, Window
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import RadioList
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text

from vin.auth.chatgpt import login_with_chatgpt
from vin.auth.credentials import load_active_session
from vin.auth.openrouter import (
    load_openrouter_account,
    mask_key,
)
from vin.auth.search import load_search_key
from vin.provider.model_cache import load_cached_chatgpt_models
from vin.provider.openrouter import load_openrouter_whitelist
from vin.provider.preferences import load_model_preference
from vin.sessions.records import new_session
from vin.settings import DEFAULT_SETTINGS, load_settings
from vin.tools import LAUNCH_DIR
from vin.tui import accounts, conversations, menus, model_menu, settings_menu, turns
from vin.tui.clipboard import copy_to_clipboard
from vin.tui.constants import PROVIDER_LABELS, is_command
from vin.tui.layout import build_workspace
from vin.tui.menus import task_is_cancelling
from vin.tui.model_picker import ModelPicker
from vin.tui.transcript import TranscriptControl, format_duration
from vin.types.auth import AccountRecord, Credentials, OpenRouterAccount
from vin.types.events import TokenUsage
from vin.types.history import History
from vin.types.models import ModelMetadata, Provider
from vin.types.sessions import ActiveTurn, SessionRecord
from vin.types.settings import Settings, SettingsChange
from vin.types.transcript import AssistantEntry, Subagent, TranscriptEntry


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
        self.transcript_control = TranscriptControl(self.get_transcript_entries, self.copy_selection)
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
        self.credential_refresh: asyncio.Task[Credentials] | None = None
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
        self.settings: Settings = DEFAULT_SETTINGS.copy()
        self.openrouter: OpenRouterAccount | None = None
        self.secret_future: asyncio.Future[str] | None = None
        self.picker: ModelPicker | None = None
        self.exit_armed_until = 0.0
        self.copied_until = 0.0
        self.picker_tab_focus = False
        self.search_key: str | None = None

    async def ask_secret(self, message: str) -> str:
        return await menus.ask_secret(self, message)

    async def ask_search_key(self) -> str | None:
        return await settings_menu.ask_search_key(self)

    def update_settings(self, **changes: Unpack[SettingsChange]) -> Settings:
        return settings_menu.update_settings(self, **changes)

    def search_on(self) -> bool:
        return settings_menu.search_on(self)

    async def toggle_search(self) -> None:
        return await settings_menu.toggle_search(self)

    async def change_search_key(self) -> None:
        return await settings_menu.change_search_key(self)

    async def sign_in_openrouter(self) -> None:
        return await accounts.sign_in_openrouter(self)

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

    def copy_selection(self, text: str) -> None:
        copy_to_clipboard(text)
        self.copied_until = monotonic() + 2
        self.app.invalidate()

    async def prepare_request_credentials(self) -> AccountRecord | None:
        return await accounts.prepare_request_credentials(self)

    async def choose_inline(
        self,
        message: str,
        options: Sequence[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        return await menus.choose_inline(self, message, options, default)

    async def show_panel(self, content: RenderableType) -> None:
        return await menus.show_panel(self, content)

    def accept_input(self, buffer: Buffer) -> bool:
        text = buffer.text
        if is_command(text):
            self.submitted.put_nowait(text)
        elif self.selected_model is None and text.strip():
            notice: TranscriptEntry = {
                "role": "notice",
                "content": "No model selected. Use /model to choose one.",
            }
            if self.transcript[-1:] != [notice]:
                self.transcript.append(notice)
            self.app.invalidate()
            return True
        elif text.strip() or (self.queue_paused and self.queued_prompts):
            if text.strip():
                self.queued_prompts.append(text)
            self.queue_paused = False
            self.submitted.put_nowait(None)
        return False

    async def stream_reply(
        self, text: str, *, compact_only: bool = False
    ) -> Literal["error", "rate_limit"] | None:
        return await turns.stream_reply(self, text, compact_only=compact_only)

    async def resume_conversation(self, query: str | None = None) -> str | None:
        return await conversations.resume_conversation(self, query)

    def rename_conversation(self, name: object) -> str | None:
        return conversations.rename_conversation(self, name)

    async def wait_for_models(
        self, load: Coroutine[object, object, list[ModelMetadata]]
    ) -> list[ModelMetadata] | None:
        return await model_menu.wait_for_models(self, load)

    def select_model(self, model: ModelMetadata) -> str:
        return model_menu.select_model(self, model)

    def save_whitelist(self, models: list[ModelMetadata]) -> None:
        return model_menu.save_whitelist(self, models)

    def auto_select_model(self) -> str | None:
        return model_menu.auto_select_model(self)

    async def pick_models(
        self,
        title: str,
        models: list[ModelMetadata],
        *,
        multi: bool,
        action: str,
        current: tuple[str, str] | None = None,
        search: str = "",
        tabs: tuple[str, ...] = (),
        tab: int = 0,
    ) -> list[ModelMetadata] | int | None:
        return await model_menu.pick_models(
            self, title, models, multi=multi, action=action, current=current, search=search,
            tabs=tabs, tab=tab,
        )

    async def change_model(self) -> str | Text | None:
        return await model_menu.change_model(self)

    async def add_openrouter_model(self, query: str) -> str | None:
        return await model_menu.add_openrouter_model(self, query)

    async def remove_openrouter_model(self) -> str | None:
        return await model_menu.remove_openrouter_model(self)

    async def run_auth_command(self, text: str) -> None:
        return await accounts.run_auth_command(self, text)

    async def sign_out(self, *, announce: bool = True) -> None:
        return await accounts.sign_out(self, announce=announce)

    async def sign_out_openrouter(self, *, announce: bool = True, load_models: bool = True) -> None:
        return await accounts.sign_out_openrouter(self, announce=announce, load_models=load_models)

    async def chat_loop(self) -> str:

        self.session.message = "❯ "
        self.session.default_buffer.accept_handler = self.accept_input

        async def receive_messages() -> None:
            if self.loading_model:
                try:
                    await self.initialize_session()
                except asyncio.CancelledError:
                    if task_is_cancelling():
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
                            self.transcript.append({"role": "user", "content": text})
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
                            # Menus only: nothing goes in the transcript unless saving fails.
                            await self.change_settings()
                        elif text == "/tasks":
                            outcome = await self.open_task()
                            if outcome is not None:
                                await self.show_panel(Text(outcome))
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
                                if task_is_cancelling():
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
                    if task_is_cancelling():
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
        except KeyboardInterrupt, EOFError:
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
                lines.append(f"OpenRouter: {mask_key(self.openrouter['key'])}")
            notice = "\n".join(lines) if lines else "Signed out. Use /login to sign in"
            return Text(notice)

    async def discover_chatgpt_models(self) -> list[ModelMetadata]:
        return await model_menu.discover_chatgpt_models(self)

    async def load_initial_model(self) -> None:
        return await model_menu.load_initial_model(self)

    def set_models(self, provider: Provider, models: list[ModelMetadata]) -> None:
        return model_menu.set_models(self, provider, models)

    async def initialize_session(self, *, sign_in: bool = False) -> None:
        if not sign_in:
            try:
                self.settings = load_settings()
            except RuntimeError as error:
                self.transcript.append(
                    {"role": "notice", "content": Text(str(error), style="red")}
                )
            try:
                self.search_key = load_search_key()
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
        return model_menu.model_indicator_text(self)

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

    async def toggle_fetch(self) -> None:
        return await settings_menu.toggle_fetch(self)

    async def change_subagent_settings(self) -> None:
        return await settings_menu.change_subagent_settings(self)

    async def change_web_search_settings(self) -> None:
        return await settings_menu.change_web_search_settings(self)

    async def change_settings(self) -> None:
        return await settings_menu.change_settings(self)

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
            if task_is_cancelling():
                raise
            return None
        self.viewed = choice
        self.panel_index = None
        self.transcript_control.scroll_to_bottom()
        return None
