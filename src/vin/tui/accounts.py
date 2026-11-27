"""Signing in and out of ChatGPT and OpenRouter, and fresh credentials for a request."""

import asyncio
from typing import TYPE_CHECKING

from rich.text import Text

from vin.auth.chatgpt import (
    logout_chatgpt,
    refresh_credentials,
)
from vin.auth.openrouter import (
    check_openrouter_key,
    delete_openrouter_account,
    mask_key,
    save_openrouter_account,
)
from vin.provider.openrouter import load_openrouter_whitelist
from vin.tui.menus import choose_or_back, task_is_cancelling
from vin.types.auth import AccountRecord, Credentials, Registration

if TYPE_CHECKING:
    from vin.tui.app import VinApp


async def sign_in_openrouter(owner: "VinApp") -> None:
    try:
        key = await owner.ask_secret("Paste your OpenRouter API key")
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return
    account = await asyncio.to_thread(check_openrouter_key, key)
    save_openrouter_account(account)
    owner.openrouter = account
    if owner.selected_provider is None:
        owner.selected_provider = "openrouter"
    owner.transcript.append(
        {
            "role": "notice",
            "content": f"Signed in to OpenRouter · {mask_key(account['key'])}.",
        }
    )
    owner.set_models("openrouter", load_openrouter_whitelist())
    if not any(model["provider"] == "openrouter" for model in owner.models):
        added = await owner.add_openrouter_model("")
        if added:
            owner.transcript.append({"role": "notice", "content": added})
    chosen = owner.auto_select_model()
    if chosen:
        owner.transcript.append({"role": "notice", "content": chosen})


async def prepare_request_credentials(owner: "VinApp") -> AccountRecord | None:
    if not owner.credentials or not owner.credentials.get("access_token"):
        return owner.credentials

    assert "access_token" in owner.credentials
    if owner.credential_refresh is None:
        original = owner.credentials

        async def refresh() -> Credentials:
            updated = await asyncio.to_thread(refresh_credentials, original)
            if owner.credentials is not original:
                raise RuntimeError("ChatGPT session changed during token refresh.")
            owner.credentials = updated
            return updated

        task = asyncio.create_task(refresh())
        owner.credential_refresh = task

        def finish(done: asyncio.Task[Credentials]) -> None:
            # A cancelled waiter must not cancel a rotating-token exchange.
            if not done.cancelled():
                done.exception()
            if owner.credential_refresh is done:
                owner.credential_refresh = None

        task.add_done_callback(finish)

    return await asyncio.shield(owner.credential_refresh)


async def logout_targets(owner: "VinApp") -> list[str]:
    """Which accounts /logout signs out of: asks when both are signed in."""
    chatgpt = bool(owner.credentials and owner.credentials.get("access_token"))
    openrouter = owner.openrouter is not None
    if chatgpt and openrouter:
        choice = await choose_or_back(
            owner, "Sign out of",
            [("chatgpt", "ChatGPT"), ("openrouter", "OpenRouter"), ("both", "Both")],
        )
        return [] if choice is None else ["chatgpt", "openrouter"] if choice == "both" else [choice]
    if chatgpt or openrouter:
        return ["chatgpt" if chatgpt else "openrouter"]
    owner.transcript.append({"role": "notice", "content": "You're not signed in."})
    return []


async def run_auth_command(owner: "VinApp", text: str) -> None:
    targets = [] if text == "/login" else await logout_targets(owner)
    if text == "/logout" and not targets:
        return

    async def work() -> None:
        if text == "/login":
            await owner.initialize_session(sign_in=True)
            return
        for target in targets:
            if target == "openrouter":
                await owner.sign_out_openrouter(load_models="chatgpt" not in targets)
            else:
                await owner.sign_out()

    owner.loading_model = True
    task = asyncio.create_task(work())
    owner.pending_auth = task
    owner.app.invalidate()

    try:
        await task
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
    finally:
        owner.pending_auth = None
        owner.auth_status = None
        owner.loading_model = False
        owner.app.invalidate()


async def sign_out(owner: "VinApp", *, announce: bool = True) -> None:
    credentials: AccountRecord | None = None
    notices: list[str] = []
    owner.auth_status = "Signing out…"
    owner.app.invalidate()

    async def finish_sign_out() -> Registration | None:
        nonlocal credentials
        if owner.credential_refresh is not None:
            try:
                await asyncio.shield(owner.credential_refresh)
            except Exception:
                pass
        credentials = (
            owner.credentials.copy() if owner.credentials is not None else None
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
            owner.auth_status = "Finishing sign-out…"
            owner.app.invalidate()
            registration = await asyncio.shield(task)
        owner.credentials = registration
    except RuntimeError, OSError, ValueError:
        if credentials is not None and not credentials.get("access_token"):
            owner.credentials = credentials
        raise
    finally:
        if not owner.credentials or not owner.credentials.get("access_token"):
            owner.set_models("chatgpt", [])
            if owner.selected_provider != "openrouter":
                owner.selected_provider = (
                    "openrouter" if owner.openrouter is not None else None
                )
                owner.selected_model = None
                owner.last_usage = None
        owner.auth_status = None
        owner.app.invalidate()

    for message in notices:
        owner.transcript.append(
            {
                "role": "notice",
                "content": Text(message),
            }
        )

    if announce:
        owner.transcript.append(
            {
                "role": "notice",
                "content": "Signed out of ChatGPT.",
            }
        )

    if cancelled:
        raise asyncio.CancelledError


async def sign_out_openrouter(
    owner: "VinApp", *, announce: bool = True, load_models: bool = True
) -> None:
    delete_openrouter_account()
    owner.openrouter = None
    owner.set_models("openrouter", [])
    if owner.selected_provider == "openrouter":
        owner.selected_model = None
        owner.last_usage = None
        owner.selected_provider = (
            "chatgpt"
            if owner.credentials and owner.credentials.get("access_token")
            else None
        )
    if announce:
        owner.transcript.append(
            {"role": "notice", "content": "Signed out of OpenRouter."}
        )
    if load_models and owner.selected_provider == "chatgpt":
        await owner.load_initial_model()
