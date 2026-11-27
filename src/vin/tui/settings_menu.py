"""The /settings menus: subagents, web search and its key, web fetch."""

import asyncio
from typing import TYPE_CHECKING, Unpack

from vin.auth.openrouter import mask_key
from vin.auth.search import (
    check_search_key,
    delete_search_key,
    save_search_key,
)
from vin.settings import save_settings
from vin.tui.menus import choose_or_back, setting_row, task_is_cancelling
from vin.types.settings import Settings, SettingsChange

if TYPE_CHECKING:
    from vin.tui.app import VinApp


async def ask_search_key(owner: "VinApp") -> str | None:
    try:
        key = await owner.ask_secret("Paste your Tavily API key")
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    was_loading = owner.loading_model
    owner.auth_status = "Checking Tavily key…"
    owner.loading_model = True
    owner.app.invalidate()
    try:
        return await asyncio.to_thread(check_search_key, key)
    finally:
        owner.auth_status = None
        owner.loading_model = was_loading
        owner.app.invalidate()


def update_settings(owner: "VinApp", **changes: Unpack[SettingsChange]) -> Settings:
    updated: Settings = {**owner.settings, **changes}
    save_settings(updated)
    owner.settings = updated
    return updated


def search_on(owner: "VinApp") -> bool:
    return owner.settings["search"] and owner.search_key is not None


async def toggle_search(owner: "VinApp") -> None:
    if owner.search_key is None:
        key = await owner.ask_search_key()
        if key is None:
            return
        save_search_key(key)
        owner.search_key = key
        _ = owner.update_settings(search=True)
    else:
        _ = owner.update_settings(search=not owner.settings["search"])


async def change_search_key(owner: "VinApp") -> None:
    if owner.search_key is None:
        return
    choice = await choose_or_back(
        owner, "Web search key", [("replace", "Replace key"), ("remove", "Remove key")]
    )
    if choice == "remove":
        delete_search_key()
        owner.search_key = None
        _ = owner.update_settings(search=False)
    elif choice == "replace":
        key = await owner.ask_search_key()
        if key is not None:
            save_search_key(key)
            owner.search_key = key


async def toggle_fetch(owner: "VinApp") -> None:
    _ = owner.update_settings(fetch=not owner.settings["fetch"])


async def change_subagent_settings(owner: "VinApp") -> None:
    focus = "enabled"
    while True:
        settings = owner.settings
        rows = [
            ("enabled", setting_row("Subagents", "on" if settings["subagents"] else "off")),
            ("budget", setting_row("Request limit", f"{settings['subagent_budget']} ›")),
        ]
        choice = await choose_or_back(owner, "Subagents", rows, focus)
        if choice is None:
            return
        focus = choice
        if choice == "enabled":
            _ = owner.update_settings(subagents=not settings["subagents"])
            continue
        value = await choose_or_back(
            owner,
            "Subagent request limit",
            [(str(number), str(number)) for number in range(1, 33)],
            str(settings["subagent_budget"]),
        )
        if value is not None:
            _ = owner.update_settings(subagent_budget=int(value))


async def change_web_search_settings(owner: "VinApp") -> None:
    focus = "enabled"
    while True:
        rows = [("enabled", setting_row("Web search", "on" if owner.search_on() else "off"))]
        if owner.search_key is not None:
            rows.append(("key", setting_row("Key", f"{mask_key(owner.search_key)} ›")))
        choice = await choose_or_back(owner, "Web search", rows, focus)
        if choice is None:
            return
        focus = choice
        if choice == "key":
            await owner.change_search_key()
        else:
            await owner.toggle_search()


async def change_settings(owner: "VinApp") -> None:
    """The /settings menu. It stays open and shows each change until Esc."""
    focus = "subagents"
    while True:
        rows = [
            ("subagents", setting_row("Subagents", "›")),
            ("search", setting_row("Web search", "›")),
            ("fetch", setting_row("Web fetch", "on" if owner.settings["fetch"] else "off")),
        ]
        choice = await choose_or_back(owner, "Settings", rows, focus)
        if choice is None:
            return
        focus = choice
        if choice == "search":
            await owner.change_web_search_settings()
        elif choice == "subagents":
            await owner.change_subagent_settings()
        else:
            await owner.toggle_fetch()
