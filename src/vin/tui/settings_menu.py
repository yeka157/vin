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
from vin.tui.menus import setting_row, task_is_cancelling
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


async def toggle_search(owner: "VinApp") -> str | None:
    if owner.search_key is None:
        key = await owner.ask_search_key()
        if key is None:
            return None
        save_search_key(key)
        owner.search_key = key
        updated = owner.update_settings(search=True)
    else:
        updated = owner.update_settings(search=not owner.settings["search"])
    return f"Web search {'on' if updated['search'] else 'off'}"


async def change_search_key(owner: "VinApp") -> str | None:
    if owner.search_key is None:
        return None
    try:
        choice = await owner.choose_inline(
            "Web search key", [("replace", "Replace key"), ("remove", "Remove key")]
        )
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    if choice == "remove":
        delete_search_key()
        owner.search_key = None
        _ = owner.update_settings(search=False)
        return "Web search key removed · web search off"
    key = await owner.ask_search_key()
    if key is None:
        return None
    save_search_key(key)
    owner.search_key = key
    return f"Web search key replaced · {mask_key(key)}"


async def toggle_fetch(owner: "VinApp") -> str:
    updated = owner.update_settings(fetch=not owner.settings["fetch"])
    return f"Web fetch {'on' if updated['fetch'] else 'off'}"


async def change_subagent_settings(owner: "VinApp") -> str | None:
    settings = owner.settings
    rows: list[tuple[str, str]] = [
        (
            "enabled",
            setting_row("Subagents", "on" if settings["subagents"] else "off"),
        ),
        (
            "budget",
            setting_row("Request limit", f"{settings['subagent_budget']} ›"),
        ),
    ]
    try:
        choice = await owner.choose_inline("Subagents", rows)
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    if choice == "enabled":
        updated = owner.update_settings(subagents=not settings["subagents"])
    else:
        try:
            value = await owner.choose_inline(
                "Subagent request limit",
                [(str(number), str(number)) for number in range(1, 33)],
                default=str(settings["subagent_budget"]),
            )
        except asyncio.CancelledError:
            if task_is_cancelling():
                raise
            return None

        updated = owner.update_settings(subagent_budget=int(value))
    return f"Subagents {'on' if updated['subagents'] else 'off'} · request limit {updated['subagent_budget']}"


async def change_web_search_settings(owner: "VinApp") -> str | None:
    rows: list[tuple[str, str]] = [
        ("enabled", setting_row("Web search", "on" if owner.search_on() else "off")),
    ]
    if owner.search_key is not None:
        rows.append(("key", setting_row("Key", f"{mask_key(owner.search_key)} ›")))
    try:
        choice = await owner.choose_inline("Web search", rows)
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    if choice == "key":
        return await owner.change_search_key()
    return await owner.toggle_search()


async def change_settings(owner: "VinApp") -> str | None:
    try:
        choice = await owner.choose_inline(
            "Settings",
            [
                ("subagents", setting_row("Subagents", "›")),
                ("search", setting_row("Web search", "›")),
                (
                    "fetch",
                    setting_row(
                        "Web fetch",
                        "on" if owner.settings["fetch"] else "off",
                    ),
                ),
            ],
        )
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    if choice == "search":
        return await owner.change_web_search_settings()
    elif choice == "subagents":
        return await owner.change_subagent_settings()
    elif choice == "fetch":
        return await owner.toggle_fetch()
    return None
