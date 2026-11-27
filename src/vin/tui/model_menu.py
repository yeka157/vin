"""Loading, choosing and remembering models for both providers."""

import asyncio
from collections.abc import Coroutine, Sequence
from time import monotonic
from typing import TYPE_CHECKING

from rich.text import Text

from vin.provider.chatgpt import list_chatgpt_models
from vin.provider.model_cache import save_cached_chatgpt_models
from vin.provider.openrouter import (
    list_openrouter_models,
    save_openrouter_whitelist,
)
from vin.provider.preferences import load_model_preference, save_model_preference
from vin.tui.constants import PROVIDER_LABELS
from vin.tui.menus import choose_chatgpt_model, task_is_cancelling
from vin.tui.model_picker import ModelPicker
from vin.types.models import ModelMetadata, ModelPreference, Provider
from vin.types.transcript import TranscriptEntry

if TYPE_CHECKING:
    from vin.tui.app import VinApp


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


async def wait_for_models(owner: "VinApp", load: Coroutine[object, object, list[ModelMetadata]]
) -> list[ModelMetadata] | None:
    owner.loading_model = True
    task = asyncio.create_task(load)
    owner.pending_model_load = task
    owner.app.invalidate()
    try:
        return await task
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    finally:
        owner.pending_model_load = None
        owner.loading_model = False
        owner.app.invalidate()


def select_model(owner: "VinApp", model: ModelMetadata) -> str:
    save_model_preference(model["provider"], model["slug"])
    owner.selected_provider = model["provider"]
    owner.selected_model = model["slug"]
    owner.last_usage = None
    return f"Model: {PROVIDER_LABELS[model['provider']]} · {model['display_name']}"


def save_whitelist(owner: "VinApp", models: list[ModelMetadata]) -> None:
    save_openrouter_whitelist(models)
    owner.set_models("openrouter", models)


def auto_select_model(owner: "VinApp") -> str | None:
    if owner.selected_model is None and len(owner.models) == 1:
        return owner.select_model(owner.models[0])
    return None


async def pick_models(owner: "VinApp",
    title: str,
    models: list[ModelMetadata],
    *,
    multi: bool,
    action: str,
    current: tuple[str, str] | None = None,
    search: str = "",
) -> list[ModelMetadata] | None:
    if owner.choice_future is not None:
        raise RuntimeError("A choice is already open")
    picker = ModelPicker(
        title, models, multi=multi, action=action, current=current, search=search
    )
    owner.picker = picker
    owner.choice_menu = picker.container
    owner.choice_future = asyncio.get_running_loop().create_future()
    try:
        owner.session.layout.focus(picker.search_window)
        owner.app.invalidate()
        await owner.choice_future
        return picker.chosen()
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        return None
    finally:
        owner.picker = None
        owner.choice_menu = None
        owner.choice_future = None
        owner.session.layout.focus(owner.input_window)
        owner.app.invalidate()


async def change_model(owner: "VinApp") -> str | None:
    chatgpt = bool(owner.credentials and owner.credentials.get("access_token"))
    if not chatgpt and owner.openrouter is None:
        return "Sign in with /login before choosing a model."
    if chatgpt and not any(model["provider"] == "chatgpt" for model in owner.models):
        models = await owner.wait_for_models(owner.discover_chatgpt_models())
        if models is None:
            return None
        owner.set_models("chatgpt", models)
    if not owner.models:
        return "No models yet. Use /model add to add an OpenRouter model."
    picked = await owner.pick_models(
        "Choose a model",
        owner.models,
        multi=False,
        action="choose",
        current=(owner.selected_provider or "", owner.selected_model or ""),
    )
    if not picked:
        return None
    chosen = picked[0]
    if (
        chosen["provider"] == owner.selected_provider
        and chosen["slug"] == owner.selected_model
    ):
        return None
    return owner.select_model(chosen)


async def add_openrouter_model(owner: "VinApp", query: str) -> str | None:
    if owner.openrouter is None:
        return "Sign in to OpenRouter with /login before adding a model."
    catalog = await owner.wait_for_models(asyncio.to_thread(list_openrouter_models))
    if catalog is None:
        return None
    whitelist = [
        model for model in owner.models if model["provider"] == "openrouter"
    ]
    fresh = {model["slug"]: model for model in catalog}
    refreshed = [fresh.get(model["slug"], model) for model in whitelist]
    if refreshed != whitelist:
        owner.save_whitelist(refreshed)
        whitelist = refreshed
    added = {model["slug"] for model in whitelist}
    matches = [model for model in catalog if model["slug"] not in added]
    if not matches:
        return "Every OpenRouter model with tool support is already added."
    picked = await owner.pick_models(
        "Add OpenRouter models", matches, multi=True, action="add", search=query
    )
    if not picked:
        return None
    owner.save_whitelist(whitelist + picked)
    message = f"Added OpenRouter · {', '.join(m['display_name'] for m in picked)}."
    selected = owner.auto_select_model()
    return (
        f"{message} {selected}" if selected else f"{message} Use /model to switch."
    )


async def remove_openrouter_model(owner: "VinApp") -> str | None:
    whitelist = [
        model for model in owner.models if model["provider"] == "openrouter"
    ]
    if not whitelist:
        return "No OpenRouter models to remove."
    picked = await owner.pick_models(
        "Remove OpenRouter models", whitelist, multi=True, action="remove"
    )
    if not picked:
        return None
    removed = {m["slug"] for m in picked}
    owner.save_whitelist([m for m in whitelist if m["slug"] not in removed])
    message = (
        f"Removed OpenRouter · {', '.join(m['display_name'] for m in picked)}."
    )
    if owner.selected_provider == "openrouter" and owner.selected_model in removed:
        owner.selected_model = None
        owner.last_usage = None
        outcome = owner.auto_select_model()
        if not outcome and owner.models:
            outcome = await owner.change_model()
        if outcome:
            return f"{message} {outcome}"
    return message


async def discover_chatgpt_models(owner: "VinApp") -> list[ModelMetadata]:
    credentials = await owner.prepare_request_credentials()

    models = await asyncio.to_thread(list_chatgpt_models, credentials)

    save_cached_chatgpt_models(credentials, models)
    return models


async def load_initial_model(owner: "VinApp") -> None:
    models = await owner.discover_chatgpt_models()
    provider_name = PROVIDER_LABELS["chatgpt"]
    if not models:
        raise RuntimeError(
            f"No {provider_name} models are available for this account."
        )

    preference: ModelPreference | None = get_chatgpt_model_preference(
        owner.transcript, models
    )
    if preference is None:
        selected_model = await owner.choose_inline(
            f"Choose a {provider_name} model",
            [(model["slug"], model["display_name"]) for model in models],
        )
        preference = {"provider": "chatgpt", "model": selected_model}
        save_model_preference(preference["provider"], preference["model"])
    owner.set_models("chatgpt", models)
    owner.selected_provider = preference["provider"]
    owner.selected_model = preference["model"]


def set_models(owner: "VinApp", provider: Provider, models: list[ModelMetadata]) -> None:
    owner.models = [
        model for model in owner.models if model["provider"] != provider
    ] + models


def model_indicator_text(owner: "VinApp") -> str:
    if monotonic() < owner.exit_armed_until:
        return " Press again to exit"
    subagent = owner.viewed_subagent()
    if subagent is not None:
        return (
            f" Viewing task · {subagent['description']} · read-only · Esc to return"
        )
    if owner.selected_model:
        assert owner.selected_provider is not None
        return (
            f" {PROVIDER_LABELS[owner.selected_provider]} · "
            + model_display_name(owner.models, owner.selected_model)
        )
    if owner.selected_provider == "openrouter":
        return " OpenRouter · No model selected"
    return (
        " No model selected · /login"
        if owner.credentials and owner.credentials.get("access_token")
        else " Not signed in · /login"
    )
