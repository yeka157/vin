"""Provider labels and recognized terminal commands."""

PROVIDER_LABELS = {"chatgpt": "ChatGPT", "openrouter": "OpenRouter"}
COMMANDS = {
    "/help",
    "/user",
    "/clear",
    "/resume",
    "/rename",
    "/model",
    "/login",
    "/logout",
    "/quit",
    "/exit",
    "/compact",
    "/settings",
    "/tasks",
}


def is_command(text: str) -> bool:
    parts = text.split(maxsplit=1)
    return text in COMMANDS or bool(parts and parts[0] in ("/resume", "/rename", "/model"))
