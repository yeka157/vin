"""Provider labels and recognized terminal commands."""

PROVIDER_LABELS = {"chatgpt": "ChatGPT"}
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
}


def is_command(text):
    parts = text.split(maxsplit=1)
    return text in COMMANDS or bool(parts and parts[0] in ("/resume", "/rename"))
