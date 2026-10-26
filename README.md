# Vin

Vin is a terminal coding assistant that can explore a project, explain code, edit files, and run commands. Sign in with ChatGPT, choose a model, and describe what you want to do.

![Vin reading a project and summarizing its structure](assets/demo.gif)

Vin is in early development. Development and manual testing currently take place on macOS. Linux has not been verified, and Windows is not supported.

> [!WARNING]
> Vin edits files and runs Bash commands without asking first. Commands run with your operating-system permissions and can reach paths outside the project folder. Start Vin in a repository with a clean `git status` so you can review its changes with `git diff` and undo them with `git restore`.

## How it works

Vin signs in with your ChatGPT account through the browser (OAuth with PKCE), so it needs no OpenAI API key. Each message you send starts a loop: Vin sends the conversation to the model, the model can call tools, Vin runs them and returns the results, and the loop repeats until the model answers in text. One message allows up to 16 model requests and up to 8 tool calls per request.

The model has four tools:

| Tool | What it does |
| --- | --- |
| `read` | Read a file, 200 lines at a time by default |
| `write` | Create or overwrite a file |
| `edit` | Replace an exact piece of text in a file |
| `bash` | Run a shell command, with a 30-second default timeout |

`read`, `write` and `edit` only accept paths inside the folder where you launched Vin. `bash` has no such limit.

## Install

You need Git, an internet connection, and Python **3.14.3 or newer**. The instructions below use [uv](https://docs.astral.sh/uv/getting-started/installation/), which can install Python and keep Vin's dependencies in a separate environment.

Clone the repository and install Vin:

```sh
git clone https://github.com/yeka157/vin.git
cd vin
uv python install 3.14.3
uv tool install --python 3.14.3 .
```

The final `.` means the folder you just cloned. Once installed, the `vin` command works from any folder without activating an environment.

If the command is not found, run `uv tool update-shell`, then open a new terminal. See the [uv tool guide](https://docs.astral.sh/uv/guides/tools/) for details.

### Install with pip instead

With Python 3.14.3 or newer installed, open the cloned repository and run:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Activate this environment again when opening a new terminal. You can change to another project folder after activation and run `vin` there.

## Start a conversation

Open the project you want to work on, then start Vin:

```sh
cd /path/to/your/project
vin
```

On the first launch, follow the browser sign-in steps and choose a ChatGPT model. No OpenAI API key or `.env` file is required. The models available depend on your signed-in account. Vin remembers your account and selected model for later launches.

Describe your goal in ordinary language. For example:

> Explain how this project is organized.

> Review this repository's cancellation handling for bugs.

> Fix the failing tests and explain what changed.

Vin shows its progress and tool activity as it works. Completed tool groups collapse into a summary; click the summary to expand it, and click again to close it. File edits include a diff.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | Show available commands |
| `/user` | Show your account status |
| `/model` | Choose a model and remember it for future launches |
| `/login` | Sign in with ChatGPT |
| `/logout` | Sign out and clear local tokens |
| `/clear` | Start a fresh conversation without deleting saved conversations |
| `/resume` | Choose a saved conversation for the current project folder |
| `/resume <title-or-id>` | Resume a conversation by its exact title or full session ID |
| `/rename <name>` | Rename the current conversation |
| `/exit` or `/quit` | Exit Vin |

## Keyboard and navigation

| Key or action | What it does |
| --- | --- |
| Enter | Send your message |
| Escape, then Enter | Insert a newline |
| Page Up / Page Down or mouse wheel | Scroll through the conversation |
| Ctrl+End | Return to the latest output |
| Escape in a picker or panel | Cancel or close it |
| Tab on a session suggestion | Fill in its session ID; press Enter to resume |
| Ctrl+C during a response | Interrupt the current response |
| Ctrl+C while idle | Exit Vin |
| Ctrl+D on empty input | Exit Vin |

Shift+Enter behavior depends on your terminal. Use Escape followed by Enter for a newline.

You can keep typing while Vin works. Sending another message adds it to a queue, and Vin handles queued messages in order. Interrupting a response with Ctrl+C also lets the next queued message start. If a rate limit pauses the queue, press Enter to resume.

Scrolling up keeps your reading position while new output arrives. Return to the bottom or press Ctrl+End to follow the latest output again.

## Save and resume conversations

Conversations save automatically. Every launch starts fresh; use `/resume` to continue an earlier conversation.

Give a conversation a recognizable name:

```text
/rename cancellation review
```

Later, start Vin from the same project folder and enter:

```text
/resume cancellation review
```

You can also type `/resume ` followed by part of a title to see suggestions. Use the arrow keys to select one, press Tab to fill in its full ID, then press Enter to resume. If multiple conversations have the same title, use the picker or a full ID.

Session lists belong to the folder where you launched Vin. Starting from a subfolder gives you a different list. `/clear` starts fresh while keeping saved conversations available.

Resuming an interrupted conversation restores its saved progress without rerunning old tools. Interrupting a response does not undo file changes that already completed.

## Local data

| Data | Location |
| --- | --- |
| Saved conversations | `~/.local/share/vin/sessions/` |
| Account credentials and model preference | `~/.config/vin/` |
| Cached model list | `~/.cache/vin/chatgpt/` |

Saved conversations can contain prompts, responses, file contents and command output. Credential files are stored locally with owner-only permissions; they are not encrypted.

## Troubleshooting

**`vin: command not found`**

For a uv installation, run `uv tool update-shell` and open a new terminal. For a pip installation, activate the environment where you installed Vin.

**Sign-in or model errors**

Use `/login` to sign in again, or `/model` to choose an available model. Check the error shown in the conversation and your internet connection.

**A response pauses and then arrives in a burst**

Some long responses can pause before delivering more text. The cause is still unresolved.

**A long conversation stops accepting requests**

Automatic context compaction is not implemented yet. Use `/clear` and describe the task again in a fresh conversation. The old session remains saved, but resuming it does not reduce its size.

## Update or uninstall

For a uv installation, open your cloned Vin repository, get the latest source, and reinstall:

```sh
git pull --ff-only
uv tool install --reinstall --python 3.14.3 .
```

For a pip installation, activate its environment and run these commands from the cloned repository:

```sh
git pull --ff-only
python -m pip install --upgrade .
```

To uninstall:

```sh
# uv installation
uv tool uninstall vin

# pip installation, with its environment active
python -m pip uninstall vin
```

Uninstalling keeps saved conversations, preferences and credentials. Use `/logout` before uninstalling to clear local tokens.
