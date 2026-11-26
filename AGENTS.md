# AGENTS.md

Vin is a terminal coding agent in Python: a prompt-toolkit/Rich TUI, ChatGPT sign-in over OAuth or an OpenRouter API key, and a provider-neutral tool loop with four tools (`read`, `write`, `edit`, `bash`). This file is the single source of agent guidance for the repo; `CLAUDE.md` imports it.

## Where things are

- `src/vin/agent.py`: the provider-neutral tool loop (`run_agent_turn`), tool execution and compaction pruning. It never imports `openai`; a provider hands it a `request(history)` function.
- `src/vin/provider/shared.py`: what every provider shares. `stream_reply` checks sign-in and model, connects, runs the agent loop and maps API errors to user messages. A provider passes only `connect`, its one-request `complete` function and any status codes with their own wording.
- `src/vin/provider/chatgpt.py` and `src/vin/provider/openrouter.py`: the provider-specific parts: the wire format (Responses or Chat Completions), sign-in details and the model list.
- `src/vin/provider/events.py`: the event queue between the worker thread and the UI.
- `src/vin/tools.py`: the four tools and `TOOL_REGISTRY`.
- `src/vin/tui/`: the app (`app.py` holds `VinApp`, `stream_reply` and the command receiver), key bindings, layout and transcript rendering.
- `src/vin/sessions/`: saving and resuming conversations.
- `test/`: offline `unittest` suite. `*_helpers.py` hold shared fixtures and typed helpers.

## Commands

Run from the repo root; running inside `test/` breaks the helper imports.

- Tests: `uv run python -m unittest discover -s test`
- Types: `uv run basedpyright` (strict, `reportAny` is an error)
- Whitespace: `git diff --check`

Done means all three are clean. Paste the summary line of each as evidence.

## Code standards

- Strict typing everywhere, `src` and `test`. Narrow untyped JSON with the guards in `vin.types` (`is_record`, `is_list`, `json_value`) instead of `Any` or loose casts.
- Python 3.14 syntax: `type` aliases, `TypedDict`, `X | None`.
- Model runs happen on a worker thread and talk to the UI only through `ReplyEvents`. Cancellation is a `threading.Event`.
- `openai` and `vin.tools` are imported lazily inside functions to keep startup fast. Keep new heavy imports lazy too.
- User-facing error strings are part of the contract: tests assert on them, so change them deliberately.
- Reach for the standard library and existing helpers before adding a dependency.

## Tests

- Offline only: HTTP is mocked, no real account or network.
- A new test must fail without the change it covers. Check by stashing or reverting the source change and running it.
- Each feature covers the unhappy paths: empty, invalid, failure, cancellation, busy/repeat.

## Git

- Conventional commits, one logical change each. Stage paths by name.
- No co-author or attribution lines in commit messages.
- Push only when Kevin asks.
- Local development notes are gitignored on purpose and never committed: `BACKLOG.md`, `CLAUDE.local.md`, `docs/`, `engineering/`, `prompt/`, `ux/`.
- The repo is a public portfolio and has no license on purpose. Leave it unlicensed.

## Feature workflow

1. Write or reuse a brief at `docs/features/<slug>.md` before building a feature; get Kevin's approval.
2. Build in small committed steps, each with green checks.
3. Record offline evidence and live acceptance separately. Live runs need Kevin's ChatGPT sign-in, so he does them; mark them pending until he reports results.
4. Keep `docs/HANDOFF.md` current: add a dated section at the top with what was committed and the exact next step. `BACKLOG.md` IDs are stable and referenced from the handoff.

## Agent skills

### Issue tracker

Issues are bullets in the local `BACKLOG.md` (not GitHub Issues). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), written as `[role]` tags on backlog items. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Which skill when

Matt Pocock's skills (`mattpocock-skills` plugin), mapped to this repo:

- **Shaping a feature**: `/grill-with-docs` to pressure-test the idea and record terms and decisions, then write the brief in `docs/features/`.
- **Splitting work**: `/to-tickets` turns an approved brief into backlog bullets; `/wayfinder` for work bigger than one session.
- **Building**: `/tdd` when a behavior can be pinned by a test first.
- **Bugs**: `/diagnosing-bugs` before proposing a fix.
- **Reviewing**: `/code-review` since a commit; it checks the code standards above and the brief.
- **Design**: `/improve-codebase-architecture` or `/codebase-design` for module boundaries; `/domain-modeling` for `GLOSSARY.md` and ADRs.
- **Unsure which**: `/ask-matt`.

Handoffs go in `docs/HANDOFF.md`, not the temporary directory `/handoff` uses.