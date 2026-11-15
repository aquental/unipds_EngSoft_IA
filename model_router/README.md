# model-router

Python model router for the OpenAI, Anthropic, and Google GenAI APIs.

## Setup

Create a `.env` file in this directory containing your API keys:

```
OPENAI_API_KEY=sk-...
```

`.env` is git-ignored — never commit it.

## Running

`uv run` does **not** load `.env` automatically, so you must pass `--env-file`:

```
uv run --env-file .env python -m model_router.main
```

Without the flag you get `openai.OpenAIError: Missing credentials`.

Alternatively, export the key for your shell session, or set `UV_ENV_FILE=.env`
so the flag is implied for every `uv run`:

```
export UV_ENV_FILE=.env
uv run python -m model_router.main
```

Note: `.vscode/settings.json` sets `python.terminal.useEnvFile`, but that only
applies to VS Code's integrated terminal — not to other terminals such as Warp.