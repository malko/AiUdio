# Contributing

Thanks for contributing! This repository follows a few simple conventions.

## Commit conventions — gitmoji

Commit messages follow **[gitmoji](https://gitmoji.dev)**:

```
<gitmoji> <type>(<scope>): <subject in imperative present>
```

Examples:

- ✨ `feat(tts): add dynamic batching of requests`
- 🐛 `fix(stt): fix realtime WebSocket passthrough`
- 📝 `docs: document environment variables`
- 🐳 `chore(docker): add the ROCm override`
- ✅ `test(chunking): cover sentence boundaries`
- ♻️ `refactor(engine): factor out model hot-swap`

The repository ships a `commit-msg` hook that checks for a gitmoji. Enable it:

```bash
make hooks      # git config core.hooksPath .githooks ; commit.template .gitmessage
```

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

pytest -q            # tests (no GPU, mocked engine)
ruff check app tests # style
uvicorn app.main:app --reload
```

The real TTS engine is not needed for tests: `tests/conftest.py` provides a `DummyEngine`.
GPU integration tests are run manually on the target machine.

## Language policy

The repository, source code, comments and documentation are written in **English**. The
product UI is multilingual (see `ui/i18n.js`) — add user-facing strings there, not in the
source language of the code.

## Structure

See [`docs/architecture.md`](docs/architecture.md). In short:

- `app/` — FastAPI gateway (engine, scheduler, API, proxies).
- `ui/` — static web UI (with i18n).
- `docker/` — Dockerfiles (TTS gateway, STT) and the STT entrypoint.
- `tests/` — unit and API tests (mocked engine).
- `scripts/` — utilities (model download).

## Pull requests

1. One branch per topic (`feat/...`, `fix/...`).
2. Green tests and a clean `ruff check`.
3. gitmoji commits.
4. Describe the tested hardware (GPU, backend) when relevant.
