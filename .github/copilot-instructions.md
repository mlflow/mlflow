<!--
This file provides compact, actionable guidance for AI coding agents (Copilot-style)
to be immediately productive in the MLflow repository. Keep it short and concrete.
-->

# Quick agent instructions for working in the MLflow repo

- Big picture: MLflow is a multi-language monorepo (Python core in `mlflow/`,
  React frontend in `mlflow/server/js/`, TypeScript libs under `libs/`, and
  many examples under `examples/`). The `dev/` scripts orchestrate common
  developer workflows. Key config: `pyproject.toml`, `requirements/`,
  and `mlflow/ml-package-versions.yml`.

- Typical dev flow (use `uv` wrapper for reproducible environments):
  - Start full dev server: `uv run bash dev/run-dev-server.sh` (backend + frontend).
  - Run Python tests: `uv run pytest tests/` or `uv run pytest tests/path/to/test.py`.
  - Install lint/test deps: `uv sync` then `uv pip install -r requirements/test-requirements.txt`.

- Linting & pre-commit: repo uses `pre-commit`, `ruff`, and a custom linter `clint`.
  Commands (use `uv`):
  - `uv run --only-group lint ruff check . --fix`
  - `uv run --only-group lint clint .`
  - Install hooks: `uv run --only-group lint pre-commit install --install-hooks`
  - When modifying `mlflow/server/js/` run `yarn --cwd mlflow/server/js check-all`.

- Testing variations you should know about:
  - Extra deps: `uv run --with transformers pytest tests/transformers`
  - Skinny client tests: `uv run bash dev/run-python-skinny-tests.sh`
  - JS tests: `yarn --cwd mlflow/server/js test`.

- Project conventions and gotchas (do not invent these):
  - Always sign commits with DCO: `git commit -s -m "message"`.
  - Pre-commit runs only on changed files in a PR by default; CI enforces this.
  - Some hooks rely on external tools installed to `bin/` via `uv run --only-group lint bin/install.py`.
  - Use `uv` wrapper to run commands to ensure consistent virtualenv / deps.

- Code structure examples to reference:
  - Backend CLI and server entrypoints: `mlflow/cli/__init__.py`, `dev/run-dev-server.sh`.
  - Security and server middleware: `mlflow/server/security.py`, `mlflow/server/fastapi_security.py`.
  - Frontend app: `mlflow/server/js/` (run `yarn` there for JS dev tasks).

- When creating or editing tests:
  - Prefer pytest-style tests under `tests/`.
  - Keep test dependencies minimal when possible; use `--with` flags to opt into heavy extras.

- Integration & infra notes:
  - The dev server can proxy to Databricks if `DATABRICKS_HOST` and related env vars are set (see `CLAUDE.md`).
  - Many CI and developer workflows are orchestrated with `uv` and `dev/` scripts; inspect `.github/workflows/` for automation cues.

- Security & server runtime: the server enables security middleware by default. Use `mlflow server --help` for flags like
  `--allowed-hosts`, `--cors-allowed-origins`, and `--x-frame-options` (see `mlflow/server/AGENTS.md`).

- How to be helpful as an agent:
  - Make small, self-contained edits and run the relevant tests locally using `uv`.
  - When touching JS, run `yarn check-all` before proposing changes.
  - Add or update tests for behavior changes. Reference nearby tests in `tests/` for style.
  - Don't modify CI or release files without explicit instruction; these are sensitive.

If any section is unclear or you'd like more examples (e.g., a minimal test + run example or how to run a frontend-only change), tell me which area to expand.
