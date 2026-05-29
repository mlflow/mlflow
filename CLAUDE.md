# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Knowledge Cutoff Note

Claude's training data may lag behind current releases. When reviewing docs or code, don't flag unfamiliar names as speculative or non-existent. Assume the authors are referencing newer, valid resources (e.g., model names like GPT-5, GitHub runner types like ubuntu-slim, library versions, etc.).

## Code Style Principles

- Use top-level imports (only use lazy imports when necessary)
- Only add docstrings in tests when they provide additional context
- Only add comments that explain non-obvious logic or provide additional context
- When touching the SQLAlchemy tracking store, keep all workspace-aware paths and validations intact; never drop workspace plumbing even if the change focuses on single-tenant behavior
- New functionality in the tracking layer should be mirrored by workspace-aware tests (e.g., add workspace variants in `tests/store/tracking/test_sqlalchemy_store_workspace.py` when applicable)

## Repository Overview

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for:

- Experiment tracking
- Model versioning and deployment
- LLM observability and tracing
- Model evaluation
- Prompt management

## Quick Start: Development Server

### Start the Full Development Environment (Recommended)

```bash
# Start both MLflow backend and React frontend dev servers
# (The script will automatically clean up any existing servers)
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &

# Monitor the logs
tail -f /tmp/mlflow-dev-server.log

# Servers will be available at:
# - MLflow backend: http://localhost:5000
# - React frontend: http://localhost:3000
```

This uses `uv` (fast Python package manager) to automatically manage dependencies and run the development environment.

> **Note — single environment only.** On startup `dev/run-dev-server.sh` kills _all_ running MLflow `--dev` backends and `yarn start` frontends, matching by command line rather than by directory. That is fine when you run one dev environment at a time, but it will tear down dev servers started from **other git worktrees**. To run multiple worktrees at once, see [Running Multiple Worktrees Concurrently](#running-multiple-worktrees-concurrently) below.

## Debugging

For debugging errors, enable debug logging (must be set before importing mlflow):

```bash
export MLFLOW_LOGGING_LEVEL=DEBUG
```

### Start Development Server with Databricks Backend

To run the MLflow dev server that proxies requests to a Databricks workspace:

```bash
# IMPORTANT: All four environment variables below are REQUIRED for proper Databricks backend operation
# Set them in this exact order:
export DATABRICKS_HOST="https://your-workspace.databricks.com"  # Your Databricks workspace URL
export DATABRICKS_TOKEN="your-databricks-token"                # Your Databricks personal access token
export MLFLOW_TRACKING_URI="databricks"                        # Must be set to "databricks"
export MLFLOW_REGISTRY_URI="databricks-uc"                     # Use "databricks-uc" for Unity Catalog, or "databricks" for workspace model registry

# Start the dev server with these environment variables
# (The script will automatically clean up any existing servers)
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &

# Monitor the logs
tail -f /tmp/mlflow-dev-server.log

# The MLflow server will now proxy tracking and model registry requests to Databricks
# Access the UI at http://localhost:3000 to see your Databricks experiments and models
```

**Note**: The MLflow server acts as a proxy, forwarding API requests to your Databricks workspace while serving the local React frontend. This allows you to develop and test UI changes against real Databricks data.

### Running Multiple Worktrees Concurrently

`dev/run-dev-server.sh` kills sibling dev servers on startup (see the note above), so it cannot run two environments at once. To work in several git worktrees simultaneously, skip the wrapper and start the two processes directly, and neither process kills the other. Give each worktree its own port pair:

```bash
# Pick a unique port pair per worktree, e.g. A=5000/3000, B=5001/3001
BACKEND_PORT=5001
FRONTEND_PORT=3001

# Terminal 1 — backend (./mlflow.db, ./mlruns, ./mlartifacts are per-worktree)
uv run mlflow server --port "$BACKEND_PORT" --dev

# Terminal 2 — frontend
cd mlflow/server/js
yarn install   # first time per worktree; node_modules is not shared across worktrees
PORT="$FRONTEND_PORT" \
MLFLOW_PROXY="http://127.0.0.1:$BACKEND_PORT" \
MLFLOW_DEV_PROXY_MODE=false \
yarn start
# open http://localhost:$FRONTEND_PORT
```

Nuances:

- `--dev` gives the backend autoreload + debug logging — it is the only thing the wrapper adds over a bare `mlflow server`.
- `MLFLOW_PROXY` must point at _this_ worktree's backend. If it doesn't, the UI silently talks to a different (or the default `:5000`) backend and shows the wrong worktree's data — this is the most common mistake.
- `MLFLOW_DEV_PROXY_MODE=false` tells craco to actually use the proxy server.
- The backend store and artifacts are resolved relative to the current directory (`./mlflow.db`, `./mlruns`, `./mlartifacts`), so each worktree root is isolated automatically. Pointing `MLFLOW_TRACKING_URI`/`MLFLOW_BACKEND_STORE_URI` at an absolute path defeats this isolation.
- This is a dev-only workflow. Production builds the frontend (`yarn build`) and serves it from the backend, with no separate JS server and no proxy.

## Development Commands

### Offline / No-Network Usage

If PyPI is unreachable, add `--frozen` to `uv run` commands that should use the existing `uv.lock` as-is without modifying the environment. This works when the required dependencies are already installed or available in the local cache:

```bash
uv run --frozen pytest tests/
```

### Testing

```bash
# First-time setup: Install test dependencies
uv sync
uv pip install -r requirements/test-requirements.txt

# Run Python tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_version.py

# Run tests with specific package versions
uv run --with 'abc==1.2.3,xyz==4.5.6' pytest tests/test_version.py

# Run tests with optional dependencies/extras
uv run --with transformers pytest tests/transformers
uv run --extra gateway pytest tests/gateway
```

### Code Quality

```bash
# Python linting and formatting with Ruff
uv run ruff check . --fix         # Lint with auto-fix
uv run ruff format .              # Format code

# Custom MLflow linting with Clint
uv run clint .                    # Run MLflow custom linter

# Check for MLflow spelling typos
uv run bash dev/mlflow-typo.sh .
```

### Special Testing

```bash
# Run tests with minimal dependencies (skinny client)
uv run bash dev/run-python-skinny-tests.sh
```

### Documentation

```bash
# Build documentation site (needs gateway extras for API doc generation)
uv run --all-extras bash dev/build-docs.sh --build-api-docs

# Build with R docs included
uv run --all-extras bash dev/build-docs.sh --build-api-docs --with-r-docs

# Serve documentation locally (after building)
cd docs && npm run serve --port 8080
```

## Important Files

- `pyproject.toml`: Package configuration and tool settings
- `.python-version`: Minimum Python version (3.10)
- `requirements/`: Dependency specifications
- `mlflow/ml-package-versions.yml`: Supported ML framework versions

## Common Development Tasks

### Modifying the UI

For frontend development (React, TypeScript, UI components), see [mlflow/server/js/CLAUDE.md](./mlflow/server/js/CLAUDE.md) which covers:

- Development server setup with hot reload
- Available yarn scripts (testing, linting, formatting, type checking)
- UI components and design system usage
- Project structure and best practices

## Git Workflow

### Committing Changes

When committing changes:

- DCO sign-off: All commits MUST use the `-s` flag (otherwise CI will reject them)
- Co-Authored-By trailer: Include when Claude Code authors or co-authors changes
- Pre-commit hooks: Run before committing (see [Pre-commit Hooks](#pre-commit-hooks))

```bash
# Commit with required DCO sign-off
git commit -s -m "Your commit message

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push your changes
git push origin <your-branch>
```

### Creating Pull Requests

When creating pull requests, read the instructions at the top of [the PR template](./.github/pull_request_template.md) and follow them carefully.

### Checking CI Status

Use GitHub CLI to check for failing CI:

```bash
# Check workflow runs for current branch
gh run list --branch $(git branch --show-current)

# View details of a specific run
gh run view <run-id>

# Watch a run in progress
gh run watch
```

## Pre-commit Hooks

The repository uses pre-commit for code quality. Install hooks with:

```bash
uv run pre-commit install --install-hooks
uv run pre-commit run install-bin -a -v
```

Run pre-commit manually:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on specific files
uv run pre-commit run --files path/to/file.py

# Run a specific hook
uv run pre-commit run ruff --all-files
```

This runs Ruff, typos checker, and other tools automatically before commits.
