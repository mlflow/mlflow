# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**For contribution guidelines, code standards, and additional development information not covered here, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).**

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
# Kill any existing servers
pkill -f "mlflow server" || true; pkill -f "yarn start" || true

# Start both MLflow backend and React frontend dev servers
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &

# Monitor the logs
tail -f /tmp/mlflow-dev-server.log

# Servers will be available at:
# - MLflow backend: http://localhost:5000
# - React frontend: http://localhost:3000
```

This uses `uv` (fast Python package manager) to automatically manage dependencies and run the development environment.

## Development Commands

### Testing

```bash
# First-time setup: Install test dependencies
uv sync
uv pip install -r requirements/test-requirements.txt

# Run Python tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_version.py

# Run JavaScript tests
yarn --cwd mlflow/server/js test
```

### Code Quality

```bash
# Python linting and formatting with Ruff
uv run ruff check . --fix         # Lint with auto-fix
uv run ruff format .              # Format code

# Check for MLflow spelling typos
uv run bash dev/mlflow-typo.sh .

# JavaScript linting and formatting
yarn --cwd mlflow/server/js lint
yarn --cwd mlflow/server/js prettier:check
yarn --cwd mlflow/server/js prettier:fix

# Type checking
yarn --cwd mlflow/server/js type-check

# Run all checks
yarn --cwd mlflow/server/js check-all
```

### Special Testing

```bash
# Run tests with minimal dependencies (skinny client)
uv run bash dev/run-python-skinny-tests.sh

# Test in Docker container
uv run bash dev/run-test-container.sh
```

### Documentation

```bash
# Build documentation site (needs gateway extras for API doc generation)
uv run --all-extras bash dev/build-docs.sh --build-api-docs

# Build with R docs included
uv run --all-extras bash dev/build-docs.sh --build-api-docs --with-r-docs

# Serve documentation locally (after building)
cd docs && yarn serve --port 8080
```

## Important Files

- `pyproject.toml`: Package configuration and tool settings
- `.python-version`: Minimum Python version (3.10)
- `requirements/`: Dependency specifications
- `mlflow/ml-package-versions.yml`: Supported ML framework versions
- `.github/copilot-instructions.md`: Additional coding guidelines

## Common Development Tasks

### Modifying the UI

See `mlflow/server/js/` for frontend development.

## Git Workflow

### Committing Changes

**IMPORTANT**: After making your commits, run pre-commit hooks on your PR changes to ensure code quality:

```bash
# Make your commit first (with DCO sign-off)
git commit -s -m "Your commit message"

# Then check all files changed in your PR
uv run pre-commit run --from-ref origin/master --to-ref HEAD

# Fix any issues and amend your commit if needed
git add <fixed files>
git commit --amend -s

# Re-run pre-commit to verify fixes
uv run pre-commit run --from-ref origin/master --to-ref HEAD

# Only push once all checks pass
git push origin <your-branch>
```

This workflow ensures you only check files you've actually modified in your PR, avoiding false positives from unrelated files.

**IMPORTANT**: You MUST sign all commits with DCO (Developer Certificate of Origin). Always use the `-s` flag:

```bash
# REQUIRED: Always use -s flag when committing
git commit -s -m "Your commit message"

# This will NOT work - missing -s flag
# git commit -m "Your commit message"  ❌
```

Commits without DCO sign-off will be rejected by CI.

**Frontend Changes**: If your PR touches any code in `mlflow/server/js/`, you MUST run `yarn check-all` before committing:

```bash
yarn --cwd mlflow/server/js check-all
```

### Creating Pull Requests

Follow [the PR template](./.github/pull_request_template.md) when creating pull requests. The template will automatically appear when you create a PR on GitHub.

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
```

Run pre-commit manually:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on all files, skipping hooks that require external tools
SKIP=taplo,typos,conftest uv run pre-commit run --all-files

# Run on specific files
uv run pre-commit run --files path/to/file.py

# Run a specific hook
uv run pre-commit run ruff --all-files
```

This runs Ruff, typos checker, and other tools automatically before commits.

**Note about external tools**: Some pre-commit hooks require external tools that aren't Python packages:

- `taplo` - TOML formatter
- `typos` - Spell checker
- `conftest` - Policy testing tool

If you want to run these hooks, **ASK THE USER FIRST** before installing:

```bash
# On Linux/CI: Use the provided install scripts
bash dev/install-taplo.sh
bash dev/install-typos.sh
bash dev/install-conftest.sh

# On macOS: Use cargo or download directly (scripts don't support macOS)
cargo install taplo-cli@0.9.3 --locked
cargo install typos-cli@1.28.0 --locked
# For conftest, download from: https://github.com/open-policy-agent/conftest/releases

# Alternative for macOS (homebrew - may have different versions):
brew install taplo typos-cli conftest
```

These tools are optional. Use `SKIP=taplo,typos,conftest` if they're not installed.

**Note**: If the typos hook fails, you only need to fix typos in code that was changed by your PR, not pre-existing typos in the codebase.
