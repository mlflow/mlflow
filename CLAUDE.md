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
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &

# Monitor the logs
tail -f /tmp/mlflow-dev-server.log

# The MLflow server will now proxy tracking and model registry requests to Databricks
# Access the UI at http://localhost:3000 to see your Databricks experiments and models
```

**Note**: The MLflow server acts as a proxy, forwarding API requests to your Databricks workspace while serving the local React frontend. This allows you to develop and test UI changes against real Databricks data.

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

# Run tests with specific package versions
uv run --with abc==1.2.3 --with xyz==4.5.6 pytest tests/test_version.py

# Run tests with optional dependencies/extras
uv run --with transformers pytest tests/transformers
uv run --extra gateway pytest tests/gateway

# Run JavaScript tests
yarn --cwd mlflow/server/js test
```

**IMPORTANT**: `uv` may fail initially because the environment has not been set up yet. Follow the instructions to set up the environment and then rerun `uv` as needed.

### Code Quality

```bash
# Python linting and formatting with Ruff
uv run --only-group lint ruff check . --fix         # Lint with auto-fix
uv run --only-group lint ruff format .              # Format code

# Custom MLflow linting with Clint
uv run --only-group lint clint .                    # Run MLflow custom linter

# Check for MLflow spelling typos
uv run --only-group lint bash dev/mlflow-typo.sh .

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

## Common Development Tasks

### Modifying the UI

See `mlflow/server/js/` for frontend development.

## Language-Specific Style Guides

- [Python](/dev/guides/python.md)

## Git Workflow

### Committing Changes

**IMPORTANT**: After making your commits, run pre-commit hooks on your PR changes to ensure code quality:

```bash
# Make your commit first (with DCO sign-off)
git commit -s -m "Your commit message"

# Then check all files changed in your PR
uv run --only-group lint pre-commit run --from-ref origin/master --to-ref HEAD

# Fix any issues and amend your commit if needed
git add <fixed files>
git commit --amend -s

# Re-run pre-commit to verify fixes
uv run --only-group lint pre-commit run --from-ref origin/master --to-ref HEAD

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

Follow [the PR template](./.github/pull_request_template.md) when creating pull requests. Remove any unused checkboxes from the template to keep your PR clean and focused.

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
uv run --only-group lint pre-commit install --install-hooks
```

Run pre-commit manually:

```bash
# Run on all files
uv run --only-group lint pre-commit run --all-files

# Run on all files, skipping hooks that require external tools
SKIP=taplo,typos,conftest uv run --only-group lint pre-commit run --all-files

# Run on specific files
uv run --only-group lint pre-commit run --files path/to/file.py

# Run a specific hook
uv run --only-group lint pre-commit run ruff --all-files
```

This runs Ruff, typos checker, and other tools automatically before commits.

**Note about external tools**: Some pre-commit hooks require external tools that aren't Python packages:

- `taplo` - TOML formatter
- `typos` - Spell checker
- `conftest` - Policy testing tool

To install these tools:

```bash
# Install all tools at once (recommended)
uv run --only-group lint bin/install.py
```

This automatically downloads and installs the correct versions of all external tools to the `bin/` directory. The tools work on both Linux and ARM Macs.

These tools are optional. Use `SKIP=taplo,typos,conftest` if they're not installed.

**Note**: If the typos hook fails, you only need to fix typos in code that was changed by your PR, not pre-existing typos in the codebase.
