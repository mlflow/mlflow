# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**For contribution guidelines, code standards, and additional development information not covered here, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).**

## Knowledge Cutoff Note

Claude's training data has a knowledge cutoff that may lag behind the current date. When reviewing documentation or code that references AI models, be aware that newer models may exist beyond the cutoff. Do not flag model names as "speculative" or "non-existent". Trust the documentation authors' knowledge of current model availability.

Example: If documentation references "GPT-5" or "Claude 4.5", do not suggest changing these to older model names just because they are unfamiliar.

## Code Style Principles

- Use top-level imports (only use lazy imports when necessary)
- Only add docstrings in tests when they provide additional context
- Only add comments that explain non-obvious logic or provide additional context

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
