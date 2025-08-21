# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
# Run Python tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_version.py

# Run JavaScript tests
pushd mlflow/server/js && yarn test; popd
```

### Code Quality

```bash
# Python linting and formatting with Ruff
uv run ruff check . --fix         # Lint with auto-fix
uv run ruff format .              # Format code

# Check for MLflow spelling typos
uv run bash dev/mlflow-typo.sh .

# JavaScript linting and formatting
pushd mlflow/server/js && yarn lint; popd
pushd mlflow/server/js && yarn prettier:check; popd
pushd mlflow/server/js && yarn prettier:fix; popd

# Type checking
pushd mlflow/server/js && yarn type-check; popd

# Run all checks
pushd mlflow/server/js && yarn check-all; popd
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

## Architecture Overview

### Core Components

1. **MLflow Tracking** (`mlflow/tracking/`): Records and queries experiments, runs, parameters, metrics, and artifacts
2. **MLflow Models** (`mlflow/models/`): General format for packaging models that can be deployed to various serving environments
3. **MLflow Projects** (`mlflow/projects/`): Packaging format for reproducible runs with dependency management
4. **MLflow Server** (`mlflow/server/`): REST API server with FastAPI backend and React frontend
5. **MLflow Tracing** (`mlflow/tracing/`): LLM observability and debugging tools

### Key Directories

- `mlflow/`: Core Python package
  - `entities/`: Data model definitions
  - `store/`: Backend storage implementations (file, database)
  - `pyfunc/`: Python function flavor for models
  - `server/`: REST API and web UI
    - `js/`: React-based frontend application
  - `protos/`: Protocol buffer definitions
  - `gateway/`: MLflow AI Gateway for LLMs

- `tests/`: Test suite
- `examples/`: Usage examples for various ML frameworks
- `docs/`: Documentation source files

### Frontend Architecture

The UI is a React application located in `mlflow/server/js/` using:

- TypeScript for type safety
- Redux for state management
- Apollo Client for GraphQL
- Ant Design components
- AG-Grid for data tables

### Database Schema

MLflow uses SQLAlchemy with Alembic for migrations. Schema changes require:

1. Modifying models in `mlflow/store/db/`
2. Creating migration with Alembic
3. Testing against supported databases (SQLite, MySQL, PostgreSQL)

## Code Review Guidelines

### Python Standards

- Use type hints for all new functions
- Specify concrete types (e.g., `list[str]` not `list`)
- Handle specific exceptions, not generic `Exception`
- Functions returning >2 elements should use dataclasses
- Follow Google-style docstrings

### Testing Requirements

- Write tests for all new features
- Use pytest fixtures for test setup
- Mock external dependencies
- Test both success and error cases

### Backwards Compatibility

- Maintain API compatibility in minor releases
- Deprecate features before removal
- Document breaking changes in CHANGELOG.md

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
pushd mlflow/server/js && yarn check-all; popd
```

### Creating Pull Requests

Follow the PR template when creating pull requests. The template will automatically appear when you create a PR on GitHub.

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
uv run pre-commit install
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

**Note**: If the typos hook fails, you only need to fix typos in code that was changed by your PR, not pre-existing typos in the codebase.
