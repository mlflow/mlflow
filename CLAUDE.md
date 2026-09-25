# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Knowledge Cutoff Note

Claude's training data may lag behind current releases. When reviewing docs or code, don't flag unfamiliar names as speculative or non-existent. Assume the authors are referencing newer, valid resources (e.g., model names like GPT-5, GitHub runner types like ubuntu-slim, library versions, etc.).

## Code Style Principles

- Use top-level imports (only use lazy imports when necessary)
- Only add docstrings in tests when they provide additional context
- Only add comments that explain non-obvious logic or provide additional context
- In source files, use full `https://github.com/<owner>/<repo>/issues/<number>` URLs for cross-repository issue references instead of `<owner>/<repo>#<number>`, because the shorthand does not autolink or identify the target type; it remains fine in PR descriptions and issue comments, where GitHub autolinks it.
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
LOG=$(mktemp) && echo "Logs: $LOG"
uv run dev/run_dev_server.py > "$LOG" 2>&1 &

# Monitor the logs (server URLs are printed there)
tail -f "$LOG"
```

### Reviewing provider-gated UI without credentials

Features gated on an external provider/credentials can be rendered without real
keys, cost, or nondeterminism via credential-free stubs (see `dev/dev_stubs/`):

```bash
uv run dev/run_dev_server.py --stub-providers claude
```

- `claude` — a fake `claude` CLI on the dev server's PATH, so the MLflow
  Assistant's Claude Code provider passes its auth probe and the chat panel
  renders without `ANTHROPIC_API_KEY`.

The stubs are dev/CI-only; the `ui-review` bot always passes `--stub-providers claude`
so the Assistant is reviewable on any PR.

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

# Start the dev server with these environment variables (unique log per invocation)
LOG=$(mktemp) && echo "Logs: $LOG"
uv run dev/run_dev_server.py > "$LOG" 2>&1 &

# Monitor the logs (server URLs are printed there)
tail -f "$LOG"
```

**Note**: The MLflow server acts as a proxy, forwarding API requests to your Databricks workspace while serving the local React frontend. This allows you to develop and test UI changes against real Databricks data.

## Development Commands

### Package Cooldown Period

7-day cooldown on new package releases to guard against compromised or broken
versions that get pulled and yanked within a few days. Keep these in sync:

- Python: `exclude-newer = "P7D"` in `pyproject.toml` (`torch`/`torchvision` opted out).
- JavaScript: `min-release-age=7` in `.npmrc`; `npmMinimalAgeGate: 7d` in `.yarnrc.yml`.

Pass `--min-release-age=7` to any new `npx` invocations.

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

### Keep Pull Requests to One Concern

One PR = one concern. NEVER EVER bundle unrelated changes. They multiply review
cost rather than adding to it, and since this repo squash-merges, they land as
one commit that can't be reverted piece by piece and is hard to reason about
later. When in doubt, split.

### Creating Pull Requests

- Follow the instructions at the top of [the PR template](./.github/pull_request_template.md) carefully.
- Inside `gh pr ... --body "$(cat <<'EOF' ... EOF)"`, write backticks plain. The quoted `'EOF'` delimiter already suppresses command substitution, so escaping as `` \` `` is unnecessary and the backslashes get persisted in the PR body, rendering literally instead of as code spans.

  ```bash
  gh pr create --body "$(cat <<'EOF'
  Updated \`pyproject.toml\` to bump the version. # BAD
  Updated `pyproject.toml` to bump the version.   # GOOD
  EOF
  )"
  ```

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
uv run --only-group lint pre-commit run install-bin -a -v
```

Run pre-commit manually:

```bash
# Run on all files
uv run --only-group lint pre-commit run --all-files

# Run on specific files
uv run --only-group lint pre-commit run --files path/to/file.py

# Run a specific hook
uv run --only-group lint pre-commit run ruff --all-files
```

`--only-group lint` keeps uv from syncing the full `dev` environment just to run
the hooks.
