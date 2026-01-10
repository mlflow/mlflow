# Skills CLI

A Python package that provides CLI commands for Claude Code skills.

## Usage

From the `.claude/skills` directory:

```bash
uv run skills <command> [args]
```

## Available Commands

| Command                     | Description                                   |
| --------------------------- | --------------------------------------------- |
| `fetch-diff`                | Fetch PR diff with filtering and line numbers |
| `fetch-unresolved-comments` | Fetch unresolved PR review comments           |
| `analyze-ci`                | Analyze failed GitHub Action jobs             |

## Examples

```bash
# Fetch a PR diff
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123

# Fetch unresolved comments
uv run skills fetch-unresolved-comments https://github.com/mlflow/mlflow/pull/123

# Analyze CI failures
uv run skills analyze-ci https://github.com/mlflow/mlflow/pull/123
```

## Development

To add a new command:

1. Create `src/skills/cmds/<command_name>.py`
2. Implement `register(subparsers)` function
3. Register in `src/skills/cli.py`
