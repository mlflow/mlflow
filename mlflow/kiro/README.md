# MLflow Kiro CLI Integration

This module provides automatic tracing integration between the
[Kiro CLI](https://kiro.dev) (Amazon's AI coding agent) and MLflow.

## Module Structure

| File | Purpose |
|------|---------|
| `config.py` | Configuration helpers (hook files, env-var storage) |
| `hooks.py` | Stop-hook handler called by Kiro on session end |
| `cli.py` | MLflow CLI commands (`mlflow autolog kiro`) |
| `tracing.py` | Core tracing logic — session → spans → MLflow trace |

## Installation

```bash
pip install mlflow
```

Kiro CLI must also be installed. Download it at <https://kiro.dev/downloads/>.

## Usage

Set up tracing in any project directory:

```bash
# Set up tracing in the current directory (local storage)
mlflow autolog kiro

# Set up tracing in a specific project directory
mlflow autolog kiro -d ~/my-project

# Set up with a custom tracking URI
mlflow autolog kiro -u sqlite:///mlflow.db -n "Kiro Sessions"

# Set up with Databricks
mlflow autolog kiro -u databricks -e 123456789

# Check status
mlflow autolog kiro --status

# Disable tracing
mlflow autolog kiro --disable
```

## How It Works

1. **Setup**: `mlflow autolog kiro` writes a `.kiro/hooks/mlflow_autolog.json`
   file that registers a **Shell Command** action on the `AgentStop` event.
2. **Automatic capture**: When you use Kiro inside the configured directory,
   the hook fires at the end of each agent turn and calls
   `mlflow autolog kiro stop-hook`. Kiro pipes the session JSON to the
   command's stdin.
3. **Trace creation**: The handler parses the session, creates an MLflow
   `AGENT` root span with `LLM` and `TOOL` child spans for every turn, and
   persists the trace to your configured tracking store.
4. **View traces**: Run `mlflow server` and open the UI.

## Configuration Files

### Hook file: `.kiro/hooks/mlflow_autolog.json`

```json
{
  "version": "1.0",
  "hooks": [
    {
      "name": "MLflow Autolog",
      "description": "Automatically logs Kiro agent sessions to MLflow as traces.",
      "event": "AgentStop",
      "enabled": true,
      "actions": [
        {
          "type": "command",
          "command": "mlflow autolog kiro stop-hook"
        }
      ]
    }
  ]
}
```

### Env config: `.kiro/mlflow_env.json`

```json
{
  "MLFLOW_KIRO_TRACING_ENABLED": "true",
  "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
  "MLFLOW_EXPERIMENT_NAME": "Kiro Sessions"
}
```

## Captured Data

Each Kiro session becomes one MLflow trace containing:

| Span type | Contents |
|-----------|----------|
| `AGENT` (root) | User prompt, session ID, final response |
| `LLM` | Model name, input messages, response, token usage |
| `TOOL` | Tool name, input arguments, result |

## Troubleshooting

```bash
# Check if tracing is configured
mlflow autolog kiro --status

# View the raw hook log
cat .kiro/mlflow/kiro_tracing.log

# Remove tracing
mlflow autolog kiro --disable
```

## Requirements

- Python 3.10+
- MLflow (`pip install mlflow`)
- Kiro CLI installed and available in `$PATH`
