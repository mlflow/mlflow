# MLflow Claude Code Integration

This module provides automatic tracing integration between Claude Code and MLflow.

## Module Structure

- **`config.py`** - Configuration management (settings files, environment variables)
- **`plugin.py`** - Claude plugin bootstrap and legacy hook migration
- **`hooks.py`** - Legacy compatibility shim for older Python-hook installs
- **`cli.py`** - MLflow CLI commands (`mlflow autolog claude`)
- **`tracing.py`** - Core tracing logic for Claude CLI transcript processing and Claude Agent SDK traces

## Installation

```bash
pip install mlflow
```

## Usage

Set up Claude Code tracing in any project directory:

```bash
# Set up tracing in current directory
mlflow autolog claude

# Set up tracing in specific directory
mlflow autolog claude -d ~/my-project

# Set up with custom tracking URI
mlflow autolog claude -u file://./custom-mlruns
mlflow autolog claude -u sqlite:///mlflow.db

# Set up with Databricks
mlflow autolog claude -u databricks -e 123456789

# Check status
mlflow autolog claude --status

# Disable tracing
mlflow autolog claude --disable
```

## How it Works

1. **Setup**: The `mlflow autolog claude` command installs the MLflow Claude plugin and writes tracing config into `.claude/settings.json`
2. **Automatic Tracing**: When you use the `claude` command in the configured directory, the plugin automatically traces your conversations to MLflow
3. **View Traces**: Use `mlflow server` to view your conversation traces

## Configuration

The setup creates two types of configuration:

### Claude Plugin Runtime

- `mlflow autolog claude` installs `mlflow-tracing` from the MLflow Claude marketplace
- Claude Code automatically loads the plugin's `Stop` hook from the plugin bundle

### Environment Variables

- `MLFLOW_CLAUDE_TRACING_ENABLED=true`: Enables tracing
- `MLFLOW_TRACKING_URI`: Where to store traces (defaults to the active MLflow tracking URI)
- `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`: Which experiment to use

## Examples

### Basic Local Setup

```bash
mlflow autolog claude
cd .
claude "help me write a function"
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### Databricks Integration

```bash
mlflow autolog claude -u databricks -e 123456789
claude "analyze this data"
# View traces in Databricks
```

### Custom Project Setup

```bash
mlflow autolog claude -d ~/my-ai-project -u sqlite:///mlflow.db -n "My AI Project"
cd ~/my-ai-project
claude "refactor this code"
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

### Check Status

```bash
mlflow autolog claude --status
```

### Disable Tracing

```bash
mlflow autolog claude --disable
```

### View Raw Configuration

The configuration is stored in `.claude/settings.json`:

```bash
cat .claude/settings.json
```

## Requirements

- Python 3.10+ (required by MLflow)
- MLflow installed (`pip install mlflow`)
- Claude Code CLI installed
