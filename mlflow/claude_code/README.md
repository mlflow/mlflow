# MLflow Claude Code Integration

This module provides automatic tracing integration between Claude Code and MLflow.

## Module Structure

- **`config.py`** - Configuration management (settings files, environment variables)
- **`hooks.py`** - Claude Code hook setup and management
- **`cli.py`** - MLflow CLI commands (`mlflow autolog claude`)
- **`tracing.py`** - Core tracing logic and processors
- **`hooks/`** - Hook implementation handlers

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
mlflow autolog claude ~/my-project

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

1. **Setup**: The `mlflow autolog claude` command configures Claude Code hooks in a `.claude/settings.json` file
2. **Automatic Tracing**: When you use the `claude` command in the configured directory, your conversations are automatically traced to MLflow
3. **View Traces**: Use `mlflow ui` to view your conversation traces

## Configuration

The setup creates two types of configuration:

### Claude Code Hooks

- **PostToolUse**: Captures tool usage during conversations
- **Stop**: Processes complete conversations into MLflow traces

### Environment Variables

- `MLFLOW_CLAUDE_TRACING_ENABLED=true`: Enables tracing
- `MLFLOW_TRACKING_URI`: Where to store traces (defaults to local `.claude/mlflow/runs`)
- `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`: Which experiment to use

## Examples

### Basic Local Setup

```bash
mlflow autolog claude
cd .
claude "help me write a function"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Databricks Integration

```bash
mlflow autolog claude -u databricks -e 123456789
claude "analyze this data"
# View traces in Databricks
```

### Custom Project Setup

```bash
mlflow autolog claude ~/my-ai-project -u sqlite:///mlflow.db -n "My AI Project"
cd ~/my-ai-project
claude "refactor this code"
mlflow ui --backend-store-uri sqlite:///mlflow.db
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
