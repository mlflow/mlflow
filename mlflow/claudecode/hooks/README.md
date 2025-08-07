# MLflow Claude Code Integration

This directory contains hooks and utilities for automatic tracing of Claude Code interactions with MLflow.

## Quick Start

1. **Install MLflow with the claude-mlflow binary:**

   ```bash
   pip install mlflow
   # or with uvx:
   uvx mlflow claude-mlflow --help
   ```

2. **Use claude-mlflow instead of claude** (tracing is automatically enabled):

   ```bash
   claude-mlflow --help
   claude-mlflow "Help me analyze this codebase"
   ```

   Optional: Set tracking URI to customize where traces are stored:

   ```bash
   export MLFLOW_TRACKING_URI="file://./.claude/mlflow/runs"
   ```

## Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking URI (default: `file://./.claude/mlflow/runs`)
- `MLFLOW_EXPERIMENT_ID`: Experiment ID to use (takes precedence over name)
- `MLFLOW_EXPERIMENT_NAME`: Experiment name to use (ignored if ID is set)
- If neither ID nor name is set, defaults to experiment 0 (the default experiment)

**Note**: `MLFLOW_CLAUDE_TRACING_ENABLED` is automatically set to `true` when using `claude-mlflow`

### Hook Configuration

To set up hooks manually in your Claude Code configuration:

1. **Create hooks configuration** in your project's `.claude/hooks.json`:

   ```json
   {
     "postToolUse": {
       "command": ["python", "/path/to/mlflow/hooks/post_tool_use.py"]
     },
     "stop": {
       "command": ["python", "/path/to/mlflow/hooks/stop_hook.py"]
     }
   }
   ```

2. **Set environment variables** in your `.claude/settings.json`:
   ```json
   {
     "environment": {
       "MLFLOW_CLAUDE_TRACING_ENABLED": "true",
       "MLFLOW_TRACKING_URI": "file://./.claude/mlflow/runs"
     }
   }
   ```

## Directory Structure

When tracing is enabled, MLflow will create:

```
$working_directory/
├── .claude/
│   └── mlflow/
│       ├── claude_tracing.log    # Hook logs
│       └── runs/                 # MLflow tracking data
│           ├── experiments/
│           └── traces/
```

## Trace Structure

Each Claude Code conversation creates a trace with:

- **Root span**: The entire conversation
- **LLM spans**: Each Claude API call with token usage and cost
- **Tool spans**: Each tool execution (Read, Write, Bash, etc.) with inputs/outputs

## Testing Hooks

Test the stop hook directly:

```bash
python /path/to/mlflow/hooks/stop_hook.py /path/to/transcript.jsonl
```

## Troubleshooting

1. **Check if using claude-mlflow** (tracing is automatically enabled):

   ```bash
   which claude-mlflow
   ```

2. **Check logs:**

   ```bash
   tail -f .claude/mlflow/claude_tracing.log
   ```

3. **Verify MLflow setup:**
   ```bash
   python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```
