# @mlflow/opencode

MLflow tracing plugin for [OpenCode](https://opencode.ai).

This plugin automatically traces OpenCode conversations to MLflow, capturing:
- User prompts and assistant responses
- LLM calls with token usage
- Tool invocations and results
- Session metadata

## Installation

```bash
npm install @mlflow/opencode mlflow-tracing
```

## Usage

1. Add to your `opencode.json`:

```json
{
  "plugin": ["@mlflow/opencode"]
}
```

2. Set environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_ID=123
```

3. Run OpenCode normally - traces are created automatically when sessions become idle.

## Configuration

The plugin is configured via environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | MLflow tracking server URI (e.g., `http://localhost:5000`) |
| `MLFLOW_EXPERIMENT_ID` | Yes | MLflow experiment ID |
| `MLFLOW_OPENCODE_DEBUG` | No | Set to `true` to enable debug logging |

## Viewing Traces

Start an MLflow server and view your traces in the UI:

```bash
mlflow server
# Open http://localhost:5000
```

## License

Apache-2.0
