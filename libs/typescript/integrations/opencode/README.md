# @mlflow/opencode

MLflow tracing plugin for [OpenCode](https://opencode.ai).

This plugin automatically traces OpenCode conversations to MLflow, capturing:

- User prompts and assistant responses
- LLM calls with token usage
- Tool invocations and results
- Session metadata

## Installation

```bash
npm install @mlflow/opencode
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

| Variable                | Required | Description                                                |
| ----------------------- | -------- | ---------------------------------------------------------- |
| `MLFLOW_TRACKING_URI`   | Yes      | MLflow tracking server URI (e.g., `http://localhost:5000`) |
| `MLFLOW_EXPERIMENT_ID`  | Yes      | MLflow experiment ID                                       |
| `MLFLOW_TRACE_LOCATION` | For UC   | Existing `catalog.schema.table_prefix` trace location      |
| `MLFLOW_OPENCODE_DEBUG` | No       | Set to `true` to enable debug logging                      |

For a Databricks experiment backed by Unity Catalog, set `MLFLOW_TRACE_LOCATION`
alongside the required variables:

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_ID=123
export MLFLOW_TRACE_LOCATION=my_catalog.my_schema.my_table_prefix
```

The UC trace location must already exist. Without this setting, the SDK uses the
experiment-backed path, and traces will not appear in the UC table. An invalid
location stops tracing and prints a warning instead of silently falling back.

## Viewing Traces

Start an MLflow server and view your traces in the UI:

```bash
mlflow server
# Open http://localhost:5000
```

## License

Apache-2.0
