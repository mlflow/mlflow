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

| Variable                | Required | Description                                                                          |
| ----------------------- | -------- | ------------------------------------------------------------------------------------ |
| `MLFLOW_TRACKING_URI`   | Yes      | MLflow tracking server URI (e.g., `http://localhost:5000`)                           |
| `MLFLOW_EXPERIMENT_ID`  | Yes      | MLflow experiment ID                                                                 |
| `MLFLOW_TRACE_LOCATION` | No       | Databricks Unity Catalog trace location as `catalog.schema.table_prefix` (see below) |
| `MLFLOW_OPENCODE_DEBUG` | No       | Set to `true` to enable debug logging                                                |

### Databricks Unity Catalog trace location

To route traces to a Databricks Unity Catalog location instead of the default
experiment-backed path, set `MLFLOW_TRACE_LOCATION` to a fully-qualified
`catalog.schema.table_prefix`:

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_ID=123
export MLFLOW_TRACE_LOCATION=my_catalog.my_schema.my_prefix
```

All three parts are required. The UC trace location must already be provisioned
in the workspace; the plugin does not create it. When set, traces are written
via the Unity Catalog ingestion path (V4 trace IDs).

## Viewing Traces

Start an MLflow server and view your traces in the UI:

```bash
mlflow server
# Open http://localhost:5000
```

## License

Apache-2.0
