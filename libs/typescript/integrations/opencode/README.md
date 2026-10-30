# @mlflow/opencode

MLflow tracing plugin for [OpenCode](https://opencode.ai).

This plugin automatically traces OpenCode conversations to MLflow, capturing:

- User prompts and assistant responses
- LLM calls with token usage and cost
- Tool invocations and results
- Session metadata

Version 1.x requires OpenCode V2. For OpenCode V1, pin `@mlflow/opencode@0.4`.
When 1.x is loaded on OpenCode V1, it does not trace and prints a message that
tells you to pin `0.4`.

## Installation

```bash
opencode plugin add @mlflow/opencode
```

## Usage

1. Configure the plugin in your `opencode.json`:

```json
{
  "plugins": [
    {
      "package": "@mlflow/opencode",
      "options": {
        "trackingUri": "http://localhost:5000",
        "experimentId": "123"
      }
    }
  ]
}
```

You can also set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` instead of
the options. The OpenCode server process must have these variables. If the
OpenCode background service is already running, restart it
(`opencode service restart`) from a shell that has them.

2. Run OpenCode normally - a trace is created when each session run ends, also
   when it fails or is interrupted.

## Configuration

| Option         | Environment variable    | Required | Description                                                |
| -------------- | ----------------------- | -------- | ---------------------------------------------------------- |
| `trackingUri`  | `MLFLOW_TRACKING_URI`   | Yes      | MLflow tracking server URI (e.g., `http://localhost:5000`) |
| `experimentId` | `MLFLOW_EXPERIMENT_ID`  | Yes      | MLflow experiment ID                                       |
|                | `MLFLOW_TRACE_LOCATION` | For UC   | Existing `catalog.schema.table_prefix` trace location      |
| `debug`        | `MLFLOW_OPENCODE_DEBUG` | No       | Set to `true` to enable debug logging                      |

Plugin options take precedence over environment variables. Set credentials
only as environment variables, for example `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD`, or `MLFLOW_TRACKING_TOKEN`.

For a Databricks experiment backed by Unity Catalog, set `MLFLOW_TRACE_LOCATION`
alongside the required variables:

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_ID=123
export MLFLOW_TRACE_LOCATION=my_catalog.my_schema.my_table_prefix
```

The UC trace location must already exist. Without this setting (or with an empty
value), the SDK uses the experiment-backed path, and traces will not appear in
the UC table. A non-empty invalid location stops tracing and prints a warning
instead of silently falling back.

## Token Usage and Cost

Token usage includes cache reads and writes in `input_tokens`, and reasoning
tokens in `output_tokens`. Cost comes from the cost that OpenCode reports for
each model call. The plugin splits it into input and output cost with the
model's prices from OpenCode.

## Viewing Traces

Start an MLflow server and view your traces in the UI:

```bash
mlflow server
# Open http://localhost:5000
```

## License

Apache-2.0
