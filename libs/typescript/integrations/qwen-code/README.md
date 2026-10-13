# MLflow Typescript SDK - Qwen Code

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with [Qwen Code](https://github.com/QwenLM/qwen-code) to automatically trace your Qwen Code coding-agent conversations, including user prompts, assistant responses, tool usage, and token consumption.

| Package                 | NPM                                                                                                                                     | Description                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [@mlflow/qwen-code](./) | [![npm package](https://img.shields.io/npm/v/%40mlflow%2Fqwen-code?style=flat-square)](https://www.npmjs.com/package/@mlflow/qwen-code) | Auto-instrumentation integration for Qwen Code. |

## Installation

```bash
npm install -g @mlflow/qwen-code
```

This installs the `mlflow-qwen-code` CLI globally. If you'd rather not install globally, you can invoke it via `npx @mlflow/qwen-code` (every command below works the same way).

## Quickstart

Start MLflow Tracking Server if you don't have one already:

```bash
pip install mlflow
mlflow server --port 5000
```

Self-hosting MLflow server requires Python 3.10 or higher. If you don't have one, you can also use [managed MLflow service](https://mlflow.org/#get-started) for free to get started quickly.

Run the interactive setup. It registers a Qwen Code `Stop` hook and writes your tracking URI / experiment ID into Qwen Code's config directory:

```bash
mlflow-qwen-code setup
```

The setup command prompts you to choose between a project-local install (`./.qwen/`) or a user-level install (`~/.qwen/`), then writes:

- `settings.json`: adds a `Stop` hook entry so Qwen Code invokes `mlflow-qwen-code stop-hook` at the end of each session turn.
- `mlflow-tracing.json`: persists your MLflow tracking URI and experiment ID.

Pass `--non-interactive` / `-y` to skip prompts and use defaults, or override values with `--tracking-uri` and `--experiment-id`:

```bash
mlflow-qwen-code setup -y --tracking-uri http://localhost:5000 --experiment-id 0
```

Use Qwen Code normally:

```bash
qwen "help me refactor this function"
```

After each conversation turn, MLflow records a trace with the message history, tool calls and results, and token usage. You don't need to wait for the session to end.

## Configuration

The `mlflow-qwen-code` hook resolves configuration in this order (first match wins):

1. `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID` environment variables
2. `./.qwen/mlflow-tracing.json` (project-local)
3. `~/.qwen/mlflow-tracing.json` (user-level)

Environment variables are convenient for one-off overrides, e.g. switching between a local server and a Databricks workspace:

```bash
MLFLOW_TRACKING_URI=databricks MLFLOW_EXPERIMENT_ID=123456789 qwen "..."
```

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart). For the full Qwen Code tracing guide including troubleshooting, see the [Qwen Code integration page](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/qwen_code).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).
