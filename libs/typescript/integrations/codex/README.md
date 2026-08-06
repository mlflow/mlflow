# MLflow Typescript SDK - Codex CLI

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with [Codex CLI](https://github.com/openai/codex) to automatically trace your Codex coding-agent conversations, including user prompts, assistant responses, tool usage, and token consumption.

| Package             | NPM                                                                                                                             | Description                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [@mlflow/codex](./) | [![npm package](https://img.shields.io/npm/v/%40mlflow%2Fcodex?style=flat-square)](https://www.npmjs.com/package/@mlflow/codex) | Auto-instrumentation integration for OpenAI Codex CLI. |

## Installation

```bash
npm install -g @mlflow/codex
```

This installs the `mlflow-codex` CLI globally. If you'd rather not install globally, you can invoke it via `npx @mlflow/codex` (every command below works the same way).

## Quickstart

Start MLflow Tracking Server if you don't have one already:

```bash
pip install mlflow
mlflow server --port 5000
```

Self-hosting MLflow server requires Python 3.10 or higher. If you don't have one, you can also use [managed MLflow service](https://mlflow.org/#get-started) for free to get started quickly.

Run the interactive setup. It registers the Codex `notify` hook and writes your tracking URI / experiment ID into Codex's config directory:

```bash
mlflow-codex setup
```

The setup command prompts you to choose between a project-local install (`./.codex/`) or a user-level install (`~/.codex/`), then writes:

- `config.toml`: adds `notify = ["mlflow-codex", "notify-hook"]` so Codex invokes the hook after every turn.
- `mlflow-tracing.json`: persists your MLflow tracking URI and experiment ID.

Pass `--non-interactive` / `-y` to skip prompts and use defaults, or override values with `--tracking-uri` and `--experiment-id`:

```bash
mlflow-codex setup -y --tracking-uri http://localhost:5000 --experiment-id 0
```

Use Codex normally:

```bash
codex "help me refactor this function"
```

After each conversation turn, MLflow records a trace with the message history, tool calls and results, and token usage. You don't need to wait for the session to end.

## Configuration

The `mlflow-codex` hook resolves configuration in this order (first match wins):

1. `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID` environment variables
2. `./.codex/mlflow-tracing.json` (project-local)
3. `~/.codex/mlflow-tracing.json` (user-level)

Environment variables are convenient for one-off overrides, e.g. switching between a local server and a Databricks workspace:

```bash
MLFLOW_TRACKING_URI=databricks MLFLOW_EXPERIMENT_ID=123456789 codex "..."
```

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart). For the full Codex CLI tracing guide including troubleshooting and OTLP support, see the [Codex CLI integration page](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/codex).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).
