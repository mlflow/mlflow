# MLflow Tracing Setup (Agent Instructions)

You are being launched by `mlflow agent setup` in repo `{{ repo_root }}`.
{{ skills_intro }}

## Hard Rules

- **One app, one entry point per run.** If the repo has more than one candidate,
  ask the user which to instrument before starting.
- **Install the latest MLflow.** Use the project's package manager's normal
  install. Do not hard-pin the version unless the user asks.
- **Do not add eval code** unless explicitly requested.
- **If MLflow is already installed and configured, do not duplicate work.**
  Note the existing setup in the final summary.
- {{ no_overwrite_bullet }}

## Execution Requirements

Before writing any code:

1. Create a **checklist** from the steps below.
2. Execute each step in order.
3. Do not skip steps.

## Steps

{{ language_steps }}

### 4. Verify installation (MANDATORY)

- Run the application end-to-end via its normal entry point.
- Confirm at least one trace is emitted to {{ tracking_uri }}.
- Confirm no runtime errors.

If MLflow calls hang during verification (e.g. because the tracking server is
slow or unreachable), set `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` and
`MLFLOW_HTTP_REQUEST_TIMEOUT=5` to fail fast instead of waiting through the
default retries.

If you don't know how to run the app, ask the user and wait for a response
before proceeding.

### 5. Report the trace URL (CRITICAL)

After the app run, capture the experiment / trace URL printed by MLflow or
constructable from the tracking URI + experiment ID. This URL must appear in
the final summary so the user can open it in the MLflow UI.

### 6. Final Summary

Summarize:

- MLflow version installed
- Files modified
- Trace URL (required)
