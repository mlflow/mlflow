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

## Steps

{{ language_steps }}

### 4. Verify installation

First, verify the real application:

- Run the application end-to-end via its normal entry point.
- Confirm at least one application trace is emitted to {{ tracking_uri }}.
- Confirm no runtime errors.

If the real application cannot run because required credentials, services, or
user-specific data are unavailable, do not block setup. Instead, emit exactly
one synthetic verification trace with a one-off MLflow Tracing API invocation.
Do not add a setup-only file to the repository or call a real weather service.
The synthetic trace must contain:

- A root `weather_agent` span with span type `AGENT` and a child `get_weather`
  span with span type `TOOL`.
- Tool inputs `{"location":"Sydney"}` and deterministic tool outputs
  `{"location":"Sydney","temperature_c":22,"conditions":"Sunny","synthetic":true}`.
- Root inputs under a `messages` key in OpenAI chat format, in this sequence:
  1. A user message with `What's the weather in Sydney right now?`.
  2. An assistant message with `content` set to `null` and one `tool_calls`
     entry whose ID is `call_weather_sydney_001`, type is `function`, function
     name is `get_weather`, and arguments are the JSON string
     `{"location":"Sydney"}`.
  3. A tool message whose `tool_call_id` is `call_weather_sydney_001` and whose
     content is the JSON-string form of the synthetic tool output above.
- Root outputs under `choices[0].message`, with role `assistant` and content
  `Sydney is 22°C and sunny in this synthetic setup example. This is not live weather data.`

Preserve `tool_calls` and `tool_call_id` exactly so the trace drawer can render
the conversation in Pretty view.

Validate whichever path ran:

- For real-application verification, confirm the emitted trace represents the
  application operation and has meaningful inputs, outputs, and expected child
  spans.
- For synthetic fallback verification, confirm the trace is emitted and its
  conversation renders in Pretty view. Report that this validates only the
  MLflow connection and trace rendering, not the application's instrumentation.

If MLflow calls hang during verification (e.g. because the tracking server is
slow or unreachable), set `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` and
`MLFLOW_HTTP_REQUEST_TIMEOUT=5` to fail fast instead of waiting through the
default retries.

If you don't know how to run the app, ask the user and wait for a response
before proceeding.

### 5. Report the trace URL

After the app run, capture the experiment / trace URL printed by MLflow or
constructable from the tracking URI + experiment ID. Include this URL in the
Final Summary below so the user can open the trace.

### 6. Final Summary

Summarize:

- MLflow version installed
- Files modified
- Trace URL
