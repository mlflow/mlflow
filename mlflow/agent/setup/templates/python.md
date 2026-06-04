{server_setup}### 1. Install MLflow

Detect this project's Python package manager and add `mlflow` as a dependency
if it is not already declared:

- `uv` (look for `uv.lock` or `[tool.uv]` in `pyproject.toml`) -> `uv add mlflow`
- `poetry` (look for `poetry.lock`) -> `poetry add mlflow`
- `pip` / plain `requirements.txt` -> append `mlflow` and `pip install mlflow`

Skip this step if `mlflow` is already a declared dependency.

### 2. Configure tracking URI

Configure MLflow to log to `{tracking_uri}`. Pick whichever of these fits the
project's conventions:

- Set `MLFLOW_TRACKING_URI={tracking_uri}` in the project's env file (`.env`,
  `.env.example`, etc.).
- Call `mlflow.set_tracking_uri("{tracking_uri}")` once during application
  startup, before any `mlflow.*` calls.

Don't do both. If the project already sets a tracking URI, leave it alone and
note the existing value in the final summary.

### 3. Instrument with `mlflow.autolog`

Consult the `instrumenting-with-mlflow-tracing` skill in `{skills_dir}/` for
the supported libraries and per-integration setup. That skill is the source
of truth for what `mlflow.autolog()` covers.

For most applications, `mlflow.autolog()` is the recommended entry point:

```python
import mlflow

mlflow.set_tracking_uri("{tracking_uri}")
mlflow.autolog()
```

Wire this into the application's entry point(s):

- Find the main entry (e.g. `main.py`, `app.py`, `__main__.py`, FastAPI
  lifespan / `Depends`, Django app config `ready` hook, Lambda handler init).
- Call `mlflow.autolog()` once, before any LLM clients are created.
- Do not add it to library modules or tests.

For library-specific instrumentation (LangChain, LangGraph, OpenAI, Anthropic,
LlamaIndex, DSPy, etc.), many libraries have a dedicated
`mlflow.<library>.autolog()` flavor. The skill above lists them.
