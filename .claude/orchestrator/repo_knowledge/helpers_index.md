---
name: helpers_index
description: Tiny index pointing at the four per-section helpers files. Read this first to decide which helpers_*.md to load.
applies_to: any PR. Read this index, then load the matching helpers_<slug>.md based on the diff's areas.
last_verified: 2026-05-05
---

# Helpers index

`helpers.md` is split into per-section files. Read this index, then open only the per-section files relevant to the PR's diff.

| File | Section | Symbols | When to read |
|---|---|---:|---|
| `helpers_exceptions.md` | `mlflow/exceptions.py` | 14 | any PR that raises MlflowException, RestException, or any subclass; touches mlflow/exceptions.py; calls error_code / sqlstate / error_class. |
| `helpers_utils.md` | `mlflow/utils/` | 571 | any PR that touches mlflow/utils/; introduces a new utility helper anywhere in the repo; or could plausibly reuse an existing utility (lazy_load, annotations, rest_utils, file_utils, etc.). |
| `helpers_tracing_utils.md` | `mlflow/tracing/utils/` | 63 | any PR under mlflow/tracing/, mlflow/<flavor>/autolog.py, mlflow/<flavor>/chat.py, or any code that emits OTLP spans / sets span attributes. |
| `helpers_types.md` | `mlflow/types/` | 202 | any PR that touches mlflow/types/, defines a chat / agent / response BaseModel, calls validate_compat / model_dump_compat / model_validate, or threads ChatMessage / ChatTool / ResponsesAgent types. |

