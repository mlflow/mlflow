---
name: helpers_exceptions
description: Auto-generated public-symbol reference for `mlflow/exceptions.py`. Use this before suggesting a new helper.
applies_to: any PR that raises MlflowException, RestException, or any subclass; touches mlflow/exceptions.py; calls error_code / sqlstate / error_class.
last_verified: 2026-05-05
citation_policy: each `path:line` is the `def` / `class` line. If the snippet drifts, search by symbol name.
generated_by: .claude/orchestrator/scripts/generate_helpers_md.py (refreshed weekly by .github/workflows/refresh-helpers.yml).
---

# Helpers: `mlflow/exceptions.py`

Auto-generated. Walks `mlflow/exceptions.py` and lists every public symbol with its signature and first docstring sentence.

## How to use this file

- **Before suggesting a new utility function in a review**, grep this file for the area you're touching. If a helper already exists, point at its `path:line` instead of asking for a new one.
- **Class entries** list public methods in the same row group (`ClassName.method` form).
- **Search by symbol name**, not by line number: line numbers drift after reformats.

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_error_code` | function | `(http_status)` |  | 62 |
| `MlflowException` | class | `(Exception)` | Generic exception thrown to surface failure information about external-facing operations. | 68 |
| `MlflowException.serialize_as_json` | method | `(self)` |  | 120 |
| `MlflowException.get_http_status_code` | method | `(self)` |  | 129 |
| `MlflowException.invalid_parameter_value` | classmethod | `(cls, message: str, sqlstate: str \| None, error_class: str \| None, **kwargs)` | Constructs an `MlflowException` object with the `INVALID_PARAMETER_VALUE` error code. | 133 |
| `RestException` | class | `(MlflowException)` | Exception thrown on non 200-level responses from the REST API | 155 |
| `ExecutionException` | class | `(MlflowException)` | Exception thrown when executing a project fails | 204 |
| `MissingConfigException` | class | `(MlflowException)` | Exception thrown when expected configuration file/directory not found | 208 |
| `InvalidUrlException` | class | `(MlflowException)` | Exception thrown when a http request fails to send due to an invalid URL | 212 |
| `MlflowTracingException` | class | `(MlflowException)` | Exception thrown from tracing logic  Tracing logic should not block the main execution flow in general, hence this exception is... | 243 |
| `MlflowTraceDataException` | class | `(MlflowTracingException)` | Exception thrown for trace data related error | 255 |
| `MlflowTraceDataNotFound` | class | `(MlflowTraceDataException)` | Exception thrown when trace data is not found | 272 |
| `MlflowTraceDataCorrupted` | class | `(MlflowTraceDataException)` | Exception thrown when trace data is corrupted | 279 |
| `MlflowNotImplementedException` | class | `(MlflowException)` | Exception thrown when a feature is not implemented | 286 |

