---
name: helpers_tracing_utils
description: Auto-generated public-symbol reference for `mlflow/tracing/utils/`. Use this before suggesting a new helper.
applies_to: any PR under mlflow/tracing/, mlflow/<flavor>/autolog.py, mlflow/<flavor>/chat.py, or any code that emits OTLP spans / sets span attributes.
last_verified: 2026-05-05
citation_policy: each `path:line` is the `def` / `class` line. If the snippet drifts, search by symbol name.
generated_by: .claude/orchestrator/scripts/generate_helpers_md.py (refreshed weekly by .github/workflows/refresh-helpers.yml).
---

# Helpers: `mlflow/tracing/utils/`

Auto-generated. Walks `mlflow/tracing/utils/` and lists every public symbol with its signature and first docstring sentence.

## How to use this file

- **Before suggesting a new utility function in a review**, grep this file for the area you're touching. If a helper already exists, point at its `path:line` instead of asking for a new one.
- **Class entries** list public methods in the same row group (`ClassName.method` form).
- **Search by symbol name**, not by line number: line numbers drift after reformats.

## `mlflow/tracing/utils/__init__.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `capture_function_input_args` | function | `(func, args, kwargs) -> dict[str, Any] \| None` |  | 45 |
| `TraceJSONEncoder` | class | `(json.JSONEncoder)` | Custom JSON encoder for serializing non-OpenTelemetry compatible objects in a trace or span. | 67 |
| `TraceJSONEncoder.default` | method | `(self, obj)` |  | 75 |
| `dump_span_attribute_value` | function | `(value: Any) -> str` |  | 125 |
| `try_json_loads` | function | `(value: Any) -> Any` | Try to parse a value as JSON, returning the original value on failure. | 143 |
| `encode_span_id` | function | `(span_id: int) -> str` | Encode the given integer span ID to a 16-byte hex string. | 152 |
| `encode_trace_id` | function | `(trace_id: int) -> str` | Encode the given integer trace ID to a 32-byte hex string. | 163 |
| `decode_id` | function | `(span_or_trace_id: str) -> int` | Decode the given hex string span or trace ID to an integer. | 170 |
| `get_mlflow_span_for_otel_span` | function | `(span: OTelSpan) -> LiveSpan \| None` | Get the active MLflow span for the given OpenTelemetry span. | 177 |
| `build_otel_context` | function | `(trace_id: int, span_id: int) -> trace_api.SpanContext` | Build an OpenTelemetry SpanContext object from the given trace and span IDs. | 188 |
| `aggregate_usage_from_spans` | function | `(spans: list[LiveSpan]) -> dict[str, int] \| None` | Aggregate token usage information from all spans in the trace. | 265 |
| `aggregate_cost_from_spans` | function | `(spans: list[LiveSpan]) -> dict[str, float] \| None` | Aggregate cost information from all spans in the trace. | 276 |
| `calculate_span_cost` | function | `(span: LiveSpan) -> dict[str, float] \| None` | Calculate cost for a single span using LiteLLM pricing data. | 286 |
| `calculate_cost_by_model_and_token_usage` | function | `(model_name: str \| None, usage: dict[str, int] \| None, model_provider: str \| None) -> dict[str, float] \...` |  | 307 |
| `get_otel_attribute` | function | `(span: trace_api.Span, key: str) -> str \| None` | Get the attribute value from the OpenTelemetry span in a decoded format. | 388 |
| `maybe_get_request_id` | function | `(is_evaluate) -> str \| None` | Get the request ID if the current prediction is as a part of MLflow model evaluation. | 420 |
| `maybe_get_dependencies_schemas` | function | `() -> dict[str, Any] \| None` |  | 437 |
| `maybe_get_logged_model_id` | function | `() -> str \| None` | Get the logged model ID associated with the current prediction context. | 442 |
| `exclude_immutable_tags` | function | `(tags: dict[str, str]) -> dict[str, str]` | Exclude immutable tags e.g. | 450 |
| `generate_mlflow_trace_id_from_otel_trace_id` | function | `(otel_trace_id: int) -> str` | Generate an MLflow trace ID from an OpenTelemetry trace ID. | 455 |
| `generate_trace_id_v4_from_otel_trace_id` | function | `(otel_trace_id: int, location: str) -> str` | Generate a trace ID in v4 format from the given OpenTelemetry trace ID. | 468 |
| `generate_trace_id_v4` | function | `(span: OTelSpan, location: str) -> str` | Generate a trace ID for the given span. | 482 |
| `generate_trace_id_v3` | function | `(span: OTelSpan) -> str` | Generate a trace ID for the given span (V3 trace schema). | 496 |
| `generate_request_id_v2` | function | `() -> str` | Generate a request ID for the given span. | 505 |
| `construct_full_inputs` | function | `(func, *args, **kwargs) -> dict[str, Any]` | Construct the full input arguments dictionary for the given function, including positional and keyword arguments. | 516 |
| `maybe_set_prediction_context` | function | `(context: 'Context' \| None)` | Set the prediction context if the given context is not None. | 532 |
| `set_span_chat_tools` | function | `(span: LiveSpan, tools: list[ChatTool])` | Set the `mlflow.chat.tools` attribute on the specified span. | 546 |
| `add_size_stats_to_trace_metadata` | function | `(trace: Trace)` | Calculate the stats of trace and span sizes and add it as a metadata to the trace. | 640 |
| `update_trace_state_from_span_conditionally` | function | `(trace, root_span)` | Update trace state from span status, but only if the user hasn't explicitly set a different trace status. | 694 |
| `get_experiment_id_for_trace` | function | `(span: OTelReadableSpan) -> str` | Determine the experiment ID to associate with the trace. | 721 |
| `get_active_spans_table_name` | function | `() -> str \| None` | Get active Unity Catalog spans table name that's set by `mlflow.tracing.set_destination`. | 753 |
| `generate_assessment_id` | function | `() -> str` | Generates an assessment ID of the form 'a-<uuid4>' in hex string format. | 767 |
| `parse_trace_id_v4` | function | `(trace_id: str \| None) -> tuple[str \| None, str \| None]` | Parse the trace ID into location and trace ID components. | 796 |
| `construct_trace_id_v4` | function | `(location: str, trace_id: str) -> str` | Construct a trace ID for the given location and trace ID. | 814 |
| `set_span_model_attribute` | function | `(span: LiveSpan, inputs: dict[str, Any]) -> None` | Set the model attribute on a span using parsed model information. | 821 |
| `should_compute_cost_client_side` | function | `() -> bool` | Whether LLM cost should be computed on the client side. | 840 |
| `set_span_cost_attribute` | function | `(span: LiveSpan) -> None` | Set the cost attribute on a span using calculated cost information. | 853 |

## `mlflow/tracing/utils/artifact_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_artifact_uri_for_trace` | function | `(trace_info: TraceInfo)` | Get the artifact uri for accessing the trace data. | 9 |

## `mlflow/tracing/utils/copy.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `copy_trace_to_experiment` | function | `(trace_dict: dict[str, Any], experiment_id: str \| None) -> str` | Copy the given trace to the current experiment. | 9 |

## `mlflow/tracing/utils/environment.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `resolve_env_metadata` | function | `()` | Resolve common environment metadata to be saved in the trace info. | 20 |

## `mlflow/tracing/utils/exception.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `raise_as_trace_exception` | function | `(f)` | A decorator to make sure that the decorated function only raises MlflowTracingException. | 6 |

## `mlflow/tracing/utils/once.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `Once` | class | `` | Execute a function exactly once and block all callers until the function returns | 6 |
| `Once.do_once` | method | `(self, func: Callable[[], None])` | Execute ``func`` if it hasn't been executed or return. | 13 |

## `mlflow/tracing/utils/otlp.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `build_otlp_headers` | function | `(experiment_id: str) -> dict[str, str]` | Build OTLP exporter headers with experiment ID and auth credentials. | 23 |
| `should_use_otlp_exporter` | function | `() -> bool` | Determine if OTLP traces should be exported based on environment configuration. | 37 |
| `should_export_otlp_metrics` | function | `() -> bool` | Determine if OTLP metrics should be exported based on environment configuration. | 44 |
| `get_otlp_exporter` | function | `() -> SpanExporter` | Get the OTLP exporter based on the configured protocol. | 53 |
| `decompress_otlp_body` | function | `(raw_body: bytes, content_encoding: str) -> bytes` | Decompress OTLP request body according to Content-Encoding. | 222 |
| `resource_to_otel_proto` | function | `(resource: OTelResource \| None) -> OTelProtoResource` | Convert an OpenTelemetry SDK Resource to protobuf Resource format. | 263 |

## `mlflow/tracing/utils/processor.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `apply_span_processors` | function | `(span)` | Apply configured span processors sequentially to the span. | 8 |
| `validate_span_processors` | function | `(span_processors)` | Validate that the span processor is a valid function. | 35 |

## `mlflow/tracing/utils/prompt.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `update_linked_prompts_tag` | function | `(current_tag_value: str \| None, prompt_versions: list[PromptVersion])` | Utility method to update linked prompts tag value with a new prompt version. | 9 |

## `mlflow/tracing/utils/search.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `traces_to_df` | function | `(traces: list[Trace], extract_fields: list[str] \| None) -> 'pandas.DataFrame'` | Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces" containing string representations of each... | 16 |

## `mlflow/tracing/utils/timeout.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_trace_cache_with_timeout` | function | `() -> Cache` | Return a cache object that stores traces in-memory while they are in-progress. | 28 |
| `MlflowTraceTimeoutCache` | class | `(_TimedCache)` | A different implementation of cachetools.TTLCache that logs the expired traces to the backend. | 129 |
| `MlflowTraceTimeoutCache.expire` | method | `(self, time)` | Trigger the expiration of traces that have exceeded the timeout. | 200 |
| `MlflowTraceTimeoutCache.clear` | method | `(self)` |  | 247 |

## `mlflow/tracing/utils/token.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `SpanWithToken` | class | `` | A utility container to hold an MLflow span and its corresponding OpenTelemetry token. | 8 |

## `mlflow/tracing/utils/truncation.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `set_request_response_preview` | function | `(trace_info: TraceInfo, trace_data: TraceData) -> None` | Set the request and response previews for the trace info. | 15 |

## `mlflow/tracing/utils/warning.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `LogDemotionFilter` | class | `(logging.Filter)` |  | 9 |
| `LogDemotionFilter.filter` | method | `(self, record: logging.LogRecord) -> bool` |  | 15 |
| `suppress_warning` | function | `(module: str, message: str)` | Convert the "Failed to detach context" log raised by the OpenTelemetry logger to DEBUG level so that it does not show up in the... | 31 |
| `request_id_backward_compatible` | function | `(func)` | A decorator to support backward compatibility for the `request_id` parameter, which is deprecated and replaced by the `trace_id... | 50 |

