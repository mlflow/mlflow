# CLAUDE.md - MLflow Tracing

## Overview

MLflow Tracing provides observability for GenAI applications and traditional ML workflows. It's built on OpenTelemetry and integrates seamlessly with MLflow's experiment tracking system.

### Quick Example

```python

# Decorate a function to create a span
@mlflow.trace
def my_function():
    ...

# Can also use a context manager to create a span
with mlflow.start_span("my_span"):
   ...

# Auto-tracing for LLM providers and GenAI frameworks.
mlflow.openai.autolog()
```

## Architecture

### Data Models

- Refer to @docs/genai/tracing/data-model.mdx for the basic data models for tracing.
- The Python data classes are defined in `mlflow/entities` directory.
- The protobuf definitions are defined in `mlflow/protos/service.proto` and `mlflow/protos/assessments.proto`.
- MLflow spans are **OpenTelemetry Compatible**, i.e., they can be exported to many backends that support OTLP.

------
[Note] V2 and V3 schemas

Trace format has been updated to V3 schema since MLflow 3.0. It includes major changes in the span schema (to be compatible with OTLP) and rename of some trace info fields, such as request_id -> trace_id. The Python data models are kept backward compatible with V2 schema.
------

A few other important notes that are not covered in the user-facing documentation:
- In MLflow tracking server, `TraceInfo` is stored in the backend store (e.g., a relational table with SQLAlchemy) and `TraceData` is stored in the artifact store (e.g., S3), because of the size of the data.
- MLflow's `Span` and `LiveSpan` are wrappers around OTel's `Span` object. They provide similar interfaces as OTel spans, but provides more convenient and flexible interfaces for GenAI/LLM use cases, such as span types, inputs, outputs, etc.


### Core Code Components
- **`fluent.py`** - Public API (`@mlflow.trace`, `start_span()`, `get_last_active_trace_id()`)
- **`client.py`** - Lower-level TracingClient for programmatic access
- **`provider.py`** - Global tracer provider management and routing
- **`trace_manager.py`** - In-memory trace manager that aggregates spans into traces and manages trace lifecycle during trace generation.
- **`assessments.py`** - Feedback tracking APIs for GenAI applications, such as `mlflow.log_feedback`.
- **`processor/`** - Span processors for different export backends
- **`export/`** - Export mechanisms for various destinations
- **`display/`** - Handling in-line trace display in Jupyter/Databricks notebooks.

### Data Flow

1. User instruments code with automatic tracing (`mlflow.<flavor>.autolog`) and/or manual tracing (`@mlflow.trace` or `mlflow.start_span()`).
2. When the user-instrumented code is executed, OpenTelemetry tracer generates a span.
3. MLflow wraps the span with `LiveSpan` object.
4. OTel SDK triggers the `on_start` callback of span processor. Different span processors have different implementations of this callback, but typically they construct the initial `TraceInfo` object with basic metadata.
5. The `TraceInfo` and the root span are registered in the trace manager.
6. In-memory trace_manager aggregates any child spans until the trace is completed.
7. Once the root span is ended, the span processor's `on_end` callback is triggered for the trace. Indeed, OTel SDK triggers span processor when every span is ended, but we skip non-root spans because MLflow backend expects traces to be logged as a whole.
8. Subsequently, OTel triggers the `on_end` callback of span exporter. It pops out the whole trace from the trace manager and sends it to configured destinations (MLflow experiments, OTLP, Databricks).

### Lineage Support

Traces are associated with various entities in MLflow.

- **Experiments**: Traces are by default associated with MLflow experiments.
- **Runs**: Traces can link to MLflow run **only if** they are generated under an active MLflow run. The association is not mandatory.
- **LoggedModels**: Traces can be linked to a LoggedModel object, when it is explicitly set as an active model.
- **PromptVersions**: Traces can be linked to a PromptVersion object.
- **Evaluation**: Traces are foundation of GenAI evaluation feature defined in `mlflow/genai`. Refer to @mlflow/genai/CLAUDE.md for more details.

Most of the links are implemented as tags or request metadata, except experiment association that has a top-level field in the `TraceInfo` object.

## Design Principles

- **Simplicity**: MLflow tracing is designed to be simple and easy to use. We should not provide more than what is necessary and complex configuration knobs. Auto-tracing is a good example of this principle.
- **Performance**: MLflow tracing is designed to be performant and scalable. Tracing should not introduce visible overhead to the user's application.
- **Safety**: Tracing-related logic should not interrupt the user's application. Exceptions should be handled gracefully.
- **Modularity**: Tracing architecture is designed to be modular and extensible. For example, processors and exporters allow supporting different destinations without branching the core logic.

## Development Setup

Tracing doesn't require additional setup beyond the normal MLflow development setup, unless you are working on auto-tracing integrations. Refer to `@ml-package-versions.yaml` to install the correct versions of dependencies for auto-tracing integrations.

## Testing

### Running Tracing Tests

```bash
# Run main tracing tests
pytest tests/tracing/

# Run auto-tracing tests (e.g., OpenAI)
pytest tests/openai/test_openai_autolog.py

# Run entity tests
pytest tests/entities/test_trace_info.py

# Run backend tests (e.g., SQLAlchemyStore)
pytest tests/stores/tracking/test_sqlalchemy_store.py -k trace

# Test with debug logging enabled
pytest tests/tracing/ --log-level debug
```

### Debug Tips
```python
# Enable debug logging
import logging
logging.getLogger("mlflow.tracing").setLevel(logging.DEBUG)

# Check if the trace exporter is properly configured
from mlflow.tracing.provider import _get_trace_exporter
print(_get_trace_exporter())
```

Adding `print()` statements in the data flow is a naive but effective way to debug tracing issues.

### Common Pitfalls
- `mlflow/tracing/conftest.py` defines many fixtures and some of them are automatically used by pytest. They may add implicit side effects that are not obvious.
- Tracing includes several global states such as traces stored in the trace manager, global active span context, tracer provider, etc. If you don't refresh the state, it may cause confusing side effects between test cases. Most of the states are reset by fixtures defined in `conftest.py`, but when you add new state, make sure to reset it in the test case.
- For auto-tracing integrations, the root cause of an issue can be in the integrated library, not in MLflow. When you see an issue that only happens in the latest version of the library, visit the library's release note and check if there is a known issue or breaking change.


## Documentation

Tracing documentation is written in the `docs/genai/tracing/` directory.

## FAQ

Q. Why do we separate Trace Info and Trace Data? Can we simply add their fields to the Trace class?
A. To enable loading metadata and actual spans separately. The tree of spans can be huge (> 100 MB), especially for LLM use cases where the full prompt, retrieved documents, generated text, are recorded in each span (+ their parent spans sometimes). Sometimes we want to avoid loading the full spans data, but just need metadata attached to it. For example, in Trace UI, we list the high-level information about each trace such as name, ID, tags, and only if users click a particular trace row, it shows the full span data. Having a separate data model allows this implementation without complex handling at application level.

Q. Can I add / remove a field from Span JSON schema?
A. No. Spans must be compatible with OpenTelemetry specification, so we should not add or remove a new field. If you want to add additional information, please use attributes key-value fields with a reserved key format like these.

Q. What is the difference between Tags and Request Metadata?
A. TraceInfo object contains two different key-value fields: tags and request_metadata. The difference is that tags can be mutated after the trace is created, but request metadata is immutable once set. For example, traces can be renamed after being created, so the trace name is stored as a tag. On the other hand, source run ID (an ID of MLflow Run in which the trace is created) should not be modified, so it is stored in the request_metadata field.

Q. Does MLflow logs traces asynchronously?
A. Yes. MLflow logs traces asynchronously by default, using an internal queue with background worker thread.
