# MLflow Tracing Attachments

Attach files to trace spans and store them efficiently.

## Quick Start

```python
from mlflow.tracing.attachments import Attachment
import mlflow

# Create attachment
attachment = Attachment.from_file("image.png")

# Add to trace
with mlflow.start_span("my_span") as span:
    span.set_inputs({"image": attachment})
```

## How It Works

```mermaid
sequenceDiagram
    participant User
    participant Attachment
    participant LiveSpan
    participant TraceManager
    participant Exporter
    participant ArtifactRepo

    User->>Attachment: Create attachment from file/content
    User->>LiveSpan: span.set_inputs({"file": attachment})
    LiveSpan->>Attachment: Generate reference with trace_id, span_id
    Attachment-->>LiveSpan: Return "mlflow-attachment:{encoded_json}"
    LiveSpan->>LiveSpan: Store attachment in _attachments dict
    LiveSpan->>LiveSpan: Replace attachment with reference in span data

    LiveSpan->>TraceManager: Register span (no attachments collected yet)

    Note over TraceManager: Span registration happens before set_inputs/outputs

    User->>LiveSpan: Trace completes, export triggered
    LiveSpan->>TraceManager: to_mlflow_trace() called
    TraceManager->>TraceManager: Collect attachments from all spans
    TraceManager->>Exporter: Send trace with collected attachments

    Exporter->>ArtifactRepo: Upload attachments to {artifact_uri}/attachments/
    ArtifactRepo-->>Exporter: Confirm upload
    Exporter->>Exporter: Log trace with attachment references
```

## Key Features

- **Reference-based**: Files are replaced with lightweight references in span data
- **Centralized storage**: All attachments collected at trace level for efficient export
- **Automatic upload**: Files uploaded to artifact repository during trace export
- **Error resilient**: Traces succeed even if attachment uploads fail

## Storage Location

Attachments are stored at: `{artifact_uri}/attachments/{attachment_id}`
