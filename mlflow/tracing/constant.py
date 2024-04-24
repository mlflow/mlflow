# NB: These keys are placeholders and subject to change
class TraceMetadataKey:
    INPUTS = "mlflow.traceInputs"
    OUTPUTS = "mlflow.traceOutputs"
    SOURCE_NAME = "mlflow.source.name"
    SOURCE_TYPE = "mlflow.source.type"


class TraceTagKey:
    TRACE_NAME = "mlflow.traceName"


# A set of reserved attribute keys
class SpanAttributeKey:
    EXPERIMENT_ID = "mlflow.experimentId"
    REQUEST_ID = "mlflow.traceRequestId"
    INPUTS = "mlflow.spanInputs"
    OUTPUTS = "mlflow.spanOutputs"
    SPAN_TYPE = "mlflow.spanType"
    FUNCTION_NAME = "mlflow.spanFunctionName"


# All storage backends are guaranteed to support key values up to 250 characters
MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS = 250
TRUNCATION_SUFFIX = "..."

# Trace request ID must have the prefix "tr-" appended to the OpenTelemetry trace ID
TRACE_REQUEST_ID_PREFIX = "tr-"
