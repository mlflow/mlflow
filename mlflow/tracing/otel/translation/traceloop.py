"""
Translation utilities for Traceloop/OpenLLMetry semantic conventions.

Reference: https://github.com/traceloop/openllmetry/
"""

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class TraceloopTranslator(OtelSchemaTranslator):
    """
    Translator for Traceloop/OpenLLMetry semantic conventions.

    Only defines the attribute keys and mappings. All translation logic
    is inherited from the base class.
    """

    # Traceloop span kind attribute key
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L301
    SPAN_KIND_ATTRIBUTE_KEY = "traceloop.span.kind"

    # Mapping from Traceloop span kinds to MLflow span types
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "workflow": SpanType.WORKFLOW,
        "task": SpanType.TASK,
        "agent": SpanType.AGENT,
        "tool": SpanType.TOOL,
        "unknown": SpanType.UNKNOWN,
    }

    # Token usage attribute keys
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py
    INPUT_TOKEN_KEY = "gen_ai.usage.prompt_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.completion_tokens"
    TOTAL_TOKEN_KEY = "llm.usage.total_tokens"

    # Input/Output attribute keys
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py
    INPUT_VALUE_KEY = "traceloop.entity.input"
    OUTPUT_VALUE_KEY = "traceloop.entity.output"
