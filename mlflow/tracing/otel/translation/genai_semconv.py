"""
Translation utilities for GenAI (Generic AI) semantic conventions.

Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class GenAiTranslator(OtelSchemaTranslator):
    """
    Translator for GenAI semantic conventions.

    Only defines the attribute keys. All translation logic is inherited from the base class.

    Note: GenAI semantic conventions don't define a total_tokens field,
    so TOTAL_TOKEN_KEY is left as None (inherited from base).
    """

    # OpenTelemetry GenAI semantic conventions span kind attribute key
    # Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference
    SPAN_KIND_ATTRIBUTE_KEY = "gen_ai.operation.name"

    # Mapping from OpenTelemetry GenAI semantic conventions span kinds to MLflow span types
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "chat": SpanType.CHAT_MODEL,
        "create_agent": SpanType.AGENT,
        "embeddings": SpanType.EMBEDDING,
        "execute_tool": SpanType.TOOL,
        "generate_content": SpanType.LLM,
        "invoke_agent": SpanType.AGENT,
        "text_completion": SpanType.LLM,
        "response": SpanType.LLM,
    }

    # Token usage attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#genai-attributes
    INPUT_TOKEN_KEY = "gen_ai.usage.input_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.output_tokens"

    # Input/Output attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#gen-ai-input-messages
    INPUT_VALUE_KEYS = ["gen_ai.input.messages", "gen_ai.tool.call.arguments"]
    OUTPUT_VALUE_KEYS = ["gen_ai.output.messages", "gen_ai.tool.call.result"]
