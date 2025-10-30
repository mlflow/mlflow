"""
Translation utilities for GenAI (Generic AI) semantic conventions.

Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class GenAiTranslator(OtelSchemaTranslator):
    """
    Translator for GenAI semantic conventions.

    Only defines the attribute keys. All translation logic is inherited from the base class.

    Note: GenAI semantic conventions don't define a total_tokens field,
    so TOTAL_TOKEN_KEY is left as None (inherited from base).
    """

    # Token usage attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#genai-attributes
    INPUT_TOKEN_KEY = "gen_ai.usage.input_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.output_tokens"

    # Input/Output attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#gen-ai-input-messages
    INPUT_VALUE_KEY = "gen_ai.input.messages"
    OUTPUT_VALUE_KEY = "gen_ai.output.messages"
