from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class VoltAgentTranslator(OtelSchemaTranslator):
    """
    Translator for VoltAgent semantic conventions.

    VoltAgent provides clean chat-formatted messages in `agent.messages` and `llm.messages`.
    For tools, input/output are passed through as-is.
    """

    # Input/Output attribute keys
    # VoltAgent provides messages in standard chat format, no parsing needed
    INPUT_VALUE_KEYS = ["agent.messages", "llm.messages", "input"]
    OUTPUT_VALUE_KEYS = ["output"]

    # Span type mapping
    SPAN_KIND_ATTRIBUTE_KEY = "span.type"
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "agent": SpanType.AGENT,
        "llm": SpanType.LLM,
        "tool": SpanType.TOOL,
        "memory": SpanType.MEMORY,
    }

    # Message format for chat UI rendering
    MESSAGE_FORMAT = "voltagent"

    # VoltAgent-specific attribute keys for detection
    DETECTION_KEYS = [
        "voltagent.operation_id",
        "voltagent.conversation_id",
        "agent.messages",
        "llm.messages",
    ]

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        """
        Translate VoltAgent span type to MLflow span type.

        Checks both `span.type` (for LLM/tool spans) and `entity.type` (for agent spans).
        """
        # Check span.type first
        span_type = super().translate_span_type(attributes)
        if span_type:
            return span_type

        # Fallback to entity.type for agent spans
        entity_type = attributes.get("entity.type")
        if entity_type and entity_type in self.SPAN_KIND_TO_MLFLOW_TYPE:
            return self.SPAN_KIND_TO_MLFLOW_TYPE[entity_type]

        return None

    def get_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        """Get input token count."""
        return attributes.get("usage.prompt_tokens") or attributes.get("llm.usage.prompt_tokens")

    def get_output_tokens(self, attributes: dict[str, Any]) -> int | None:
        """Get output token count."""
        return attributes.get("usage.completion_tokens") or attributes.get(
            "llm.usage.completion_tokens"
        )

    def get_total_tokens(self, attributes: dict[str, Any]) -> int | None:
        """Get total token count."""
        return attributes.get("usage.total_tokens") or attributes.get("llm.usage.total_tokens")
