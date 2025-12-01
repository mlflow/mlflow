import json
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
    # The ordeing is important here. Child spans inherit entity.type from parent,
    # so we must check span.type first, then fallback to entity.type
    # (for root agent spans which don't have span.type)
    # Example of trace data from voltagent:
    # parent:
    #  {
    #    "name": "my-voltagent-app",
    #    "span_type": null,
    #    "attributes": {
    #      "entity.type": "agent",
    #      "span.type": null
    #    }
    #  }
    # child:
    #  {
    #    name": "llm:streamText",
    #    "span_type": "LLM",
    #    "attributes": {
    #      "entity.id": "my-voltagent-app",
    #      "entity.type": "agent",
    #      "entity.name": "my-voltagent-app",
    #      "span.type": "llm",
    #      "llm.operation": "streamText",
    #      "mlflow.spanType": "LLM"
    #    }
    #  }
    SPAN_KIND_ATTRIBUTE_KEYS = ["span.type", "entity.type"]
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
    ]

    def _decode_json_value(self, value: Any) -> Any:
        """Decode JSON-serialized string values."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        return value

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        """
        Translate VoltAgent span type to MLflow span type.

        VoltAgent uses different attributes for different span types:
        - Child spans (LLM/tool/memory): span.type attribute
        - Root agent spans: entity.type attribute (no span.type set)

        We check span.type FIRST because child spans have entity.type set to
        their parent agent's type ("agent"), not their own type. Only root
        agent spans have entity.type correctly set to "agent" without span.type.
        """
        # Check span.type first (for LLM/tool/memory child spans)
        for span_kind_key in self.SPAN_KIND_ATTRIBUTE_KEYS:
            span_type = self._decode_json_value(attributes.get(span_kind_key))
            if span_type and (mlflow_type := self.SPAN_KIND_TO_MLFLOW_TYPE.get(span_type)):
                return mlflow_type

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        """
        Get message format identifier for VoltAgent traces.

        Returns 'voltagent' if VoltAgent-specific attributes are detected.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            'voltagent' if VoltAgent attributes detected, None otherwise
        """
        for key in self.DETECTION_KEYS:
            if key in attributes:
                return self.MESSAGE_FORMAT
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
