"""
Translation utilities for Spring AI semantic conventions.

Spring AI uses OpenTelemetry GenAI semantic conventions but stores
prompt/completion content in events rather than attributes:
- gen_ai.content.prompt event with gen_ai.prompt attribute
- gen_ai.content.completion event with gen_ai.completion attribute

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
"""

from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class SpringAiTranslator(OtelSchemaTranslator):
    """
    Translator for Spring AI spans.

    Spring AI uses GenAI semantic conventions but stores prompt/completion
    in events. This translator extends the base to handle event-based
    input/output extraction.
    """

    # Spring AI uses gen_ai.operation.name for span kind (same as GenAI)
    SPAN_KIND_ATTRIBUTE_KEY = "gen_ai.operation.name"

    SPAN_KIND_TO_MLFLOW_TYPE = {
        "chat": SpanType.CHAT_MODEL,
        "embeddings": SpanType.EMBEDDING,
    }

    # Token usage attribute keys (same as GenAI semantic conventions)
    INPUT_TOKEN_KEY = "gen_ai.usage.input_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.output_tokens"

    # Spring AI doesn't use attribute-based input/output
    # Instead, it uses events (handled via get_input_value_from_events/get_output_value_from_events)
    INPUT_VALUE_KEYS = None
    OUTPUT_VALUE_KEYS = None

    # Event names for Spring AI prompt/completion content
    PROMPT_EVENT_NAME = "gen_ai.content.prompt"
    COMPLETION_EVENT_NAME = "gen_ai.content.completion"

    # Attribute keys within events
    PROMPT_ATTRIBUTE_KEY = "gen_ai.prompt"
    COMPLETION_ATTRIBUTE_KEY = "gen_ai.completion"

    def get_input_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get input value from Spring AI prompt events.

        Args:
            events: List of span events

        Returns:
            Input value or None if not found
        """
        return self._get_value_from_event(events, self.PROMPT_EVENT_NAME, self.PROMPT_ATTRIBUTE_KEY)

    def get_output_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get output value from Spring AI completion events.

        Args:
            events: List of span events

        Returns:
            Output value or None if not found
        """
        return self._get_value_from_event(
            events, self.COMPLETION_EVENT_NAME, self.COMPLETION_ATTRIBUTE_KEY
        )

    def _get_value_from_event(
        self, events: list[dict[str, Any]], event_name: str, attribute_key: str
    ) -> Any:
        """
        Extract a value from a specific event.

        Args:
            events: List of span events
            event_name: The event name to look for
            attribute_key: The attribute key within the event

        Returns:
            The attribute value or None if not found
        """
        for event in events:
            if event.get("name") == event_name:
                event_attrs = event.get("attributes", {})
                if value := event_attrs.get(attribute_key):
                    return value
        return None
