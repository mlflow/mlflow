"""
Translation utilities for GenAI (Generic AI) semantic conventions.

Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

import json
from typing import Any

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
    CACHE_READ_INPUT_TOKEN_KEY = "gen_ai.usage.cache_read.input_tokens"
    CACHE_CREATION_INPUT_TOKEN_KEY = "gen_ai.usage.cache_creation.input_tokens"

    # Input/Output attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#gen-ai-input-messages
    INPUT_VALUE_KEYS = ["gen_ai.input.messages", "gen_ai.tool.call.arguments"]
    OUTPUT_VALUE_KEYS = ["gen_ai.output.messages", "gen_ai.tool.call.result"]

    # Model name attribute keys from OTEL GenAI semantic conventions
    # Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    MODEL_NAME_KEYS = ["gen_ai.response.model", "gen_ai.request.model"]
    LLM_PROVIDER_KEY = "gen_ai.provider.name"
    TOOL_DEFINITION_KEYS = ["gen_ai.tool.definitions"]

    INFERENCE_OPERATION_DETAILS_EVENT_NAME = "gen_ai.client.inference.operation.details"

    def _decode_json_value(self, value: Any) -> Any:
        """Decode JSON-serialized string values."""
        while isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                break
            if decoded == value:
                break
            value = decoded
        return value

    def _get_operation_details_event_value(self, events: list[dict[str, Any]], key: str) -> Any:
        for event in events:
            if event.get("name") != self.INFERENCE_OPERATION_DETAILS_EVENT_NAME:
                continue
            event_attrs = event.get("attributes", {})
            if value := event_attrs.get(key):
                return json.dumps(self._decode_json_value(value))
        return None

    def get_input_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get input value from GenAI semantic convention events.

        Latest GenAI semantic convention event:
        - gen_ai.client.inference.operation.details with gen_ai.input.messages attribute

        Legacy GenAI semantic convention events for LLM messages:
        - gen_ai.system.message
        - gen_ai.user.message
        - gen_ai.assistant.message

        Args:
            events: List of span events

        Returns:
            JSON-serialized list of input messages or None if not found
        """
        if value := self._get_operation_details_event_value(events, "gen_ai.input.messages"):
            return value

        messages = []

        for event in events:
            event_name = event.get("name", "")
            event_attrs = event.get("attributes", {})

            if event_name == "gen_ai.system.message":
                if content := event_attrs.get("content"):
                    messages.append({"role": "system", "content": self._decode_json_value(content)})

            elif event_name == "gen_ai.user.message":
                if content := event_attrs.get("content"):
                    messages.append({"role": "user", "content": self._decode_json_value(content)})

            elif event_name == "gen_ai.assistant.message":
                if content := event_attrs.get("content"):
                    messages.append({
                        "role": "assistant",
                        "content": self._decode_json_value(content),
                    })

        return json.dumps(messages) if messages else None

    def get_output_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get output value from GenAI semantic convention events.

        Latest GenAI semantic convention event:
        - gen_ai.client.inference.operation.details with gen_ai.output.messages attribute

        Legacy GenAI semantic convention event for LLM responses:
        - gen_ai.choice

        Args:
            events: List of span events

        Returns:
            JSON-serialized list of output messages or None if not found
        """
        if value := self._get_operation_details_event_value(events, "gen_ai.output.messages"):
            return value

        messages = []

        for event in events:
            event_name = event.get("name", "")
            event_attrs = event.get("attributes", {})

            if event_name == "gen_ai.choice":
                if content := event_attrs.get("content"):
                    role = event_attrs.get("role", "assistant")
                    messages.append({
                        "role": self._decode_json_value(role),
                        "content": self._decode_json_value(content),
                    })

        return json.dumps(messages) if messages else None
