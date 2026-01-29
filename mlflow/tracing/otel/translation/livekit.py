"""
Translation utilities for LiveKit Agents semantic conventions.

LiveKit Agents (Python SDK) provides real-time AI voice agents with built-in
OpenTelemetry support. This translator maps LiveKit's span attributes to MLflow's
semantic conventions for optimal visualization.

Reference:
- https://docs.livekit.io/agents/observability/
- https://github.com/livekit/agents
"""

import json
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class LiveKitTranslator(OtelSchemaTranslator):
    """
    Translator for LiveKit Agents semantic conventions.

    LiveKit Agents generates spans for various AI agent operations including:
    - Agent sessions and turns
    - LLM inference calls
    - Speech-to-text (STT) processing
    - Text-to-speech (TTS) synthesis
    - Voice activity detection (VAD)
    - Function/tool calls

    This translator maps LiveKit's span attributes to MLflow's span types
    and extracts relevant metadata for visualization.
    """

    INPUT_VALUE_KEYS = [
        "lk.user_input",  
        "lk.user_transcript",  
        "lk.chat_ctx",  
        "lk.input_text",  
    ]
    OUTPUT_VALUE_KEYS = [
        "lk.response.text", 
        "lk.response.function_calls",  
    ]

    INPUT_TOKEN_KEY = "gen_ai.usage.input_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.output_tokens"

    SPAN_KIND_ATTRIBUTE_KEYS = ["gen_ai.operation.name"]
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "chat": SpanType.CHAT_MODEL,
        "text_completion": SpanType.LLM,
        "generate_content": SpanType.LLM,
    }

    # LiveKit-specific attribute keys for detection
    DETECTION_KEYS = [
        "lk.agent_name",  
        "lk.room_name",  
        "lk.job_id",  
        "lk.participant_identity",  
    ]

    # Message format for chat UI rendering
    MESSAGE_FORMAT = "livekit"

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
        Translate LiveKit span type to MLflow span type.

        LiveKit uses gen_ai.operation.name for LLM spans and can be inferred
        from attribute prefixes for other span types.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            MLflow span type string or None if no mapping found
        """
        if "gen_ai.request.model" in attributes:
            return SpanType.LLM

        if gen_ai_op := attributes.get("gen_ai.operation.name"):
            op_lower = str(self._decode_json_value(gen_ai_op)).lower()
            if mlflow_type := self.SPAN_KIND_TO_MLFLOW_TYPE.get(op_lower):
                return mlflow_type

        for key in attributes:
            if key == "lk.retry_count":
                return SpanType.LLM
            if key.startswith("lk.function_tool"):
                return SpanType.TOOL
            if key in ("lk.agent_name", "lk.instructions", "lk.generation_id"):
                return SpanType.AGENT
            if key.startswith("lk.tts") or key == "lk.input_text":
                return SpanType.UNKNOWN
            if key in ("lk.user_transcript", "lk.transcript_confidence", "lk.transcription_delay"):
                return SpanType.UNKNOWN

        return None

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        """
        Get message format identifier for LiveKit traces.

        Returns 'livekit' if LiveKit-specific attributes are detected,
        enabling proper chat UI rendering.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            'livekit' if LiveKit attributes detected, None otherwise
        """
        for key in self.DETECTION_KEYS:
            if key in attributes:
                return self.MESSAGE_FORMAT

        if any(key.startswith("lk.") for key in attributes):
            return self.MESSAGE_FORMAT

        return None

    def get_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        """
        Get input token count from LiveKit spans.

        LiveKit uses standard GenAI semantic conventions for token usage.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Input token count or None if not found
        """
        if value := attributes.get(self.INPUT_TOKEN_KEY):
            return int(value)
        return None

    def get_output_tokens(self, attributes: dict[str, Any]) -> int | None:
        """
        Get output token count from LiveKit spans.

        LiveKit uses standard GenAI semantic conventions for token usage.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Output token count or None if not found
        """
        if value := attributes.get(self.OUTPUT_TOKEN_KEY):
            return int(value)
        return None

    def get_input_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get input value from LiveKit/GenAI events.

        LiveKit uses GenAI semantic convention events for LLM messages:
        - gen_ai.system.message (EVENT_GEN_AI_SYSTEM_MESSAGE)
        - gen_ai.user.message (EVENT_GEN_AI_USER_MESSAGE)
        - gen_ai.assistant.message (EVENT_GEN_AI_ASSISTANT_MESSAGE)

        Args:
            events: List of span events

        Returns:
            JSON-serialized list of input messages or None if not found
        """
        messages = []

        for event in events:
            event_name = event.get("name", "")
            event_attrs = event.get("attributes", {})

            if event_name == "gen_ai.system.message":
                content = event_attrs.get("content")
                if content:
                    messages.append({"role": "system", "content": self._decode_json_value(content)})

            elif event_name == "gen_ai.user.message":
                content = event_attrs.get("content")
                if content:
                    messages.append({"role": "user", "content": self._decode_json_value(content)})

            elif event_name == "gen_ai.assistant.message":
                content = event_attrs.get("content")
                if content:
                    messages.append({"role": "assistant", "content": self._decode_json_value(content)})

        # Return JSON string for proper storage
        return json.dumps(messages) if messages else None

    def get_output_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        """
        Get output value from LiveKit/GenAI events.

        LiveKit uses GenAI semantic convention events for LLM responses:
        - gen_ai.choice (EVENT_GEN_AI_CHOICE)

        Args:
            events: List of span events

        Returns:
            JSON-serialized list of output messages or None if not found
        """
        messages = []

        for event in events:
            event_name = event.get("name", "")
            event_attrs = event.get("attributes", {})

            if event_name == "gen_ai.choice":
                role = event_attrs.get("role", "assistant")
                content = event_attrs.get("content")
                if content:
                    messages.append({
                        "role": self._decode_json_value(role),
                        "content": self._decode_json_value(content),
                    })

        # Return JSON string for proper storage
        return json.dumps(messages) if messages else None
