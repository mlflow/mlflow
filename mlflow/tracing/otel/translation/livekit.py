"""
Translation utilities for LiveKit Agents semantic conventions.

LiveKit Agents (Python SDK) provides real-time AI voice agents with built-in
OpenTelemetry support. This translator maps LiveKit's span attributes to MLflow's
semantic conventions for optimal visualization.

Reference:
- https://docs.livekit.io/agents/observability/
- https://github.com/livekit/agents
"""

from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator


class LiveKitTranslator(GenAiTranslator):
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

    Inherits from GenAiTranslator since LiveKit uses GenAI semantic conventions
    for token usage and event-based message extraction.
    """

    # LiveKit-specific input/output attribute keys (in addition to GenAI semconv)
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

    # LiveKit-specific attribute keys for detection
    DETECTION_KEYS = [
        "lk.agent_name",
        "lk.room_name",
        "lk.job_id",
        "lk.participant_identity",
    ]

    # Message format for chat UI rendering
    MESSAGE_FORMAT = "livekit"

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
