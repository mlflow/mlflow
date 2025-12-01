import json
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class VercelAITranslator(OtelSchemaTranslator):
    """Translator for Vercel AI SDK spans."""

    # https://ai-sdk.dev/docs/ai-sdk-core/telemetry#collected-data
    INPUT_VALUE_KEYS = [
        # generateText
        "ai.prompt",
        # tool call
        "ai.toolCall.args",
        # embed
        "ai.value",
        "ai.values",
        # NB: generateText.doGenerate inputs/outputs are handled separately
    ]
    OUTPUT_VALUE_KEYS = [
        # generateText
        "ai.response.text",
        # tool call
        "ai.toolCall.result",
        # generateObject
        "ai.response.object",
        # embed
        "ai.embedding",
        "ai.embeddings",
    ]

    SPAN_KIND_ATTRIBUTE_KEY = "ai.operationId"
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "ai.generateText": SpanType.LLM,
        "ai.generateText.doGenerate": SpanType.LLM,
        "ai.toolCall": SpanType.TOOL,
        "ai.streamText": SpanType.LLM,
        "ai.streamText.doStream": SpanType.LLM,
        "ai.generateObject": SpanType.LLM,
        "ai.generateObject.doGenerate": SpanType.LLM,
        "ai.streamObject": SpanType.LLM,
        "ai.streamObject.doStream": SpanType.LLM,
        "ai.embed": SpanType.EMBEDDING,
        "ai.embed.doEmbed": SpanType.EMBEDDING,
        "ai.embedMany": SpanType.EMBEDDING,
    }

    def get_input_value(self, attributes: dict[str, Any]) -> Any:
        if self._is_chat_span(attributes):
            inputs = self._unpack_attributes_with_prefix(attributes, "ai.prompt.")
            if "tools" in inputs:
                inputs["tools"] = [self._safe_load_json(tool) for tool in inputs["tools"]]
            # Record the message format for the span for chat UI rendering
            attributes[SpanAttributeKey.MESSAGE_FORMAT] = "vercel_ai"
            return json.dumps(inputs) if inputs else None
        return super().get_input_value(attributes)

    def get_output_value(self, attributes: dict[str, Any]) -> Any:
        if self._is_chat_span(attributes):
            outputs = self._unpack_attributes_with_prefix(attributes, "ai.response.")
            return json.dumps(outputs) if outputs else None
        return super().get_output_value(attributes)

    def _unpack_attributes_with_prefix(
        self, attributes: dict[str, Any], prefix: str
    ) -> dict[str, Any]:
        result = {}
        for key, value in attributes.items():
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                result[suffix] = self._safe_load_json(value)
        return result

    def _safe_load_json(self, value: Any, max_depth: int = 2) -> Any | None:
        if not isinstance(value, str):
            return value

        try:
            result = json.loads(value)
            if max_depth > 0:
                return self._safe_load_json(result, max_depth - 1)
            return result
        except json.JSONDecodeError:
            return value

    def _is_chat_span(self, attributes: dict[str, Any]) -> bool:
        span_kind = self._safe_load_json(attributes.get(self.SPAN_KIND_ATTRIBUTE_KEY))
        return span_kind in ["ai.generateText.doGenerate", "ai.streamText.doStream"]
