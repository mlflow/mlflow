from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class LaminarTranslator(OtelSchemaTranslator):
    SPAN_KIND_ATTRIBUTE_KEY = "lmnr.span.type"

    SPAN_KIND_TO_MLFLOW_TYPE = {
        "LLM": SpanType.LLM,
        "TOOL": SpanType.TOOL,
        "DEFAULT": SpanType.CHAIN,
        "PIPELINE": SpanType.CHAIN,
        "EXECUTOR": SpanType.AGENT,
        "EVALUATOR": SpanType.EVALUATOR,
        "HUMAN_EVALUATOR": SpanType.EVALUATOR,
        "EVALUATION": SpanType.EVALUATOR,
        "CACHED": SpanType.LLM,
    }

    INPUT_VALUE_KEYS = ["lmnr.span.input"]
    OUTPUT_VALUE_KEYS = ["lmnr.span.output"]

    # Laminar uses GenAI semantic conventions for metadata and token usage
    INPUT_TOKEN_KEY = "gen_ai.usage.input_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.output_tokens"

    MODEL_NAME_KEYS = ["gen_ai.response.model", "gen_ai.request.model"]
    LLM_PROVIDER_KEY = "gen_ai.system"

    DETECTION_KEYS = ["lmnr.span.type"]

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        for key in self.DETECTION_KEYS:
            if key in attributes:
                return "openai"
        return None
