"""
Translator for Gemini CLI OTEL spans.

Gemini CLI (google-gemini/gemini-cli) emits OTLP traces using `gen_ai.operation.name`
for span kind (same key as the GenAI semantic conventions), but with its own operation
names that don't match the standard GenAI values.  This translator maps those
Gemini-specific names to MLflow span types.

Gemini CLI operation names:
- ``llm_call``            → LLM span
- ``tool_call``           → TOOL span
- ``agent_call``          → AGENT span
- ``user_prompt``         → (no MLflow equivalent, left as UNKNOWN)
- ``system_prompt``       → (no MLflow equivalent, left as UNKNOWN)
- ``schedule_tool_calls`` → (no MLflow equivalent, left as UNKNOWN)

Reference: https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/telemetry.md
"""

import json
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator

_GEMINI_CLI_AGENT_NAME = "gemini-cli"
_AGENT_NAME_ATTRIBUTE_KEY = "gen_ai.agent.name"

_GEMINI_OPERATION_NAME_TO_MLFLOW_TYPE: dict[str, str] = {
    "llm_call": SpanType.LLM,
    "tool_call": SpanType.TOOL,
    "agent_call": SpanType.AGENT,
}


class GeminiCliTranslator(GenAiTranslator):
    """
    Translator for Gemini CLI OTEL spans.

    Extends GenAiTranslator to reuse token-usage and model-name handling,
    but scopes span-type translation to Gemini CLI spans only (identified by
    ``gen_ai.agent.name == "gemini-cli"``) and maps Gemini-specific operation names.
    """

    SPAN_KIND_TO_MLFLOW_TYPE = _GEMINI_OPERATION_NAME_TO_MLFLOW_TYPE

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        agent_name = attributes.get(_AGENT_NAME_ATTRIBUTE_KEY)
        if isinstance(agent_name, str):
            try:
                agent_name = json.loads(agent_name)
            except (ValueError, TypeError):
                pass
        if agent_name != _GEMINI_CLI_AGENT_NAME:
            return None
        return super().translate_span_type(attributes)
