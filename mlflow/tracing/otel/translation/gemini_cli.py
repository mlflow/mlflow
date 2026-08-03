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

Additionally, this translator wraps ``gen_ai.input.messages`` and
``gen_ai.output.messages`` into the Gemini SDK format so the MLflow UI's
existing Gemini chat renderer can display them correctly:
- inputs  → ``{"contents": [...]}``
- outputs → ``{"candidates": [{"content": ...}]}``

References:
- Gemini CLI telemetry: github.com/google-gemini/gemini-cli/blob/main/docs/cli/telemetry.md
- Gemini CLI OTLP source: github.com/google-gemini/gemini-cli packages/core/src/telemetry/sdk.ts
"""

import json
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator
from mlflow.tracing.utils import try_json_loads

_GEMINI_CLI_AGENT_NAME = "gemini-cli"
_AGENT_NAME_ATTRIBUTE_KEY = "gen_ai.agent.name"

_GEMINI_OPERATION_NAME_TO_MLFLOW_TYPE: dict[str, str] = {
    "llm_call": SpanType.LLM,
    "tool_call": SpanType.TOOL,
    "agent_call": SpanType.AGENT,
}

# Message format identifier that tells the MLflow UI to use the Gemini
# chat renderer (mlflow/server/js chat-utils/gemini.ts).
_MESSAGE_FORMAT = "gemini"


class GeminiCliTranslator(GenAiTranslator):
    """
    Translator for Gemini CLI OTEL spans.

    Extends GenAiTranslator to reuse token-usage and model-name handling,
    but scopes span-type translation to Gemini CLI spans only (identified by
    ``gen_ai.agent.name == "gemini-cli"``) and maps Gemini-specific operation names.

    Also sets ``message_format = "gemini"`` and wraps input/output messages into
    the Gemini SDK format so the UI renders them with the Gemini chat view.
    """

    SPAN_KIND_TO_MLFLOW_TYPE = _GEMINI_OPERATION_NAME_TO_MLFLOW_TYPE

    def _is_gemini_cli(self, attributes: dict[str, Any]) -> bool:
        agent_name = attributes.get(_AGENT_NAME_ATTRIBUTE_KEY)
        if isinstance(agent_name, str):
            try:
                agent_name = json.loads(agent_name)
            except (json.JSONDecodeError, TypeError):
                pass
        return agent_name == _GEMINI_CLI_AGENT_NAME

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        if not self._is_gemini_cli(attributes):
            return None
        return super().translate_span_type(attributes)

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        if self._is_gemini_cli(attributes):
            return _MESSAGE_FORMAT
        return None

    def get_input_value(self, attributes: dict[str, Any]) -> Any:
        """Wrap gen_ai.input.messages in Gemini SDK format {contents: [...]}.

        Gemini CLI sets gen_ai.input.messages as either:
        - A list of GeminiContent objects: [{role, parts}, ...]
        - A plain string (for simple user prompts)

        The MLflow UI Gemini renderer (gemini.ts normalizeGeminiChatInput)
        expects {contents: [GeminiContent, ...]} or {contents: "string"}.

        For system_prompt spans, we rewrite role "user" → "system" so the
        Chat UI renders them with the correct role badge.
        """
        if not self._is_gemini_cli(attributes):
            return None
        if value := super().get_input_value(attributes):
            operation = attributes.get(self.SPAN_KIND_ATTRIBUTE_KEY)
            return self._try_wrap_input(value, operation=operation)
        return None

    def get_output_value(self, attributes: dict[str, Any]) -> Any:
        """Wrap gen_ai.output.messages in Gemini SDK format {candidates: [{content: ...}]}.

        Gemini CLI sets gen_ai.output.messages as a list of GeminiContent
        objects: [{role: "model", parts: [...]}, ...]

        The MLflow UI Gemini renderer (gemini.ts normalizeGeminiChatOutput)
        expects {candidates: [{content: {role, parts}}, ...]}.
        """
        if not self._is_gemini_cli(attributes):
            return None
        if value := super().get_output_value(attributes):
            return self._try_wrap_output(value)
        return None

    @staticmethod
    def _try_wrap_input(value: Any, operation: str | None = None) -> Any:
        decoded = try_json_loads(value)
        # Already wrapped
        if isinstance(decoded, dict) and "contents" in decoded:
            return value
        # For system_prompt spans, Gemini CLI sets role="user" on the content
        # but the Chat UI should render it as "system". Rewrite the role so
        # the system prompt gets the correct role badge.
        if operation == "system_prompt" and isinstance(decoded, list):
            decoded = [
                {**item, "role": "system"}
                if isinstance(item, dict) and item.get("role") == "user"
                else item
                for item in decoded
            ]
        # List of GeminiContent or a plain string → wrap
        if isinstance(decoded, (list, str)):
            return json.dumps({"contents": decoded})
        return value

    @staticmethod
    def _try_wrap_output(value: Any) -> Any:
        decoded = try_json_loads(value)
        # Already wrapped
        if isinstance(decoded, dict) and "candidates" in decoded:
            return value
        # List of GeminiContent objects → merge all parts into a single candidate.
        # Gemini CLI emits one content object per streaming chunk/step, but the
        # UI expects a single candidate with all parts combined.
        if isinstance(decoded, list):
            all_parts = []
            for item in decoded:
                if isinstance(item, dict) and isinstance(item.get("parts"), list):
                    all_parts.extend(item["parts"])
            if all_parts:
                return json.dumps({
                    "candidates": [{"content": {"role": "model", "parts": all_parts}}]
                })
        return value
