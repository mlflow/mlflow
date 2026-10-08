"""
Translator for Claude Code OTEL spans.

Claude Code (and the Claude Agent SDK, which runs the Claude Code CLI) emits OTLP traces
with its own span schema rather than the GenAI semantic conventions. Every span carries a
``span.type`` attribute equal to its span name, which this translator uses to identify
Claude Code spans and map them to MLflow span types:

- ``claude_code.interaction`` → AGENT span (one per user prompt, the trace root)
- ``claude_code.llm_request`` → LLM span
- ``claude_code.tool``        → TOOL span

Token usage is reported under bare keys (``input_tokens``, ``cache_read_tokens``, ...),
and prompt/tool content under Claude Code-specific attributes and the ``tool.output``
span event. Content attributes are ``<REDACTED>`` unless the user opts in via
``OTEL_LOG_USER_PROMPTS`` / ``OTEL_LOG_TOOL_DETAILS`` / ``OTEL_LOG_TOOL_CONTENT``.

Reference: https://code.claude.com/docs/en/monitoring-usage#traces-beta
"""

import json
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator
from mlflow.tracing.utils import dump_span_attribute_value, try_json_loads

_SPAN_TYPE_PREFIX = "claude_code."
_INTERACTION = "claude_code.interaction"
_LLM_REQUEST = "claude_code.llm_request"
_TOOL = "claude_code.tool"
_TOOL_EXECUTION = "claude_code.tool.execution"
_TOOL_BLOCKED_ON_USER = "claude_code.tool.blocked_on_user"
_TOOL_TYPES = {_TOOL, _TOOL_EXECUTION, _TOOL_BLOCKED_ON_USER}
_KNOWN_SPAN_TYPES = {_INTERACTION, _LLM_REQUEST} | _TOOL_TYPES
_REDACTED = "<REDACTED>"
_TOOL_OUTPUT_EVENT = "tool.output"
_TOOL_OUTPUT_EVENT_KEYS = ["output", "content", "diff"]
_TOOL_DETAIL_KEYS = ["full_command", "file_path", "skill_name", "subagent_type"]


class ClaudeCodeTranslator(OtelSchemaTranslator):
    SPAN_KIND_ATTRIBUTE_KEY = "span.type"
    SPAN_KIND_TO_MLFLOW_TYPE = {
        _INTERACTION: SpanType.AGENT,
        _LLM_REQUEST: SpanType.LLM,
        _TOOL: SpanType.TOOL,
        _TOOL_EXECUTION: SpanType.TOOL,
        _TOOL_BLOCKED_ON_USER: SpanType.TOOL,
    }
    INPUT_TOKEN_KEY = "input_tokens"
    OUTPUT_TOKEN_KEY = "output_tokens"
    CACHE_READ_INPUT_TOKEN_KEY = "cache_read_tokens"
    CACHE_CREATION_INPUT_TOKEN_KEY = "cache_creation_tokens"

    def _get_claude_code_span_type(self, attributes: dict[str, Any]) -> str | None:
        span_type = try_json_loads(attributes.get(self.SPAN_KIND_ATTRIBUTE_KEY))
        if not isinstance(span_type, str):
            return None
        if span_type.startswith(_SPAN_TYPE_PREFIX):
            return span_type if span_type in _KNOWN_SPAN_TYPES else None
        # Older/newer Claude Code versions emit bare span.type values ("interaction",
        # "llm_request", "tool", "tool.execution", "tool.blocked_on_user") while only
        # the span name carries the "claude_code." prefix. Normalize them here.
        normalized = _SPAN_TYPE_PREFIX + span_type
        return normalized if normalized in _KNOWN_SPAN_TYPES else None

    def _get_content(self, attributes: dict[str, Any], key: str) -> Any:
        value = self._get_and_check_attribute_value(attributes, key)
        if value is None or try_json_loads(value) == _REDACTED:
            return None
        return value

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        if span_type := self._get_claude_code_span_type(attributes):
            return self.SPAN_KIND_TO_MLFLOW_TYPE[span_type]
        return None

    def get_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        if self._get_claude_code_span_type(attributes) != _LLM_REQUEST:
            return None
        input_tokens = self._get_int(attributes, self.INPUT_TOKEN_KEY)
        if input_tokens is None:
            return None
        # Claude reports input_tokens excluding cache tokens. Normalize to include them,
        # consistent with mlflow.anthropic autolog and cost calculation.
        return (
            input_tokens
            + (self.get_cache_read_input_tokens(attributes) or 0)
            + (self.get_cache_creation_input_tokens(attributes) or 0)
        )

    def get_output_tokens(self, attributes: dict[str, Any]) -> int | None:
        if self._get_claude_code_span_type(attributes) == _LLM_REQUEST:
            return self._get_int(attributes, self.OUTPUT_TOKEN_KEY)
        return None

    def get_cache_read_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        if self._get_claude_code_span_type(attributes) == _LLM_REQUEST:
            return self._get_int(attributes, self.CACHE_READ_INPUT_TOKEN_KEY)
        return None

    def get_cache_creation_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        if self._get_claude_code_span_type(attributes) == _LLM_REQUEST:
            return self._get_int(attributes, self.CACHE_CREATION_INPUT_TOKEN_KEY)
        return None

    @staticmethod
    def _get_int(attributes: dict[str, Any], key: str) -> int | None:
        value = try_json_loads(attributes.get(key))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def get_input_value(self, attributes: dict[str, Any]) -> Any:
        span_type = self._get_claude_code_span_type(attributes)
        if span_type == _INTERACTION:
            return self._get_content(attributes, "user_prompt") or self._get_content(
                attributes, "new_context"
            )
        if span_type == _LLM_REQUEST:
            return self._get_content(attributes, "new_context")
        if span_type in _TOOL_TYPES:
            if tool_input := self._get_content(attributes, "tool_input"):
                return tool_input
            details = {
                key: try_json_loads(value)
                for key in _TOOL_DETAIL_KEYS
                if (value := self._get_content(attributes, key))
            }
            return json.dumps(details) if details else None
        return None

    def get_output_value(self, attributes: dict[str, Any]) -> Any:
        span_type = self._get_claude_code_span_type(attributes)
        if span_type == _LLM_REQUEST:
            return self._get_content(attributes, "response.model_output")
        if span_type in _TOOL_TYPES:
            return self._get_content(attributes, "new_context")
        return None

    def get_output_value_from_events(self, events: list[dict[str, Any]]) -> Any:
        for event in events:
            if event.get("name") == _TOOL_OUTPUT_EVENT:
                attributes = event.get("attributes", {})
                for key in _TOOL_OUTPUT_EVENT_KEYS:
                    if value := attributes.get(key):
                        return dump_span_attribute_value(try_json_loads(value))
        return None
