"""
Translation utilities for Traceloop/OpenLLMetry semantic conventions.

Reference: https://github.com/traceloop/openllmetry/
"""

import re
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class TraceloopTranslator(OtelSchemaTranslator):
    """
    Translator for Traceloop/OpenLLMetry semantic conventions.

    Only defines the attribute keys and mappings. All translation logic
    is inherited from the base class.
    """

    # Traceloop span kind attribute key
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L301
    SPAN_KIND_ATTRIBUTE_KEY = "traceloop.span.kind"

    # Mapping from Traceloop span kinds to MLflow span types
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "workflow": SpanType.WORKFLOW,
        "task": SpanType.TASK,
        "agent": SpanType.AGENT,
        "tool": SpanType.TOOL,
        "unknown": SpanType.UNKNOWN,
    }

    # Token usage attribute keys
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py
    INPUT_TOKEN_KEY = "gen_ai.usage.prompt_tokens"
    OUTPUT_TOKEN_KEY = "gen_ai.usage.completion_tokens"
    TOTAL_TOKEN_KEY = "llm.usage.total_tokens"

    # Input/Output attribute keys
    # Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py
    INPUT_VALUE_KEYS = [
        "traceloop.entity.input",
        # https://github.com/traceloop/openllmetry/blob/cf28145905fcda3f5d90add78dbee16256a96db2/packages/opentelemetry-instrumentation-writer/opentelemetry/instrumentation/writer/span_utils.py#L153
        re.compile(r"gen_ai\.prompt\.\d+\.content"),
        # https://github.com/traceloop/openllmetry/blob/cf28145905fcda3f5d90add78dbee16256a96db2/packages/opentelemetry-instrumentation-writer/opentelemetry/instrumentation/writer/span_utils.py#L167
        re.compile(r"gen_ai\.completion\.\d+\.tool_calls\.\d+\.arguments"),
    ]
    OUTPUT_VALUE_KEYS = ["traceloop.entity.output", re.compile(r"gen_ai\.completion\.\d+\.content")]

    def get_attribute_value(
        self, attributes: dict[str, Any], valid_keys: list[str | re.Pattern] | None = None
    ) -> Any:
        """
        Get attribute value from OTEL attributes by checking whether
        the keys in valid_keys are present in the attributes.

        Args:
            attributes: Dictionary of span attributes
            valid_keys: List of attribute keys to check

        Returns:
            Attribute value or None if not found
        """
        if valid_keys:
            for key in valid_keys:
                if isinstance(key, str) and (
                    value := self._get_and_check_attribute_value(attributes, key)
                ):
                    return value
                elif isinstance(key, re.Pattern):
                    for attr_key, attr_value in attributes.items():
                        if (
                            isinstance(attr_key, str)
                            and key.match(attr_key)
                            and (value := self._get_and_check_attribute_value(attributes, attr_key))
                        ):
                            return value
