"""
Abstract base class for format-specific message converters.

Each LLM provider stores inputs/outputs in its own format (e.g. OpenAI, Anthropic).
Converters translate these provider-specific formats into the GenAI Semantic Convention
attributes: gen_ai.input.messages, gen_ai.output.messages, request params, and response attrs.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey

_logger = logging.getLogger(__name__)

# Request params extracted from the provider-native inputs dict.
_REQUEST_PARAM_KEYS = {
    "temperature": GenAiSemconvKey.REQUEST_TEMPERATURE,
    "max_tokens": GenAiSemconvKey.REQUEST_MAX_TOKENS,
    "top_p": GenAiSemconvKey.REQUEST_TOP_P,
    "stop": GenAiSemconvKey.REQUEST_STOP_SEQUENCES,
    "tools": GenAiSemconvKey.TOOL_DEFINITIONS,
}


class GenAiSemconvConverter(ABC):
    @abstractmethod
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Convert provider-native inputs to GenAI semconv input messages."""

    @abstractmethod
    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Convert provider-native outputs to GenAI semconv output messages."""

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Extract system instructions as a parts array. Returns None by default."""
        return None

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Extract request parameters (temperature, max_tokens, etc.) from inputs."""
        params: dict[str, Any] = {}
        for src_key, semc_key in _REQUEST_PARAM_KEYS.items():
            if (value := inputs.get(src_key)) is not None:
                if semc_key == GenAiSemconvKey.TOOL_DEFINITIONS:
                    value = json.dumps(value)
                elif semc_key == GenAiSemconvKey.REQUEST_STOP_SEQUENCES and isinstance(value, str):
                    value = [value]
                params[semc_key] = value
        return params

    def extract_response_attrs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Extract response attributes (response id, model) from outputs."""
        attrs: dict[str, Any] = {}
        if response_id := outputs.get("id"):
            attrs[GenAiSemconvKey.RESPONSE_ID] = response_id
        if model := outputs.get("model"):
            attrs[GenAiSemconvKey.RESPONSE_MODEL] = model
        return attrs

    def translate(
        self, inputs: dict[str, Any] | None, outputs: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Orchestrate conversion of inputs/outputs into GenAI semconv attributes."""
        result: dict[str, Any] = {}

        if inputs is not None:
            messages = self.convert_inputs(inputs)
            if messages is not None:
                result[GenAiSemconvKey.INPUT_MESSAGES] = json.dumps(messages)
            system_instructions = self.convert_system_instructions(inputs)
            if system_instructions is not None:
                result[GenAiSemconvKey.SYSTEM_INSTRUCTIONS] = json.dumps(system_instructions)
            result.update(self.extract_request_params(inputs))

        if outputs is not None:
            messages = self.convert_outputs(outputs)
            if messages is not None:
                result[GenAiSemconvKey.OUTPUT_MESSAGES] = json.dumps(messages)
            result.update(self.extract_response_attrs(outputs))

        return result
