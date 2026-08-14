from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)

_DATABRICKS_PROVIDERS = {"databricks", "endpoints"}


@dataclass
class AdapterInvocationInput:
    """
    Input parameters for adapter invocation.

    Attributes:
        model_uri: The full model URI (e.g., "openai:/gpt-4").
        prompt: The prompt to evaluate. Can be a string or list of ChatMessage objects.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries on transient failures.
        response_format: Optional Pydantic model class for structured output format.
        use_case: Optional use case for telemetry tracking. Only used by some adapters.
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens).
        base_url: Optional base URL to route requests through. When specified, all
            requests to the LLM provider will be routed through this URL.
        extra_headers: Optional dictionary of additional HTTP headers to include in
            requests to the LLM provider.
    """

    model_uri: str
    prompt: str | list["ChatMessage"]
    assessment_name: str
    trace: Trace | None = None
    num_retries: int = 10
    response_format: type[pydantic.BaseModel] | None = None
    use_case: str | None = None
    inference_params: dict[str, Any] | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] | None = None

    def __post_init__(self):
        self._model_provider: str | None = None
        self._model_name: str | None = None

    @property
    def model_provider(self) -> str:
        if self._model_provider is None:
            from mlflow.metrics.genai.model_utils import _parse_model_uri

            self._model_provider, self._model_name = _parse_model_uri(self.model_uri)
        return self._model_provider

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            from mlflow.metrics.genai.model_utils import _parse_model_uri

            self._model_provider, self._model_name = _parse_model_uri(self.model_uri)
        return self._model_name


@dataclass
class AdapterInvocationOutput:
    """
    Output from adapter invocation.

    Attributes:
        feedback: The feedback object with the judge's assessment.
        request_id: Optional request ID for tracking.
        num_prompt_tokens: Optional number of prompt tokens used.
        num_completion_tokens: Optional number of completion tokens used.
        cost: Optional cost of the invocation.
    """

    feedback: Feedback
    request_id: str | None = None
    num_prompt_tokens: int | None = None
    num_completion_tokens: int | None = None
    cost: float | None = None


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseJudgeAdapter(ABC):
    @classmethod
    @abstractmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        """
        Determine if this adapter can handle the given model and prompt type.

        Args:
            model_uri: The full model URI (e.g., "openai:/gpt-4").
            prompt: The prompt to evaluate (string or list of ChatMessages).

        Returns:
            True if this adapter can handle the model and prompt type, False otherwise.
        """

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        """Invoke the adapter with Databricks telemetry recording.

        Subclasses implement ``_invoke`` with the actual invocation logic.
        This method wraps it with success/failure telemetry for
        databricks/endpoints providers.
        """
        is_databricks = (
            ":/" in input_params.model_uri and input_params.model_provider in _DATABRICKS_PROVIDERS
        )

        try:
            output = self._invoke(input_params)

            if is_databricks:
                try:
                    from mlflow.genai.judges.utils.telemetry_utils import (
                        _record_judge_model_usage_success_databricks_telemetry,
                    )

                    _record_judge_model_usage_success_databricks_telemetry(
                        request_id=output.request_id,
                        model_provider=input_params.model_provider,
                        endpoint_name=input_params.model_name,
                        num_prompt_tokens=output.num_prompt_tokens,
                        num_completion_tokens=output.num_completion_tokens,
                    )
                except Exception:
                    _logger.debug("Failed to record judge model usage success telemetry")

            return output

        except MlflowException as e:
            if is_databricks:
                try:
                    from mlflow.genai.judges.utils.telemetry_utils import (
                        _record_judge_model_usage_failure_databricks_telemetry,
                    )

                    _record_judge_model_usage_failure_databricks_telemetry(
                        model_provider=input_params.model_provider,
                        endpoint_name=input_params.model_name,
                        error_code=e.error_code or "UNKNOWN",
                        error_message=str(e),
                    )
                except Exception:
                    _logger.debug("Failed to record judge model usage failure telemetry")
            raise

    @abstractmethod
    def _invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        """Subclass extension point — implement the actual invocation logic."""


__all__ = [
    "BaseJudgeAdapter",
    "AdapterInvocationInput",
    "AdapterInvocationOutput",
]
