from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback


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
    """

    model_uri: str
    prompt: str | list["ChatMessage"]
    assessment_name: str
    trace: Trace | None = None
    num_retries: int = 10
    response_format: type[pydantic.BaseModel] | None = None
    use_case: str | None = None
    inference_params: dict[str, Any] | None = None

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


class BaseJudgeAdapter(ABC):
    """
    Abstract base class for judge model adapters.
    """

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

    @abstractmethod
    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        """
        Invoke the judge model using this adapter.

        Args:
            input_params: The input parameters for the invocation.

        Returns:
            The output from the invocation including feedback and metadata.

        Raises:
            MlflowException: If the invocation fails.
        """


__all__ = ["BaseJudgeAdapter", "AdapterInvocationInput", "AdapterInvocationOutput"]
