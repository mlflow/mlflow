"""Shared LLM client for scorer packages and simulator.

Provides a single routing layer so that DeepEval, RAGAS, Phoenix, TruLens
scorers and the conversation simulator all resolve model URIs and make
chat completion calls through the same code path.

Note: This is NOT intended for judge adapters, which need lower-level access
to provider infrastructure (tool calling, request transformation, full
response objects). See mlflow/genai/judges/adapters/ for that layer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.metrics.genai.model_utils import (
    _call_llm_provider_api,
    _get_provider_instance,
    _parse_model_uri,
    call_deployments_api,
)

_logger = logging.getLogger(__name__)


class ScorerLLMClient:
    """LLM client for scorers and simulator that routes model URIs to the best available path.

    Routing:
        1. ``"databricks"`` -> Databricks managed judge
        2. ``"endpoints:/..."`` -> MLflow deployments API
        3. Providers constructable by ``_get_provider_instance`` -> native gateway
        4. All others -> litellm fallback

    Args:
        model_uri: Model URI (e.g. ``"openai:/gpt-4"``, ``"gateway:/my-endpoint"``).
    """

    def __init__(self, model_uri: str):
        self._model_uri = model_uri

        if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
            self._route = "databricks"
            self._provider = None
            self._model_name = None
            return

        self._provider, self._model_name = _parse_model_uri(model_uri)

        if self._provider == "endpoints":
            self._route = "endpoints"
        else:
            try:
                _get_provider_instance(self._provider, self._model_name)
                self._route = "native"
            except MlflowException:
                self._route = "litellm"

    @property
    def model_name(self) -> str:
        if self._route == "databricks":
            return _DATABRICKS_DEFAULT_JUDGE_MODEL
        return f"{self._provider}/{self._model_name}"

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def raw_model_name(self) -> str | None:
        """The model name without provider prefix (e.g. 'gpt-4' not 'openai/gpt-4')."""
        return self._model_name

    @property
    def route(self) -> str:
        return self._route

    @property
    def is_native(self) -> bool:
        """True if using a native provider (not litellm fallback)."""
        return self._route in ("databricks", "native", "endpoints")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: type[pydantic.BaseModel] | dict[str, Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request through the resolved route.

        Args:
            messages: List of message dicts (e.g. ``[{"role": "user", "content": "..."}]``).
            response_format: Optional Pydantic model class or pre-converted
                dict for structured output.
            num_retries: Number of retries on transient failures (default 0).
            kwargs: Additional parameters passed to the LLM (e.g. temperature).

        Returns:
            The model's response content as a string.
        """
        for attempt in range(num_retries + 1):
            try:
                return self._dispatch(messages, response_format=response_format, **kwargs)
            except MlflowException:
                if attempt >= num_retries:
                    raise
                _logger.debug(
                    f"LLM call failed (attempt {attempt + 1}/{num_retries + 1}), retrying..."
                )
                time.sleep(2**attempt)

    def complete_prompt(
        self,
        prompt: str,
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for single-prompt calls.

        Wraps the prompt in a user message and calls :meth:`complete`.
        """
        return self.complete(
            [{"role": "user", "content": prompt}],
            response_format=response_format,
            **kwargs,
        )

    def _dispatch(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        if self._route == "databricks":
            return self._complete_databricks(messages)
        elif self._route == "endpoints":
            return self._complete_endpoints(messages, response_format=response_format, **kwargs)
        elif self._route == "native":
            return self._complete_native(messages, response_format=response_format, **kwargs)
        else:
            return self._complete_litellm(messages, response_format=response_format, **kwargs)

    def _complete_databricks(self, messages: list[dict[str, str]]) -> str:
        from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
            call_chat_completions,
        )

        user_prompt = messages[-1]["content"] if messages else ""
        system_prompt = ""
        if len(messages) > 1 and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
        result = call_chat_completions(user_prompt=user_prompt, system_prompt=system_prompt)
        return result.output

    def _complete_endpoints(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        response_format_dict = self._convert_response_format(response_format)
        payload: dict[str, Any] = {"messages": messages}
        if kwargs:
            payload.update(kwargs)
        if response_format_dict:
            payload["response_format"] = response_format_dict

        result = call_deployments_api(
            self._model_name,
            payload,
            endpoint_type="llm/v1/chat",
        )
        if result is None:
            raise MlflowException("Empty response from deployment endpoint")
        return result

    def _complete_native(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        response_format_dict = self._convert_response_format(response_format)
        return _call_llm_provider_api(
            self._provider,
            self._model_name,
            messages=messages,
            eval_parameters=kwargs or None,
            response_format=response_format_dict,
        )

    def _complete_litellm(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            import litellm
        except ImportError:
            raise MlflowException.invalid_parameter_value(
                f"Provider '{self._provider}' is not natively supported. "
                "Install litellm to use it: `pip install litellm`"
            )

        call_kwargs: dict[str, Any] = {
            "model": f"{self._provider}/{self._model_name}",
            "messages": messages,
            "drop_params": True,
            **kwargs,
        }
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        response = litellm.completion(**call_kwargs)
        return response.choices[0].message.content

    @staticmethod
    def _convert_response_format(
        response_format: type[pydantic.BaseModel] | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if response_format is None:
            return None
        if isinstance(response_format, dict):
            return response_format
        from mlflow.genai.utils.message_utils import pydantic_to_response_format

        return pydantic_to_response_format(response_format)
