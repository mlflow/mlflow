"""Shared LLM backend for third-party scorer packages.

Provides a single routing layer so that DeepEval, RAGAS, Phoenix, and TruLens
scorers all resolve model URIs and make chat completion calls through the same
code path. Eliminates duplicated routing logic across 4 scorer factories and
adds native ``gateway:/`` support without litellm.
"""

from __future__ import annotations

import logging
from typing import Any

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.gateway.provider_registry import is_supported_provider
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.metrics.genai.model_utils import _call_llm_provider_api, _parse_model_uri

_logger = logging.getLogger(__name__)


class MLflowLLMBackend:
    """Shared LLM backend that routes model URIs to the best available path.

    Routing order:
        1. ``"databricks"`` → Databricks managed judge (``call_chat_completions``)
        2. ``"gateway:/endpoint"`` → MLflow AI Gateway (``send_chat_request``)
        3. Supported providers → native gateway provider (``_call_llm_provider_api``)
        4. Unsupported providers → litellm fallback

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

        if self._provider == "gateway":
            self._route = "gateway"
        elif is_supported_provider(self._provider):
            self._route = "native"
        else:
            self._route = "litellm"

    @property
    def model_name(self) -> str:
        if self._route == "databricks":
            return _DATABRICKS_DEFAULT_JUDGE_MODEL
        return f"{self._provider}/{self._model_name}"

    def complete(
        self,
        prompt: str,
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a chat completion call through the resolved route.

        Args:
            prompt: The prompt string to send.
            response_format: Optional Pydantic model class for structured output.
                Automatically converted to the OpenAI json_schema format.
            **kwargs: Additional parameters passed to the LLM (e.g. temperature).

        Returns:
            The model's response as a string.
        """
        if self._route == "databricks":
            return self._complete_databricks(prompt)
        elif self._route == "gateway":
            return self._complete_gateway(prompt, **kwargs)
        elif self._route == "native":
            return self._complete_native(prompt, response_format=response_format, **kwargs)
        else:
            return self._complete_litellm(prompt, response_format=response_format, **kwargs)

    def _complete_databricks(self, prompt: str) -> str:
        result = call_chat_completions(user_prompt=prompt, system_prompt="")
        return result.output

    def _complete_gateway(self, prompt: str, **kwargs: Any) -> str:
        from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
        from mlflow.genai.judges.adapters.utils import send_chat_request
        from mlflow.genai.utils.gateway_utils import get_gateway_config

        config = get_gateway_config(self._model_name)
        headers = {
            **(config.extra_headers or {}),
            MLFLOW_GATEWAY_CALLER_HEADER: GatewayCaller.JUDGE.value,
        }
        payload = {
            "model": config.endpoint_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        endpoint = f"{config.api_base.rstrip('/')}/chat/completions"
        response = send_chat_request(
            endpoint=endpoint, headers=headers, payload=payload, num_retries=3
        )
        content = response["choices"][0]["message"]["content"]
        return content[0]["text"] if isinstance(content, list) else content

    def _complete_native(
        self,
        prompt: str,
        *,
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        response_format_dict = None
        if response_format is not None:
            from mlflow.genai.discovery.utils import _pydantic_to_response_format

            response_format_dict = _pydantic_to_response_format(response_format)

        # Use the messages path when response_format is present, since
        # _call_llm_provider_api only applies response_format with messages.
        if response_format_dict is not None:
            return _call_llm_provider_api(
                self._provider,
                self._model_name,
                messages=[{"role": "user", "content": prompt}],
                eval_parameters=kwargs or None,
                response_format=response_format_dict,
            )

        return _call_llm_provider_api(
            self._provider,
            self._model_name,
            input_data=prompt,
            eval_parameters=kwargs or None,
        )

    def _complete_litellm(
        self,
        prompt: str,
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

        call_kwargs = {
            "model": f"{self._provider}/{self._model_name}",
            "messages": [{"role": "user", "content": prompt}],
            "drop_params": True,
            **kwargs,
        }
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        response = litellm.completion(**call_kwargs)
        return response.choices[0].message.content
