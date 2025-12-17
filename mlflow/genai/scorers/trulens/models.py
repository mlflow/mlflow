from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL

if TYPE_CHECKING:
    from typing import Sequence

_logger = logging.getLogger(__name__)


def _check_trulens_installed():
    try:
        import trulens  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "TruLens scorers require the 'trulens' package. "
            "Install it with: pip install trulens trulens-providers-openai"
        )


def _create_databricks_trulens_provider(endpoint_name: str, model_name: str):
    """
    Create a TruLens provider that uses Databricks serving endpoints.

    This dynamically creates a class that inherits from TruLens LLMProvider
    and implements _create_chat_completion using Databricks APIs.
    """
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksTruLensProvider(LLMProvider):
        """TruLens provider adapter for Databricks endpoints."""

        def __init__(self):
            # Initialize without calling parent __init__ to avoid pydantic issues
            # We just need to provide the required methods
            self._endpoint_name = endpoint_name
            self._model_name = model_name
            self.model_engine = endpoint_name

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            # Convert messages to prompt if needed
            if prompt is None and messages:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )

            if prompt is None:
                prompt = ""

            try:
                output = _invoke_databricks_serving_endpoint(
                    model_name=self._endpoint_name,
                    prompt=prompt,
                    num_retries=3,
                    response_format=None,
                )
                return output.response
            except Exception as e:
                _logger.error(f"Error invoking Databricks endpoint: {e}")
                raise

    return DatabricksTruLensProvider()


def create_trulens_provider(model_uri: str):
    """
    Create a TruLens provider from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "provider:/model" - Use provider-specific model (e.g., "openai:/gpt-4")

    Returns:
        A TruLens-compatible provider

    Raises:
        MlflowException: If the model URI format is invalid
    """
    _check_trulens_installed()

    if model_uri == "databricks":
        return _create_databricks_trulens_provider(
            endpoint_name="databricks-meta-llama-3-3-70b-instruct",
            model_name=_DATABRICKS_DEFAULT_JUDGE_MODEL,
        )
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return _create_databricks_trulens_provider(
            endpoint_name=endpoint_name,
            model_name=model_uri,
        )
    elif ":" in model_uri:
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")

        if provider == "openai":
            try:
                from trulens.providers.openai import OpenAI

                return OpenAI(model_engine=model_name)
            except ImportError:
                raise MlflowException.invalid_parameter_value(
                    "OpenAI provider requires 'trulens-providers-openai'. "
                    "Install it with: pip install trulens-providers-openai"
                )
        elif provider == "litellm":
            try:
                from trulens.providers.litellm import LiteLLM

                return LiteLLM(model_engine=model_name)
            except ImportError:
                raise MlflowException.invalid_parameter_value(
                    "LiteLLM provider requires 'trulens-providers-litellm'. "
                    "Install it with: pip install trulens-providers-litellm"
                )
        elif provider == "bedrock":
            try:
                from trulens.providers.bedrock import Bedrock

                return Bedrock(model_id=model_name)
            except ImportError:
                raise MlflowException.invalid_parameter_value(
                    "Bedrock provider requires 'trulens-providers-bedrock'. "
                    "Install it with: pip install trulens-providers-bedrock"
                )
        elif provider == "cortex":
            try:
                from trulens.providers.cortex import Cortex

                return Cortex(model_engine=model_name)
            except ImportError:
                raise MlflowException.invalid_parameter_value(
                    "Cortex provider requires 'trulens-providers-cortex'. "
                    "Install it with: pip install trulens-providers-cortex"
                )
        else:
            # Fall back to LiteLLM for other providers
            try:
                from trulens.providers.litellm import LiteLLM

                return LiteLLM(model_engine=f"{provider}/{model_name}")
            except ImportError:
                raise MlflowException.invalid_parameter_value(
                    f"Provider '{provider}' requires 'trulens-providers-litellm' as fallback. "
                    "Install it with: pip install trulens-providers-litellm"
                )
    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid model_uri format: '{model_uri}'. "
            f"Must be 'databricks', 'databricks:/<endpoint>', or include a provider prefix "
            f"(e.g., 'openai:/gpt-4', 'litellm:/gpt-4')."
        )
