from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL


def _check_trulens_installed():
    try:
        import trulens  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "TruLens scorers require the 'trulens' package. "
            "Install it with: pip install trulens trulens-providers-openai"
        )


class DatabricksTruLensProvider:
    """
    TruLens provider adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        self._model_name = _DATABRICKS_DEFAULT_JUDGE_MODEL

    def _generate(self, prompt: str) -> str:
        result = call_chat_completions(user_prompt=prompt, system_prompt="")
        return result.output

    def get_model_name(self) -> str:
        return self._model_name


class DatabricksServingEndpointTruLensProvider:
    """
    TruLens provider adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name

    def _generate(self, prompt: str) -> str:
        output = _invoke_databricks_serving_endpoint(
            model_name=self._endpoint_name,
            prompt=prompt,
            num_retries=3,
            response_format=None,
        )
        return output.response

    def get_model_name(self) -> str:
        return f"databricks:/{self._endpoint_name}"


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
        return DatabricksTruLensProvider()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return DatabricksServingEndpointTruLensProvider(endpoint_name)
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
