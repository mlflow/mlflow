from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL


def _check_phoenix_installed():
    try:
        import phoenix.evals  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Phoenix evaluators require the 'arize-phoenix-evals' package. "
            "Install it with: pip install arize-phoenix-evals"
        )


class DatabricksPhoenixModel:
    """
    Phoenix model adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        self._model_name = _DATABRICKS_DEFAULT_JUDGE_MODEL

    def __call__(self, prompt: str, **kwargs) -> str:
        result = call_chat_completions(user_prompt=prompt, system_prompt="")
        return result.output

    def get_model_name(self) -> str:
        return self._model_name


class DatabricksServingEndpointPhoenixModel:
    """
    Phoenix model adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name

    def __call__(self, prompt: str, **kwargs) -> str:
        output = _invoke_databricks_serving_endpoint(
            model_name=self._endpoint_name,
            prompt=prompt,
            num_retries=3,
            response_format=None,
        )
        return output.response

    def get_model_name(self) -> str:
        return f"databricks:/{self._endpoint_name}"


def create_phoenix_model(model_uri: str):
    """
    Create a Phoenix model adapter from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "provider:/model" - Use provider-specific model (e.g., "openai:/gpt-4")

    Returns:
        A Phoenix-compatible model adapter

    Raises:
        MlflowException: If the model URI format is invalid
    """
    _check_phoenix_installed()

    if model_uri == "databricks":
        return DatabricksPhoenixModel()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return DatabricksServingEndpointPhoenixModel(endpoint_name)
    elif ":" in model_uri:
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")

        if provider == "openai":
            from phoenix.evals import OpenAIModel

            return OpenAIModel(model=model_name)
        elif provider == "azure_openai":
            from phoenix.evals import AzureOpenAIModel

            return AzureOpenAIModel(model=model_name)
        elif provider == "bedrock":
            from phoenix.evals import BedrockModel

            return BedrockModel(model_id=model_name)
        elif provider == "anthropic":
            from phoenix.evals import AnthropicModel

            return AnthropicModel(model=model_name)
        elif provider == "gemini":
            from phoenix.evals import GeminiModel

            return GeminiModel(model=model_name)
        elif provider == "mistral":
            from phoenix.evals import MistralAIModel

            return MistralAIModel(model=model_name)
        else:
            # Fall back to LiteLLM for other providers
            from phoenix.evals import LiteLLMModel

            return LiteLLMModel(model=f"{provider}/{model_name}")
    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid model_uri format: '{model_uri}'. "
            f"Must be 'databricks', 'databricks:/<endpoint>', or include a provider prefix "
            f"(e.g., 'openai:/gpt-4', 'anthropic:/claude-3-opus')."
        )
