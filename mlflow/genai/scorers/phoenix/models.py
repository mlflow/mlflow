from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.phoenix.utils import _NoOpRateLimiter, check_phoenix_installed


# Phoenix has BaseModel in phoenix.evals.models.base, but it requires implementing
# _generate_with_extra and _async_generate_with_extra abstract methods which add complexity.
# Phoenix evaluators only require __call__ for model compatibility, so we use duck typing
# to keep the adapters simple. This mirrors how the deepeval integration works.
class DatabricksPhoenixModel:
    """
    Phoenix model adapter for Databricks managed judge.

    Uses the dedicated judge endpoint via call_chat_completions.
    """

    def __init__(self):
        self._model_name = _DATABRICKS_DEFAULT_JUDGE_MODEL
        self._verbose = False
        self._rate_limiter = _NoOpRateLimiter()

    def __call__(self, prompt, **kwargs) -> str:
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        result = call_chat_completions(user_prompt=prompt_str, system_prompt="")
        return result.output

    def get_model_name(self) -> str:
        return self._model_name


class GatewayPhoenixModel:
    """
    Phoenix model adapter for MLflow AI Gateway endpoints.

    Uses the MLflow AI Gateway to proxy requests to the configured backend.
    Routes requests through the tracking server's gateway endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        self._model_uri = f"gateway:/{endpoint_name}"
        self._verbose = False
        self._rate_limiter = _NoOpRateLimiter()

    def _get_gateway_base_url(self) -> str:
        """Get the gateway base URL from tracking URI."""
        from mlflow.environment_variables import MLFLOW_GATEWAY_URI
        from mlflow.tracking import get_tracking_uri
        from mlflow.utils.uri import append_to_uri_path, is_http_uri

        tracking_uri = MLFLOW_GATEWAY_URI.get() or get_tracking_uri()

        if not is_http_uri(tracking_uri):
            raise MlflowException(
                f"Gateway provider requires an HTTP(S) tracking URI, but got: '{tracking_uri}'. "
                "Please set MLFLOW_TRACKING_URI to a valid HTTP(S) URL "
                "(e.g., 'http://localhost:5000')."
            )

        return append_to_uri_path(tracking_uri, "gateway/mlflow/v1/")

    def __call__(self, prompt, **kwargs) -> str:
        import litellm

        api_base = self._get_gateway_base_url()
        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format
        model = f"openai/{self._endpoint_name}"

        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt_str}],
            api_base=api_base,
            api_key="mlflow-gateway-auth",  # Dummy key, gateway handles auth
        )

        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self._model_uri


class DatabricksServingEndpointPhoenixModel:
    """
    Phoenix model adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        self._verbose = False
        self._rate_limiter = _NoOpRateLimiter()

    def __call__(self, prompt, **kwargs) -> str:
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        output = _invoke_databricks_serving_endpoint(
            model_name=self._endpoint_name,
            prompt=prompt_str,
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
            - "gateway:/endpoint" - Use MLflow AI Gateway endpoint
            - "provider:/model" - Use LiteLLM model (e.g., "openai:/gpt-4")

    Returns:
        A Phoenix-compatible model adapter

    Raises:
        MlflowException: If the model URI format is invalid
    """
    check_phoenix_installed()

    if model_uri == "databricks":
        return DatabricksPhoenixModel()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return DatabricksServingEndpointPhoenixModel(endpoint_name)
    elif model_uri.startswith("gateway:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return GatewayPhoenixModel(endpoint_name)
    elif ":" in model_uri:
        from phoenix.evals import LiteLLMModel

        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")
        return LiteLLMModel(model=f"{provider}/{model_name}")
    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid model_uri format: '{model_uri}'. "
            f"Must be 'databricks', 'databricks:/<endpoint>', or include a provider prefix "
            f"(e.g., 'openai:/gpt-4', 'anthropic:/claude-3-opus')."
        )
