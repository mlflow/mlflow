"""AI Gateway adapter for judge model invocation."""

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate.

    Returns:
        The JSON response string from the model.

    Raises:
        MlflowException: If the provider is not natively supported or invocation fails.
    """
    from mlflow.metrics.genai.model_utils import get_endpoint_type, score_model_on_payload

    if provider not in _NATIVE_PROVIDERS:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    return score_model_on_payload(
        model_uri=model_uri,
        payload=prompt,
        endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
    )
