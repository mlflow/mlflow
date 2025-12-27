from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
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


def _create_databricks_managed_judge_provider():
    """
    Create a TruLens provider that uses Databricks managed judge.

    Uses call_chat_completions for the dedicated judge endpoint.
    """
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksManagedJudgeProvider(LLMProvider):
        """TruLens provider adapter for Databricks managed judge."""

        def __init__(self):
            endpoint = Endpoint(name="databricks-managed-judge")
            super().__init__(model_engine=_DATABRICKS_DEFAULT_JUDGE_MODEL, endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if prompt is None and messages:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )

            if prompt is None:
                prompt = ""

            result = call_chat_completions(user_prompt=prompt, system_prompt="")
            return result.output

    return DatabricksManagedJudgeProvider()


def _create_databricks_serving_endpoint_provider(endpoint_name: str):
    """
    Create a TruLens provider that uses Databricks serving endpoints.

    Uses _invoke_databricks_serving_endpoint for custom model endpoints.
    """
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksServingEndpointProvider(LLMProvider):
        """TruLens provider adapter for Databricks serving endpoints."""

        _databricks_endpoint_name: str = endpoint_name

        def __init__(self):
            endpoint = Endpoint(name=f"databricks-{endpoint_name}")
            super().__init__(model_engine=endpoint_name, endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if prompt is None and messages:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )

            if prompt is None:
                prompt = ""

            output = _invoke_databricks_serving_endpoint(
                model_name=self._databricks_endpoint_name,
                prompt=prompt,
                num_retries=3,
                response_format=None,
            )
            return output.response

    return DatabricksServingEndpointProvider()


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
        return _create_databricks_managed_judge_provider()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return _create_databricks_serving_endpoint_provider(endpoint_name)
    elif ":" in model_uri:
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")

        match provider:
            case "openai":
                try:
                    from trulens.providers.openai import OpenAI

                    return OpenAI(model_engine=model_name)
                except ImportError:
                    raise MlflowException.invalid_parameter_value(
                        "OpenAI provider requires 'trulens-providers-openai'. "
                        "Install it with: pip install trulens-providers-openai"
                    )
            case "litellm":
                try:
                    from trulens.providers.litellm import LiteLLM

                    return LiteLLM(model_engine=model_name)
                except ImportError:
                    raise MlflowException.invalid_parameter_value(
                        "LiteLLM provider requires 'trulens-providers-litellm'. "
                        "Install it with: pip install trulens-providers-litellm"
                    )
            case "bedrock":
                try:
                    from trulens.providers.bedrock import Bedrock

                    return Bedrock(model_id=model_name)
                except ImportError:
                    raise MlflowException.invalid_parameter_value(
                        "Bedrock provider requires 'trulens-providers-bedrock'. "
                        "Install it with: pip install trulens-providers-bedrock"
                    )
            case "cortex":
                try:
                    from trulens.providers.cortex import Cortex

                    return Cortex(model_engine=model_name)
                except ImportError:
                    raise MlflowException.invalid_parameter_value(
                        "Cortex provider requires 'trulens-providers-cortex'. "
                        "Install it with: pip install trulens-providers-cortex"
                    )
            case _:
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
