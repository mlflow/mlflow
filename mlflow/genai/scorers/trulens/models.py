from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
            "Install it with: `pip install trulens trulens-providers-litellm`"
        )


def _create_databricks_managed_judge_provider(**kwargs: Any):
    """Create a TruLens provider that uses Databricks managed judge."""
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksManagedJudgeProvider(LLMProvider):
        def __init__(self):
            endpoint = Endpoint(name="databricks-managed-judge")
            super().__init__(model_engine=_DATABRICKS_DEFAULT_JUDGE_MODEL, endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if messages:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )
            if prompt is None:
                prompt = ""

            result = call_chat_completions(user_prompt=prompt, system_prompt="")
            return result.output

    return DatabricksManagedJudgeProvider()


def _create_databricks_serving_endpoint_provider(endpoint_name: str, **kwargs: Any):
    """Create a TruLens provider that uses Databricks serving endpoints."""
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksServingEndpointProvider(LLMProvider):
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
            if messages:
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


def create_trulens_provider(model_uri: str, **kwargs: Any):
    """
    Create a TruLens provider from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "provider:/model" - Use LiteLLM with the specified model
        kwargs: Additional keyword arguments to pass to the provider.

    Returns:
        A TruLens-compatible provider

    Raises:
        MlflowException: If the model URI format is invalid
    """
    _check_trulens_installed()

    if model_uri == "databricks":
        return _create_databricks_managed_judge_provider(**kwargs)

    if model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return _create_databricks_serving_endpoint_provider(endpoint_name, **kwargs)

    if ":" in model_uri:
        # Use LiteLLM for all other providers
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")

        try:
            from trulens.providers.litellm import LiteLLM

            # Format model name for LiteLLM (e.g., "openai/gpt-4" or just "gpt-4")
            litellm_model = f"{provider}/{model_name}" if provider != "litellm" else model_name
            return LiteLLM(model_engine=litellm_model, **kwargs)
        except ImportError:
            raise MlflowException.invalid_parameter_value(
                "Non-Databricks providers require 'trulens-providers-litellm'. "
                "Install it with: `pip install trulens-providers-litellm`"
            )

    raise MlflowException.invalid_parameter_value(
        f"Invalid model_uri format: '{model_uri}'. "
        f"Must be 'databricks', 'databricks:/<endpoint>', or include a provider prefix "
        f"(e.g., 'openai:/gpt-4', 'litellm:/gpt-4')."
    )
