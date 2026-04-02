from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.utils.message_utils import serialize_chat_messages_to_prompts
from mlflow.metrics.genai.model_utils import (
    _call_llm_provider_api,
    _get_provider_instance,
    _parse_model_uri,
)

if TYPE_CHECKING:
    from typing import Sequence


def _check_trulens_installed():
    try:
        import trulens  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "TruLens scorers require the 'trulens' package. "
            "Install it with: `pip install trulens trulens-providers-litellm`"
        )


def _create_databricks_managed_judge_provider(**kwargs: Any):
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
                user_prompt, system_prompt = serialize_chat_messages_to_prompts(list(messages))
                system_prompt = system_prompt or ""
            else:
                user_prompt = prompt if prompt is not None else ""
                system_prompt = ""

            result = call_chat_completions(user_prompt=user_prompt, system_prompt=system_prompt)
            return result.output

    return DatabricksManagedJudgeProvider()


def _create_gateway_provider(provider: str, model_name: str, **kwargs: Any):
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class GatewayProvider(LLMProvider):
        def __init__(self):
            endpoint = Endpoint(name=f"gateway-{provider}")
            super().__init__(model_engine=f"{provider}/{model_name}", endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if not messages:
                messages = [{"role": "user", "content": prompt or ""}]

            # TruLens passes response_format as a Pydantic class; convert it
            # to the OpenAI json_schema dict format for the gateway provider.
            response_format = kwargs.pop("response_format", None)
            response_format_dict = None
            if response_format is not None:
                if isinstance(response_format, type) and issubclass(
                    response_format, pydantic.BaseModel
                ):
                    from mlflow.genai.utils.message_utils import pydantic_to_response_format

                    response_format_dict = pydantic_to_response_format(response_format)

            return _call_llm_provider_api(
                provider,
                model_name,
                messages=list(messages),
                eval_parameters=kwargs or None,
                response_format=response_format_dict,
            )

    return GatewayProvider()


def create_trulens_provider(model_uri: str, **kwargs: Any):
    """
    Create a TruLens provider from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "provider:/model" - Providers constructable by ``_get_provider_instance``
              (openai, anthropic, gateway, databricks, etc.) use native gateway provider;
              all others fall back to LiteLLM
        kwargs: Additional arguments passed to the underlying provider

    Returns:
        A TruLens-compatible provider

    Raises:
        MlflowException: If the model URI format is invalid
    """
    _check_trulens_installed()

    if model_uri == "databricks":
        return _create_databricks_managed_judge_provider(**kwargs)

    provider, model_name = _parse_model_uri(model_uri)

    # Use native gateway provider if _get_provider_instance can construct it,
    # otherwise fall back to litellm
    try:
        _get_provider_instance(provider, model_name)
    except MlflowException:
        pass
    else:
        return _create_gateway_provider(provider, model_name, **kwargs)

    try:
        from trulens.providers.litellm import LiteLLM

        litellm_model = f"{provider}/{model_name}" if provider != "litellm" else model_name
        return LiteLLM(model_engine=litellm_model, **kwargs)
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Non-Databricks providers require 'trulens-providers-litellm'. "
            "Install it with: `pip install trulens-providers-litellm`"
        )
