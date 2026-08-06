from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.llm_backend import ScorerLLMClient
from mlflow.genai.utils.message_utils import serialize_chat_messages_to_prompts

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


def _create_databricks_provider(backend: ScorerLLMClient, **kwargs: Any):
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class DatabricksManagedJudgeProvider(LLMProvider):
        def __init__(self):
            endpoint = Endpoint(name="databricks-managed-judge")
            super().__init__(model_engine=backend.model_name, endpoint=endpoint)

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

            from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
                call_chat_completions,
            )

            result = call_chat_completions(user_prompt=user_prompt, system_prompt=system_prompt)
            return result.output

    return DatabricksManagedJudgeProvider()


def _create_gateway_provider(backend: ScorerLLMClient, **kwargs: Any):
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class GatewayProvider(LLMProvider):
        def __init__(self):
            endpoint = Endpoint(name=f"gateway-{backend.provider}")
            super().__init__(model_engine=backend.model_name, endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if not messages:
                messages = [{"role": "user", "content": prompt or ""}]

            response_format = kwargs.pop("response_format", None)
            response_format_dict = None
            if response_format is not None:
                if isinstance(response_format, type) and issubclass(
                    response_format, pydantic.BaseModel
                ):
                    from mlflow.genai.utils.message_utils import pydantic_to_response_format

                    response_format_dict = pydantic_to_response_format(response_format)

            return backend.complete(
                list(messages),
                response_format=response_format_dict,
                **kwargs,
            )

    return GatewayProvider()


def create_trulens_provider(model_uri: str, **kwargs: Any):
    """Create a TruLens provider from a model URI.

    Routing:
        - Native providers (via ``ScorerLLMClient``) -> GatewayProvider
        - All other providers -> LiteLLM fallback
    """
    _check_trulens_installed()

    backend = ScorerLLMClient(model_uri)

    if backend.route == "databricks":
        return _create_databricks_provider(backend, **kwargs)

    if backend.is_native:
        return _create_gateway_provider(backend, **kwargs)

    try:
        from trulens.providers.litellm import LiteLLM

        litellm_model = backend.model_name
        if backend.provider == "litellm":
            litellm_model = backend.raw_model_name
        return LiteLLM(model_engine=litellm_model, **kwargs)
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Non-Databricks providers require 'trulens-providers-litellm'. "
            "Install it with: `pip install trulens-providers-litellm`"
        )
