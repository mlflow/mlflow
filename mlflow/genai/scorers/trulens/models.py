from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.llm_backend import MLflowLLMBackend
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


def _create_gateway_provider(backend: MLflowLLMBackend, **kwargs: Any):
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    class GatewayProvider(LLMProvider):
        def __init__(self):
            endpoint = Endpoint(name=f"gateway-{backend._provider}")
            super().__init__(model_engine=backend.model_name, endpoint=endpoint)

        def _create_chat_completion(
            self,
            prompt: str | None = None,
            messages: "Sequence[dict] | None" = None,
            **kwargs,
        ) -> str:
            if messages:
                user_prompt, system_prompt = serialize_chat_messages_to_prompts(list(messages))
                input_data = user_prompt
                if system_prompt:
                    input_data = f"{system_prompt}\n\n{user_prompt}"
            else:
                input_data = prompt if prompt is not None else ""

            # Extract response_format from kwargs and pass it to the backend
            response_format = kwargs.pop("response_format", None)
            import pydantic

            rf_class = None
            if isinstance(response_format, type) and issubclass(
                response_format, pydantic.BaseModel
            ):
                rf_class = response_format

            return backend.complete(input_data, response_format=rf_class, **kwargs)

    return GatewayProvider()


def create_trulens_provider(model_uri: str, **kwargs: Any):
    """Create a TruLens provider from a model URI.

    Routing:
        - ``"databricks"`` → DatabricksManagedJudgeProvider
        - All other URIs → GatewayProvider backed by MLflowLLMBackend
    """
    _check_trulens_installed()

    if model_uri == "databricks":
        return _create_databricks_managed_judge_provider(**kwargs)

    return _create_gateway_provider(MLflowLLMBackend(model_uri), **kwargs)
