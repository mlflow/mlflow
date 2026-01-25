from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.scorer_utils import (
    parse_third_party_model_uri,
    serialize_chat_messages_to_prompts,
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


def create_trulens_provider(model_uri: str, **kwargs: Any):
    """
    Create a TruLens provider from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use LiteLLM with Databricks endpoint
            - "provider:/model" - Use LiteLLM with the specified model
        kwargs: Additional arguments passed to the underlying provider

    Returns:
        A TruLens-compatible provider

    Raises:
        MlflowException: If the model URI format is invalid
    """
    _check_trulens_installed()

    provider, model_name = parse_third_party_model_uri(model_uri)

    # Use managed judge for plain "databricks" without endpoint
    if provider == "databricks" and model_name is None:
        return _create_databricks_managed_judge_provider(**kwargs)

    # Use LiteLLM for all other providers (including databricks:/endpoint)
    try:
        from trulens.providers.litellm import LiteLLM

        litellm_model = f"{provider}/{model_name}" if provider != "litellm" else model_name
        return LiteLLM(model_engine=litellm_model, **kwargs)
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Non-Databricks providers require 'trulens-providers-litellm'. "
            "Install it with: `pip install trulens-providers-litellm`"
        )
