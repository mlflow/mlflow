from __future__ import annotations

from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.phoenix.utils import _NoOpRateLimiter, check_phoenix_installed
from mlflow.metrics.genai.model_utils import _parse_model_uri


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


def create_phoenix_model(model_uri: str):
    """
    Create a Phoenix model adapter from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "provider:/model" - Use LiteLLM model (e.g., "openai:/gpt-4")

    Returns:
        A Phoenix-compatible model adapter

    Raises:
        MlflowException: If the model URI format is invalid
    """
    check_phoenix_installed()

    if model_uri == "databricks":
        return DatabricksPhoenixModel()

    # Parse provider:/model format using shared helper
    from phoenix.evals import LiteLLMModel

    provider, model_name = _parse_model_uri(model_uri)
    return LiteLLMModel(
        model=f"{provider}/{model_name}",
        model_kwargs={"drop_params": True},
    )
