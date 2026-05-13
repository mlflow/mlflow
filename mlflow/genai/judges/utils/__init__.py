"""Main utilities module for judges. Maintains backwards compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.judges.base import AlignmentOptimizer

import mlflow
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.litellm_adapter import (
    _is_litellm_available,
    _suppress_litellm_nonfatal_errors,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils.invocation_utils import (
    FieldExtraction,
    get_chat_completions_with_structured_output,
    invoke_judge_model,
)
from mlflow.genai.judges.utils.prompt_utils import (
    DatabricksLLMJudgePrompts,
    add_output_format_instructions,
    format_prompt,
    format_type,
)
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.utils.uri import is_databricks_uri


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return _DATABRICKS_DEFAULT_JUDGE_MODEL
    else:
        return "openai:/gpt-4.1-mini"


def get_default_optimizer() -> AlignmentOptimizer:
    """
    Get the default alignment optimizer.

    Returns:
        A SIMBA alignment optimizer with no model specified (uses default model).
    """
    from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

    return SIMBAAlignmentOptimizer()


def validate_judge_model(model_uri: str) -> None:
    """
    Validate that a judge model URI is well-formed.

    For the Databricks default model, also checks that databricks-agents is installed.
    Provider support is validated at invocation time via the adapter selection logic.

    Args:
        model_uri: The model URI to validate (e.g., "databricks", "openai:/gpt-4")

    Raises:
        MlflowException: If the model URI is malformed or Databricks dependencies are missing.
    """
    from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
        _check_databricks_agents_installed,
    )
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    # Special handling for Databricks default model
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        # Check if databricks-agents is available
        _check_databricks_agents_installed()
        return

    # Validate the URI format (raises if malformed)
    _parse_model_uri(model_uri)


class CategoricalRating(StrEnum):
    """
    A categorical rating for an assessment.

    Example:
        .. code-block:: python

            from mlflow.genai.judges import CategoricalRating
            from mlflow.entities import Feedback

            # Create feedback with categorical rating
            feedback = Feedback(
                name="my_metric", value=CategoricalRating.YES, rationale="The metric is passing."
            )
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return cls.UNKNOWN


__all__ = [
    # Local functions
    "get_default_model",
    "get_default_optimizer",
    "validate_judge_model",
    "CategoricalRating",
    # Databricks adapter
    "call_chat_completions",
    # LiteLLM adapter (re-exported for backwards compatibility with external consumers)
    "_is_litellm_available",
    "_suppress_litellm_nonfatal_errors",
    # Invocation utils
    "FieldExtraction",
    "invoke_judge_model",
    "get_chat_completions_with_structured_output",
    # Prompt utils
    "DatabricksLLMJudgePrompts",
    "format_type",
    "format_prompt",
    "add_output_format_instructions",
]
