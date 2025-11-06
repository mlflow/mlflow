"""Main utilities module for judges. Maintains backwards compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.judges.base import AlignmentOptimizer

import mlflow
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    InvokeDatabricksModelOutput,
    InvokeJudgeModelHelperOutput,
)
from mlflow.genai.judges.adapters.gateway_adapter import _NATIVE_PROVIDERS
from mlflow.genai.judges.adapters.litellm_adapter import _suppress_litellm_nonfatal_errors
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


def _is_litellm_available() -> bool:
    """Check if LiteLLM is available for import."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def validate_judge_model(model_uri: str) -> None:
    """
    Validate that a judge model URI is valid and has required dependencies.

    This function performs early validation at judge construction time to provide
    fast feedback about configuration issues.

    Args:
        model_uri: The model URI to validate (e.g., "databricks", "openai:/gpt-4")

    Raises:
        MlflowException: If the model URI is invalid or required dependencies are missing.
    """
    from mlflow.exceptions import MlflowException
    from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
        _check_databricks_agents_installed,
    )
    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    # Special handling for Databricks default model
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        # Check if databricks-agents is available
        _check_databricks_agents_installed()
        return

    # Validate the URI format and extract provider
    provider, model_name = _parse_model_uri(model_uri)

    # Check if LiteLLM is required and available for non-native providers
    if provider not in _NATIVE_PROVIDERS:
        if not _is_litellm_available():
            raise MlflowException(
                f"LiteLLM is required for using '{provider}' as a provider. "
                "Please install it with: `pip install litellm`",
                error_code=INVALID_PARAMETER_VALUE,
            )


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
    "InvokeDatabricksModelOutput",
    "InvokeJudgeModelHelperOutput",
    # Gateway adapter
    "_NATIVE_PROVIDERS",
    # LiteLLM adapter
    "_suppress_litellm_nonfatal_errors",
    # Invocation utils
    "FieldExtraction",
    "invoke_judge_model",
    "get_chat_completions_with_structured_output",
    # Prompt utils
    "DatabricksLLMJudgePrompts",
    "format_prompt",
    "add_output_format_instructions",
]
