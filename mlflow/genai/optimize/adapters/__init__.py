from mlflow.genai.optimize.adapters.base import BasePromptAdapter
from mlflow.genai.optimize.adapters.gepa_adapter import GepaPromptAdapter

__all__ = ["BasePromptAdapter", "GepaPromptAdapter"]


def get_default_adapter() -> BasePromptAdapter:
    """
    Get the default prompt adapter.

    Returns:
        GepaPromptAdapter: The default GEPA-based prompt adapter with default settings.
    """
    return GepaPromptAdapter()
