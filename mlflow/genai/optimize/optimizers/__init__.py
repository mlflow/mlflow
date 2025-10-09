from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer
from mlflow.genai.optimize.optimizers.gepa_optimizer import GepaPromptOptimizer

__all__ = ["BasePromptOptimizer", "GepaPromptOptimizer"]


def get_default_optimizer() -> BasePromptOptimizer:
    """
    Get the default prompt optimizer.

    Returns:
        GepaPromptOptimizer: The default GEPA-based prompt optimizer with default settings.
    """
    return GepaPromptOptimizer()
