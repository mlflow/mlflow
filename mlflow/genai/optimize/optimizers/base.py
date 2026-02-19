from abc import ABC, abstractmethod
from typing import Any, Callable

from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput
from mlflow.utils.annotations import experimental

# The evaluation function that takes candidate prompts as a dict
# (prompt template name -> prompt template) and a dataset as a list of dicts,
# and returns a list of EvaluationResultRecord.
_EvalFunc = Callable[[dict[str, str], list[dict[str, Any]]], list[EvaluationResultRecord]]


@experimental(version="3.5.0")
class BasePromptOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
    ) -> PromptOptimizerOutput:
        """
        Optimize the target prompts using the given evaluation function,
        dataset and target prompt templates.

        Args:
            eval_fn: The evaluation function that takes candidate prompts as a dict
                (prompt template name -> prompt template) and a dataset as a list of dicts,
                and returns a list of EvaluationResultRecord. Note that eval_fn is not thread-safe.
            train_data: The dataset to use for optimization. Each record should
                include the inputs and outputs fields with dict values.
            target_prompts: The target prompt templates to use. The key is the prompt template
                name and the value is the prompt template.
            enable_tracking: If True (default), automatically log optimization progress.

        Returns:
            The outputs of the prompt optimizer that includes the optimized prompts
            as a dict (prompt template name -> prompt template).
        """
