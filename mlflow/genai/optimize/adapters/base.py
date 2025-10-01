from abc import ABC, abstractmethod
from typing import Any, Callable

from mlflow.genai.optimize.types import EvaluationResultRecord, LLMParams, PromptAdapterOutput
from mlflow.utils.annotations import experimental

# The evaluation function that takes candidate prompts as a dict
# (prompt template name -> prompt template) and a dataset as a list of dicts,
# and returns a list of EvaluationResultRecord.
_EvalFunc = Callable[[dict[str, str], list[dict[str, Any]]], list[EvaluationResultRecord]]


@experimental(version="3.5.0")
class BasePromptAdapter(ABC):
    @abstractmethod
    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        optimizer_lm_params: LLMParams,
    ) -> PromptAdapterOutput:
        """
        Optimize the target prompts using the given evaluation function,
        dataset and target prompt templates.

        Args:
            eval_fn: The evaluation function that takes candidate prompts as a dict
                (prompt template name -> prompt template) and a dataset as a list of dicts,
                and returns a list of EvaluationResultRecord.
            train_data: The dataset to use for optimization. Each record should
                include the inputs and outputs fields with dict values.
            target_prompts: The target prompt templates to use. The key is the prompt template
                name and the value is the prompt template.
            optimizer_lm_params: The optimizer LLM parameters to use.

        Returns:
            The outputs of the prompt adapter that includes the optimized prompts
            as a dict (prompt template name -> prompt template).
        """
