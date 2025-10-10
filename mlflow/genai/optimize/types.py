from dataclasses import dataclass
from typing import Any, Callable

from mlflow.entities import Feedback, Trace
from mlflow.entities.model_registry import PromptVersion
from mlflow.utils.annotations import experimental

ObjectiveFn = Callable[[dict[str, bool | float | str | Feedback | list[Feedback]]], float]


@experimental(version="3.5.0")
@dataclass
class EvaluationResultRecord:
    """
    The output type of `eval_fn` in the
    :py:func:`mlflow.genai.optimize.BasePromptOptimizer.optimize()` API.

    Args:
        inputs: The inputs of the evaluation.
        outputs: The outputs of the prediction function.
        expectations: The expected outputs.
        score: The score of the evaluation result.
        trace: The trace of the evaluation execution.
    """

    inputs: dict[str, Any]
    outputs: Any
    expectations: Any
    score: float
    trace: Trace


@experimental(version="3.5.0")
@dataclass
class PromptOptimizerOutput:
    """
    An output of the :py:func:`mlflow.genai.optimize.BasePromptOptimizer.optimize()` API.

    Args:
        optimized_prompts: The optimized prompts as
            a dict (prompt template name -> prompt template).
            e.g., {"question": "What is the capital of {{country}}?"}
        initial_eval_score: The evaluation score before optimization (optional).
        final_eval_score: The evaluation score after optimization (optional).
    """

    optimized_prompts: dict[str, str]
    initial_eval_score: float | None = None
    final_eval_score: float | None = None


@experimental(version="3.5.0")
@dataclass
class PromptOptimizationResult:
    """
    Result of the :py:func:`mlflow.genai.optimize_prompts()` API.

    Args:
        optimized_prompts: The optimized prompts.
        optimizer_name: The name of the optimizer.
        initial_eval_score: The evaluation score before optimization (optional).
        final_eval_score: The evaluation score after optimization (optional).
    """

    optimized_prompts: list[PromptVersion]
    optimizer_name: str
    initial_eval_score: float | None = None
    final_eval_score: float | None = None
