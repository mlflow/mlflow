import inspect
from typing import TYPE_CHECKING, Optional

from mlflow.entities.model_registry import Prompt
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.utils import (
    _convert_to_legacy_eval_set,
)
from mlflow.genai.optimize.optimizer import _BaseOptimizer, _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import (
    OBJECTIVE_FN,
    LLMParam,
    OptimizerParam,
    PromptOptimizationResult,
)
from mlflow.genai.scorers import Scorer
from mlflow.tracking._model_registry.fluent import load_prompt, register_prompt
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from genai.evaluation.utils import EvaluationDatasetTypes

_ALGORITHMS = {"DSPy/MIPROv2": _DSPyMIPROv2Optimizer}


@experimental
def optimize_prompt(
    *,
    agent_llm: LLMParam,
    prompt: str,
    train_data: "EvaluationDatasetTypes",
    scorers: list[Scorer],
    objective: Optional[OBJECTIVE_FN] = None,
    eval_data: Optional["EvaluationDatasetTypes"] = None,
    optimizer_params: Optional[OptimizerParam] = None,
) -> PromptOptimizationResult:
    """Optimize LLM prompts using a given dataset and evaluation metrics.
    Currently, only supports MIPROv2 optimizer of DSPy.

    Args:
        agent_llm: Parameters for the agent LLM. The model name must be specified in the format
            `<provider>/<model>`.
        prompt: The URI of the MLflow prompt to optimize. This will be used as the initial
            instructions for the signature.
        train_data: Training dataset used for optimization. The format is similar to
            mlflow.genai.evaluate and must contain:
            - inputs (required): A column containing single inputs in dict format
            - expectations (optional): A column containing ground truth values or a dictionary
              of ground truths for individual output fields
        scorers: List of scorers that evaluate the inputs, outputs and expectations.
            Note: Trace input is not supported for optimization. Use inputs, outputs and
            expectations for optimization. Also, pass the `objective` argument
            when using scorers with string or Assessment type outputs.
        objective: A callable that computes the overall performance metric from individual
            assessments. Takes a dict mapping assessment names to lists of assessments and
            returns a float value (higher is better).
        eval_data: Evaluation dataset with the same format as train_data. If not provided,
            train_data will be automatically split.
        optimizer_params: Configuration parameters for the optimizer.

    Returns:
        PromptOptimizationResult: The optimized prompt.

    Example:
        .. code:: python

            import os
            import mlflow
            from mlflow.genai.scorers import scorer
            from mlflow.genai.optimize import OptimizerParam, LLMParam

            os.environ["OPENAI_API"]="YOUR_API_KEY"

            def exact_match(expectations, outputs) -> bool:
                return expectations == outputs

            prompt = mlflow.register_prompt(
                name="qa",
                template='Answer the following question.'
                'question: {{question}}',
            )

            result = mlflow.genai.optimize_prompt(
                agent_llm=LLMParam(model_name="openai/gpt-4.1-nano"),
                train_data=[
                    {"inputs": {"question": f"{i}+1"}, "expectations": {"answer": f"{i+1}"}}
                    for i in range(100)
                ],
                scorers=[exact_match],
                prompt=prompt,
                optimizer_params=OptimizerParam(num_instruction_candidates=5),
            )

            print(result.prompt.template)
    """
    if optimizer_params is None:
        optimizer_params = OptimizerParam()
    optimzer = _select_optimizer(optimizer_params)
    _validate_scorers(scorers)

    train_data = _convert_to_legacy_eval_set(train_data)
    if eval_data is not None:
        eval_data = _convert_to_legacy_eval_set(eval_data)

    prompt: Prompt = load_prompt(prompt)

    optimized_prompt_template = optimzer.optimize(
        prompt=prompt,
        agent_lm=agent_llm,
        train_data=train_data,
        scorers=scorers,
        objective=objective,
        eval_data=eval_data,
    )

    optimized_prompt = register_prompt(
        name=prompt.name,
        template=optimized_prompt_template,
    )

    return PromptOptimizationResult(prompt=optimized_prompt)


def _select_optimizer(optimizer_params: OptimizerParam) -> _BaseOptimizer:
    if optimizer_params.algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Algorithm {optimizer_params.algorithm} is not supported. "
            f"Supported algorithms are {_ALGORITHMS}."
        )

    return _ALGORITHMS[optimizer_params.algorithm](optimizer_params)


def _validate_scorers(scorers: list[Scorer]) -> None:
    for scorer in scorers:
        if not isinstance(scorer, Scorer):
            raise MlflowException.invalid_parameter_value(
                f"Scorer {scorer} is not a valid scorer. Please use the @scorer decorator "
                "to convert a function into a scorer or inherit from the Scorer class"
            )

        signature = inspect.signature(scorer)
        if "trace" in signature.parameters:
            raise MlflowException.invalid_parameter_value(
                f"Trace input is found in Scorer {scorer}. "
                "Scorers for optimization can only use inputs, outputs or expectations."
            )
