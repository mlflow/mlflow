from typing import TYPE_CHECKING, Optional

from mlflow import load_prompt, register_prompt
from mlflow.entities.model_registry import Prompt
from mlflow.genai.evaluation.utils import (
    _convert_to_legacy_eval_set,
)
from mlflow.genai.optimize.optimizer import _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import (
    OBJECTIVE_FN,
    LLMParam,
    OptimizerParam,
    PromptOptimizationResult,
)
from mlflow.genai.scorers import Scorer

if TYPE_CHECKING:
    from genai.evaluation.utils import EvaluationDatasetTypes

_ALGORITHMS = {"DSPy/MIPROv2": _DSPyMIPROv2Optimizer}


def optimize_prompts(
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
            Note: Trace input is not supported for optimization. Use the objective function
            for string or Assessment type outputs.
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

            os["OPENAI_API"]="abcdef"

            def exact_match(inputs, outputs) -> bool:
                return expectations == outputs

            prompt = mlflow.register_prompt(
                name="translation",
                template='Translate the following text into the specified language. '
                'original text: {{original_text}} language: {{language}}',
            )

            dataset = mlflow.search_traces(
                return_type="pandas"
            ).rename({
                "request": "inputs",
                "assessments": "expectations",
            })

            result = mlflow.genai.optimize_prompts(
                agent_llm=LLMParam(model_name="openai/gpt-4o"),
                data=dataset,
                metric=exact_match,
                initial_prompt=prompt,
                optimizer_params=OptimizerParam(num_instruction_candidates=5),
            )

            print(result.prompt_uri)
    """
    if optimizer_params is None:
        optimizer_params = OptimizerParam()
    optimzer = _select_optimizer(optimizer_params.algorithm)

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


def _select_optimizer(algorithm: str):
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Algorithm {algorithm} is not supported. Supported algorithms are {_ALGORITHMS}."
        )

    return _ALGORITHMS[algorithm]
