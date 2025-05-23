from typing import TYPE_CHECKING, Callable, Optional, Union

from mlflow import load_prompt, register_prompt
from mlflow.entities import Assessment
from mlflow.entities.model_registry import Prompt
from mlflow.genai.evaluation.utils import (
    _convert_to_legacy_eval_set,
)
from mlflow.genai.optimize.optimizer import _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import (
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
    objective: Optional[Callable[[dict[str, Union[bool, float, str, Assessment]]], float]] = None,
    eval_data: Optional["EvaluationDatasetTypes"] = None,
    optimizer_params: Optional[OptimizerParam] = None,
) -> PromptOptimizationResult:
    """
    Optimize the LLM prompts using the given dataset and evaluation metric.

    Args:
        agent_llm: Parameters of the agent LLM. The model name should be
                   specified in the form of `<provider>/<model>`.
        prompt: The uri of the MLflow prompt to optimize.
                This will be used as the initial instructions of the signature.
        train_data: The training dataset to be used for optimization.
              The format is the similar to mlflow.genai.evaluate and
              the dataset should contain the following information:

              - inputs (required): A column that contains a single input
                 in a dict format.
              - expectations (optional): A column that contains a ground truth,
                 or a dictionary of ground truths for individual output fields.
        scorers: Scorers that evaluate the inputs, outputs and expectations.
                For the optimization, we don't support trace input and
                users need tp pass onjective function to use str
                or Assessment type output.
        objective: a callble that conpute the overall performance metric
                from the individual assessments. The input for this callable is
                assessment name -> list of assessments and this function should
                return a float value where higher is better.
        eval_data: evaluation dataset with the same format as train_data.
                train_data is automatically split if eval_data is not provided.
        optimizer_params: Parameters of the optimizer.

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
