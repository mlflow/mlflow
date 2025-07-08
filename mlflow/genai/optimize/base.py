import inspect
import logging
from contextlib import contextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional, Union

from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.utils import (
    _convert_eval_set_to_df,
)
from mlflow.genai.optimize.optimizers import _BaseOptimizer, _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import (
    OBJECTIVE_FN,
    LLMParams,
    OptimizerConfig,
    PromptOptimizationResult,
)
from mlflow.genai.prompts import load_prompt
from mlflow.genai.scorers import Scorer
from mlflow.tracking.fluent import log_params, log_table, start_run
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import pandas as pd
    from genai.evaluation.utils import EvaluationDatasetTypes

_ALGORITHMS = {"DSPy/MIPROv2": _DSPyMIPROv2Optimizer}

_logger = logging.getLogger(__name__)


@experimental
def optimize_prompt(
    *,
    target_llm_params: LLMParams,
    prompt: Union[str, PromptVersion],
    train_data: "EvaluationDatasetTypes",
    scorers: list[Scorer],
    objective: Optional[OBJECTIVE_FN] = None,
    eval_data: Optional["EvaluationDatasetTypes"] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
) -> PromptOptimizationResult:
    """
    Optimize a LLM prompt using the given dataset and evaluation metrics.
    The optimized prompt template is automatically registered as a new version of the
    original prompt and included in the result.
    Currently, this API only supports DSPy's MIPROv2 optimizer.

    Args:
        target_llm_params: Parameters for the the LLM that prompt is optimized for.
            The model name must be specified in the format `<provider>/<model>`.
        prompt: The URI or Prompt object of the MLflow prompt to optimize.
            The optimized prompt is registered as a new version of the prompt.
        train_data: Training dataset used for optimization.
            The data must be one of the following formats:

            * An EvaluationDataset entity
            * Pandas DataFrame
            * Spark DataFrame
            * List of dictionaries

            The dataset must include the following columns:

            - inputs: A column containing single inputs in dict format.
              Each input should contain keys matching the variables in the prompt template.
            - expectations: A column containing a dictionary
              of ground truths for individual output fields.

        scorers: List of scorers that evaluate the inputs, outputs and expectations.
            Note: Trace input is not supported for optimization. Use inputs, outputs and
            expectations for optimization. Also, pass the `objective` argument
            when using scorers with string or :class:`~mlflow.entities.Feedback` type outputs.
        objective: A callable that computes the overall performance metric from individual
            assessments. Takes a dict mapping assessment names to assessment scores and
            returns a float value (greater is better).
        eval_data: Evaluation dataset with the same format as train_data. If not provided,
            train_data will be automatically split into training and evaluation sets.
        optimizer_config: Configuration parameters for the optimizer.

    Returns:
        PromptOptimizationResult: The optimization result including the optimized prompt.

    Example:

        .. code-block:: python

            import os
            import mlflow
            from typing import Any
            from mlflow.genai.scorers import scorer
            from mlflow.genai.optimize import OptimizerConfig, LLMParams

            os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


            @scorer
            def exact_match(expectations: dict[str, Any], outputs: dict[str, Any]) -> bool:
                return expectations == outputs


            prompt = mlflow.genai.register_prompt(
                name="qa",
                template="Answer the following question: {{question}}",
            )

            result = mlflow.genai.optimize_prompt(
                target_llm_params=LLMParams(model_name="openai/gpt-4.1-nano"),
                train_data=[
                    {"inputs": {"question": f"{i}+1"}, "expectations": {"answer": f"{i + 1}"}}
                    for i in range(100)
                ],
                scorers=[exact_match],
                prompt=prompt.uri,
                optimizer_config=OptimizerConfig(num_instruction_candidates=5),
            )

            print(result.prompt.template)
    """
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()
    optimzer = _select_optimizer(optimizer_config)
    _validate_scorers(scorers)

    train_data = _convert_eval_set_to_df(train_data)
    if eval_data is not None:
        eval_data = _convert_eval_set_to_df(eval_data)

    if isinstance(prompt, str):
        prompt: PromptVersion = load_prompt(prompt)

    with _maybe_start_autolog(optimizer_config, train_data, eval_data, prompt, target_llm_params):
        optimized_prompt = optimzer.optimize(
            prompt=prompt,
            target_llm_params=target_llm_params,
            train_data=train_data,
            scorers=scorers,
            objective=objective,
            eval_data=eval_data,
        )

    return PromptOptimizationResult(prompt=optimized_prompt)


def _select_optimizer(optimizer_config: OptimizerConfig) -> _BaseOptimizer:
    if optimizer_config.algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: '{optimizer_config.algorithm}'. "
            f"Available algorithms: {list(_ALGORITHMS.keys())}. "
            "Please choose from the supported algorithms above."
        )

    return _ALGORITHMS[optimizer_config.algorithm](optimizer_config)


def _validate_scorers(scorers: list[Scorer]) -> None:
    for scorer in scorers:
        if not isinstance(scorer, Scorer):
            raise MlflowException.invalid_parameter_value(
                f"Invalid scorer: {scorer}. Please use @scorer decorator "
                "on your function or inherit from Scorer class."
            )

        signature = inspect.signature(scorer)
        if "trace" in signature.parameters:
            raise MlflowException.invalid_parameter_value(
                f"Invalid scorer parameter: {scorer} contains 'trace' parameter. "
                "Optimization scorers can only use: inputs, outputs, or expectations. "
                "Remove 'trace' parameter from your scorer function."
            )


@contextmanager
def _maybe_start_autolog(
    optimizer_config: OptimizerConfig,
    train_data: "pd.DataFrame",
    eval_data: Optional["pd.DataFrame"],
    prompt: PromptVersion,
    target_llm_params: LLMParams,
):
    if optimizer_config.autolog:
        with start_run() as run:
            _logger.info(
                f"🚀 MLflow Run `{run.info.run_id}` started for prompt optimization! "
                "Watch the run to track the optimization progress."
            )
            log_table(train_data, "train_data.json")
            if eval_data is not None:
                log_table(eval_data, "eval_data.json")
            params = {
                "prompt_uri": prompt.uri,
                **{f"target_llm_params.{k}": v for k, v in asdict(target_llm_params).items()},
                **{f"optimizer_config.{k}": v for k, v in asdict(optimizer_config).items()},
            }
            log_params(params)
            yield
    else:
        yield
