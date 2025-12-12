import multiprocessing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mlflow.entities import Feedback, Trace
from mlflow.entities.model_registry import PromptVersion
from mlflow.utils.annotations import deprecated, experimental

if TYPE_CHECKING:
    from mlflow.genai.optimize.optimizers import BasePromptOptimizer


AggregationFn = Callable[[dict[str, bool | float | str | Feedback | list[Feedback]]], float]


@deprecated(
    since="3.5.0",
)
@dataclass
class LLMParams:
    """
    Parameters for configuring a LLM Model.

    Args:
        model_name: Name of the model in the format `<provider>:/<model name>` or
            `<provider>/<model name>`. For example, "openai:/gpt-4o", "anthropic:/claude-4",
            or "openai/gpt-4o".
        base_uri: Optional base URI for the API endpoint. If not provided,
            the default endpoint for the provider will be used.
        temperature: Optional sampling temperature for the model's outputs.
            Higher values (e.g., 0.8) make the output more random,
            while lower values (e.g., 0.2) make it more deterministic.
    """

    model_name: str
    base_uri: str | None = None
    temperature: float | None = None


@deprecated(
    since="3.5.0",
)
@dataclass
class OptimizerConfig:
    """
    Configuration for prompt optimization.

    Args:
        num_instruction_candidates: Number of candidate instructions to generate
            during each optimization iteration. Higher values may lead to better
            results but increase optimization time. Default: 6
        max_few_shot_examples: Maximum number of examples to show in few-shot
            demonstrations. Default: 6
        num_threads: Number of threads to use for parallel optimization.
            Default: (number of CPU cores * 2 + 1)
        optimizer_llm: Optional LLM parameters for the teacher model. If not provided,
            the target LLM will be used as the teacher.
        algorithm: The optimization algorithm to use. When a string is provided,
            it must be one of the supported algorithms: "DSPy/MIPROv2".
            When a BasePromptOptimizer is provided, it will be used as the optimizer.
            Default: "DSPy/MIPROv2"
        verbose: Whether to show optimizer logs during optimization. Default: False
        autolog: Whether to enable automatic logging and prompt registration.
            If set to True, a MLflow run is automatically created to store optimization
            parameters, datasets and metrics, and the optimized prompt is registered.
            If set to False, the raw optimized template is returned without registration.
            Default: True
        convert_to_single_text: Whether to convert the optimized prompt to a single prompt.
            Default: True
        extract_instructions: Whether to extract instructions from the initial prompt.
            Default: True
    """

    num_instruction_candidates: int = 6
    max_few_shot_examples: int = 6
    num_threads: int = field(default_factory=lambda: (multiprocessing.cpu_count() or 1) * 2 + 1)
    optimizer_llm: LLMParams | None = None
    algorithm: str | type["BasePromptOptimizer"] = "DSPy/MIPROv2"
    verbose: bool = False
    autolog: bool = True
    convert_to_single_text: bool = True
    extract_instructions: bool = True


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
        rationales: The rationales of the evaluation result.
    """

    inputs: dict[str, Any]
    outputs: Any
    expectations: Any
    score: float
    trace: Trace
    rationales: dict[str, str]


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
