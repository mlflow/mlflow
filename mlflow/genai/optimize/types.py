import multiprocessing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mlflow.entities import Feedback
from mlflow.entities.model_registry import PromptVersion
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.genai.optimize.optimizers import BasePromptOptimizer


ObjectiveFn = Callable[[dict[str, bool | float | str | Feedback | list[Feedback]]], float]


@experimental(version="3.0.0")
@dataclass
class PromptOptimizationResult:
    """
    Result of the :py:func:`mlflow.genai.optimize_prompt()` API.

    Args:
        prompt: A prompt version entity containing the optimized template.
        initial_prompt: A prompt version entity containing the initial template.
        optimizer_name: The name of the optimizer.
        final_eval_score: The final evaluation score of the optimized prompt.
        initial_eval_score: The initial evaluation score of the optimized prompt.
    """

    prompt: PromptVersion
    initial_prompt: PromptVersion
    optimizer_name: str
    final_eval_score: float | None
    initial_eval_score: float | None


@experimental(version="3.0.0")
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


@experimental(version="3.0.0")
@dataclass
class OptimizerConfig:
    """
    Configuration for prompt optimization.

    Args:
        num_instruction_candidates: Number of candidate instructions to generate
            during each optimization iteration. Higher values may lead to better
            results but increase optimization time. Default: 6
        max_few_show_examples: Maximum number of examples to show in few-shot
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
        autolog: Whether to log the optimization parameters, datasets and metrics.
            If set to True, a MLflow run is automatically created to store them.
            Default: False
        convert_to_single_text: Whether to convert the optimized prompt to a single prompt.
            Default: True
        extract_instructions: Whether to extract instructions from the initial prompt.
            Default: True
    """

    num_instruction_candidates: int = 6
    max_few_show_examples: int = 6
    num_threads: int = field(default_factory=lambda: (multiprocessing.cpu_count() or 1) * 2 + 1)
    optimizer_llm: LLMParams | None = None
    algorithm: str | type["BasePromptOptimizer"] = "DSPy/MIPROv2"
    verbose: bool = False
    autolog: bool = False
    convert_to_single_text: bool = True
    extract_instructions: bool = True


@experimental(version="3.3.0")
@dataclass(kw_only=True)
class OptimizerOutput:
    """
    Output of the `optimize` method of :py:class:`mlflow.genai.optimize.BasePromptOptimizer`.

    Args:
        optimized_prompt: The optimized prompt version entity.
        optimizer_name: The name of the optimizer.
        final_eval_score: The final evaluation score of the optimized prompt.
        initial_eval_score: The initial evaluation score of the optimized prompt.
    """

    optimized_prompt: str | dict[str, Any]
    optimizer_name: str
    final_eval_score: float | None = None
    initial_eval_score: float | None = None
