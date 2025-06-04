import multiprocessing
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from mlflow.entities import Feedback
from mlflow.entities.model_registry import Prompt
from mlflow.utils.annotations import experimental

OBJECTIVE_FN = Callable[[dict[str, Union[bool, float, str, Feedback, list[Feedback]]]], float]


@experimental
@dataclass
class PromptOptimizationResult:
    """
    Result of the :py:func:`mlflow.genai.optimize_prompt()` API.

    Args:
        prompt: A prompt entity containing the optimized template.
    """

    prompt: Prompt


@experimental
@dataclass
class LLMParams:
    """
    Parameters for configuring a LLM Model.

    Args:
        model_name: Name of the model in the format `<provider>/<model name>`.
            For example, "openai/gpt-4" or "anthropic/claude-4".
        base_uri: Optional base URI for the API endpoint. If not provided,
            the default endpoint for the provider will be used.
        temperature: Optional sampling temperature for the model's outputs.
            Higher values (e.g., 0.8) make the output more random,
            while lower values (e.g., 0.2) make it more deterministic.
    """

    model_name: str
    base_uri: Optional[str] = None
    temperature: Optional[float] = None


@experimental
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
        algorithm: The optimization algorithm to use. Default: "DSPy/MIPROv2"
        verbose: Whether to show optimizer logs during optimization. Default: False
        autolog: Whether to log the optimization parameters, datasets and metrics.
            If set to True, a MLflow run is automatically created to store them.
            Default: False
    """

    num_instruction_candidates: int = 6
    max_few_show_examples: int = 6
    num_threads: int = field(default_factory=lambda: (multiprocessing.cpu_count() or 1) * 2 + 1)
    optimizer_llm: Optional[LLMParams] = None
    algorithm: str = "DSPy/MIPROv2"
    verbose: bool = False
    autolog: bool = False
