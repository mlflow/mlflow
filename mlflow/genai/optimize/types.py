import multiprocessing
from dataclasses import dataclass
from typing import Callable, Optional, Union

from mlflow.entities import Feedback
from mlflow.entities.model_registry import Prompt
from mlflow.utils.annotations import experimental

OBJECTIVE_FN = Callable[[dict[str, Union[bool, float, str, Feedback, list[Feedback]]]], float]


@experimental
@dataclass
class PromptOptimizationResult:
    prompt: Prompt


@experimental
@dataclass
class LLMParams:
    model_name: str  # <provider>/<model name>
    base_uri: Optional[str] = None
    temperature: Optional[float] = None


@experimental
@dataclass
class OptimizerConfig:
    num_instruction_candidates: int = 8
    max_few_show_examples: int = 3
    num_threads: int = ((multiprocessing.cpu_count() or 1) * 2) + 1
    optimizer_llm: Optional[LLMParams] = None
    algorithm: str = "DSPy/MIPROv2"
