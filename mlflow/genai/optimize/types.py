import multiprocessing
from dataclasses import dataclass
from typing import Callable, Optional, Union

from mlflow.entities import Assessment
from mlflow.entities.model_registry import Prompt
from mlflow.utils.annotations import experimental

OBJECTIVE_FN = Callable[[dict[str, Union[bool, float, str, Assessment]]], float]


@experimental
@dataclass
class PromptOptimizationResult:
    prompt: Prompt


@experimental
@dataclass
class LLMParam:
    model_name: str  # <provider>/<model name>
    base_uri: Optional[str] = None
    temperature: Optional[float] = None


@experimental
@dataclass
class OptimizerConfig:
    num_instruction_candidates: int = 8
    max_few_show_examples: int = 3
    num_threads: int = (multiprocessing.cpu_count() * 2) + 1
    optimizer_llm: Optional[LLMParam] = None
    algorithm: str = "DSPy/MIPROv2"
