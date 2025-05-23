from dataclasses import dataclass
from typing import Callable, Optional, Union

from mlflow.entities import Assessment
from mlflow.entities.model_registry import Prompt

OBJECTIVE_FN = Callable[[dict[str, Union[bool, float, str, Assessment]]], float]


@dataclass
class PromptOptimizationResult:
    prompt: Prompt


@dataclass
class LLMParam:
    model_name: str  # <provider>/<model name>
    base_uri: Optional[str]
    temperature: Optional[float]


@dataclass
class OptimizerParam:
    num_instruction_candidates: int = 8
    max_few_show_examples: int = 3
    num_threads: int = 16
    optimizer_llm: Optional[LLMParam] = None
    algorithm: str = "DSPy/MIPROv2"
