from typing import Optional
from mlflow.entities.model_registry import Prompt


class PromptOptimizationResult:
    prompt: Prompt


class LLMParam:
    model_name: str  # <provider>/<model name>
    base_uri: Optional[str]
    temperature: Optional[float]


class OptimizerParam:
    num_instruction_candidates: int = 8
    max_few_show_examples: int = 3
    num_threads: int = 16
    optimizer_llm: Optional[LLMParam] = None
    algorithm: str = "DSPy/MIPROv2"
