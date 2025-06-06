import abc
from typing import TYPE_CHECKING, Optional

from mlflow.entities.model_registry import Prompt
from mlflow.genai.optimize.types import OBJECTIVE_FN, LLMParams, OptimizerConfig
from mlflow.genai.scorers import Scorer

if TYPE_CHECKING:
    import pandas as pd


class _BaseOptimizer(abc.ABC):
    def __init__(self, optimizer_config: OptimizerConfig):
        self.optimizer_config = optimizer_config

    @abc.abstractmethod
    def optimize(
        self,
        prompt: Prompt,
        target_llm_params: LLMParams,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> Prompt:
        """Optimize the given prompt using the specified configuration.

        Args:
            prompt: The prompt to optimize.
            target_llm_params: Parameters for the agent LLM.
            train_data: Training dataset for optimization.
            scorers: List of scorers to evaluate the optimization.
            objective: Optional function to compute overall performance metric.
            eval_data: Optional evaluation dataset.

        Returns:
            The optimized prompt registered in the prompt registry as a new version.
        """
