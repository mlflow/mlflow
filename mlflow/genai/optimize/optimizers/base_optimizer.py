import abc
from typing import TYPE_CHECKING, Optional

from mlflow.entities.model_registry import PromptVersion
from mlflow.genai.optimize.types import LLMParams, ObjectiveFn, OptimizerConfig, OptimizerOutput
from mlflow.genai.scorers import Scorer
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import pandas as pd


@experimental(version="3.3.0")
class BasePromptOptimizer(abc.ABC):
    def __init__(self, optimizer_config: OptimizerConfig):
        self._optimizer_config = optimizer_config

    @abc.abstractmethod
    def optimize(
        self,
        prompt: PromptVersion,
        target_llm_params: LLMParams,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: ObjectiveFn | None = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> OptimizerOutput:
        """Optimize the given prompt using the specified configuration.

        Args:
            prompt: The prompt to optimize.
            target_llm_params: Parameters for the agent LLM.
            train_data: Training dataset for optimization.
            scorers: List of scorers to evaluate the optimization.
            objective: Optional function to compute overall performance metric.
            eval_data: Optional evaluation dataset.

        Returns:
            The optimized prompt version registered in the prompt registry as a new version.
        """

    @property
    def optimizer_config(self) -> OptimizerConfig:
        return self._optimizer_config
