from abc import abstractmethod

from mlflow.entities.trace import Trace
from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental

# Forward declaration to avoid circular imports
if False:
    from mlflow.genai.judges.alignment_optimizer import AlignmentOptimizer


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    Base class for LLM-as-a-judge scorers that can be aligned with human feedback.

    Judges are specialized scorers that use LLMs to evaluate outputs based on
    configurable criteria and the results of human-provided feedback alignment.
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Plain text description of what this judge evaluates.
        """

    @experimental(version="3.4.0")
    def align(self, optimizer: "AlignmentOptimizer", traces: list[Trace]) -> "Judge":
        """
        Optimize this judge using the provided optimizer and traces.

        Args:
            optimizer: The alignment optimizer to use
            traces: Training traces for optimization

        Returns:
            A new optimized Judge instance
        """
        return optimizer.align(self, traces)
