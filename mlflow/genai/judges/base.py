from abc import ABC, abstractmethod

from mlflow.entities.trace import Trace
from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class AlignmentOptimizer(ABC):
    """
    Abstract base class for judge alignment optimizers.

    Alignment optimizers improve judge performance by learning from traces
    that contain human feedback or other alignment signals.
    """

    @abstractmethod
    def align(self, judge: "Judge", traces: list[Trace]) -> "Judge":
        """
        Align a judge using the provided traces.

        Args:
            judge: The judge to be optimized
            traces: List of traces containing alignment data (assessments, feedback)

        Returns:
            A new optimized Judge instance

        Raises:
            MlflowException: If alignment fails or insufficient data is provided
        """


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
