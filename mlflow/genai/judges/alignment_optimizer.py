from abc import ABC, abstractmethod

from mlflow.entities.trace import Trace
from mlflow.utils.annotations import experimental

# Forward declaration to avoid circular imports
if False:
    from mlflow.genai.judges.base import Judge


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
