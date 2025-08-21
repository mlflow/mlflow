from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    Base class for LLM-based scorers that can be aligned with human feedback.

    Judges are specialized scorers that use LLMs to evaluate outputs based on
    configurable criteria and the results of human-provided feedback alignment.
    """

    @property
    def description(self) -> str:
        """
        Plain text description of what this judge evaluates.
        """
        raise NotImplementedError(
            "Judge.description must be implemented by subclasses or set via factory functions"
        )
