from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from mlflow.entities.trace import Trace
from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class AlignmentOptimizer(BaseModel, ABC):
    """
    Abstract base class for judge alignment optimizers.

    Alignment optimizers improve judge accuracy by learning from traces
    that contain human feedback.
    """

    @abstractmethod
    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """
        Align a judge using the provided traces.

        Args:
            judge: The judge to be optimized
            traces: List of traces containing alignment data (feedback)

        Returns:
            A new Judge instance that is better aligned with the input traces.
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
    def align(self, optimizer: AlignmentOptimizer, traces: list[Trace]) -> Judge:
        """
        Align this judge with human preferences using the provided optimizer and traces.

        Args:
            optimizer: The alignment optimizer to use
            traces: Training traces for alignment

        Returns:
            A new Judge instance that is better aligned with the input traces.
        """
        return optimizer.align(self, traces)
