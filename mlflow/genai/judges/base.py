from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from mlflow.entities.trace import Trace
from mlflow.genai.judges.utils import get_default_optimizer
from mlflow.genai.scorers.base import Scorer
from mlflow.telemetry.events import AlignJudgeEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class AlignmentOptimizer(ABC):
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


class JudgeField(BaseModel):
    """
    Represents a field definition for judges with name and description.

    Used to define input and output fields for judge evaluation signatures.
    """

    name: str = Field(..., description="Name of the field")
    description: str = Field(..., description="Description of what the field represents")


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    Base class for LLM-as-a-judge scorers that can be aligned with human feedback.

    Judges are specialized scorers that use LLMs to evaluate outputs based on
    configurable criteria and the results of human-provided feedback alignment.
    """

    @property
    @abstractmethod
    def instructions(self) -> str:
        """
        Plain text instructions of what this judge evaluates.
        """

    @abstractmethod
    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for this judge.

        Returns:
            List of JudgeField objects defining the input fields.
        """

    @classmethod
    def get_output_fields(cls) -> list[JudgeField]:
        """
        Get the standard output fields used by all judges.
        This is the source of truth for judge output field definitions.

        Returns:
            List of JudgeField objects defining the standard output fields.
        """
        return [
            JudgeField(name="result", description="The evaluation rating/result"),
            JudgeField(name="rationale", description="Detailed explanation for the evaluation"),
        ]

    @experimental(version="3.4.0")
    @record_usage_event(AlignJudgeEvent)
    def align(self, traces: list[Trace], optimizer: AlignmentOptimizer | None = None) -> Judge:
        """
        Align this judge with human preferences using the provided optimizer and traces.

        Args:
            traces: Training traces for alignment
            optimizer: The alignment optimizer to use. If None, uses the default SIMBA optimizer.

        Returns:
            A new Judge instance that is better aligned with the input traces.

        Note on Logging:
            By default, alignment optimization shows minimal progress information.
            To see detailed optimization output, set the optimizer's logger to DEBUG::

                import logging

                # For SIMBA optimizer (default)
                logging.getLogger("mlflow.genai.judges.optimizers.simba").setLevel(logging.DEBUG)
        """
        if optimizer is None:
            optimizer = get_default_optimizer()
        return optimizer.align(self, traces)
