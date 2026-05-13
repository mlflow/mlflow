"""Agentic metrics for evaluating AI agent performance."""

from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.deepeval import DeepEvalScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class TaskCompletion(DeepEvalScorer):
    """
    Evaluates whether an agent successfully completes its assigned task.

    This metric assesses the agent's ability to fully accomplish the task it was given,
    measuring how well the final output aligns with the expected task completion criteria.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import TaskCompletion

            scorer = TaskCompletion(threshold=0.7)
            feedback = scorer(trace=trace)  # trace contains inputs, outputs, and tool calls

    """

    metric_name: ClassVar[str] = "TaskCompletion"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ToolCorrectness(DeepEvalScorer):
    """
    Evaluates whether an agent uses the correct tools for the task.

    This metric assesses if the agent selected and used the appropriate tools from its
    available toolset to accomplish the given task. It compares actual tool usage against
    expected tool selections.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ToolCorrectness

            scorer = ToolCorrectness(threshold=0.8)
            feedback = scorer(
                trace=trace
            )  # trace contains inputs, tool calls, and expected tool calls

    """

    metric_name: ClassVar[str] = "ToolCorrectness"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ArgumentCorrectness(DeepEvalScorer):
    """
    Evaluates whether an agent provides correct arguments when calling tools.

    This metric assesses the accuracy of the arguments/parameters the agent passes to
    tools, ensuring the agent uses tools with appropriate and valid inputs.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ArgumentCorrectness

            scorer = ArgumentCorrectness(threshold=0.7)
            feedback = scorer(trace=trace)  # trace contains inputs and tool calls with arguments

    """

    metric_name: ClassVar[str] = "ArgumentCorrectness"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class StepEfficiency(DeepEvalScorer):
    """
    Evaluates the efficiency of an agent's steps in completing a task.

    This metric measures whether the agent takes an optimal path to task completion,
    avoiding unnecessary steps or redundant tool calls. Higher scores indicate more
    efficient agent behavior.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import StepEfficiency

            scorer = StepEfficiency(threshold=0.6)
            feedback = scorer(trace=trace)  # trace contains inputs and sequence of tool calls

    """

    metric_name: ClassVar[str] = "StepEfficiency"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class PlanAdherence(DeepEvalScorer):
    """
    Evaluates whether an agent adheres to its planned approach.

    This metric assesses how well the agent follows the plan it generated for completing
    a task. It measures the consistency between the agent's stated plan and its actual
    execution steps.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import PlanAdherence

            scorer = PlanAdherence(threshold=0.7)
            feedback = scorer(trace=trace)  # trace contains inputs, outputs, and tool calls

    """

    metric_name: ClassVar[str] = "PlanAdherence"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class PlanQuality(DeepEvalScorer):
    """
    Evaluates the quality of an agent's generated plan.

    This metric assesses whether the agent's plan is comprehensive, logical, and likely
    to achieve the desired task outcome. It evaluates plan structure before execution.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import PlanQuality

            scorer = PlanQuality(threshold=0.7)
            feedback = scorer(
                inputs="Plan a trip to Paris",
                outputs="Plan: 1) Book flights 2) Reserve hotel 3) Create itinerary",
            )

    """

    metric_name: ClassVar[str] = "PlanQuality"
