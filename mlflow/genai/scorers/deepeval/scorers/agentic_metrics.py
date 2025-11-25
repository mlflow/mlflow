"""Agentic metrics for evaluating AI agent performance."""

from __future__ import annotations

from mlflow.genai.scorers.deepeval import DeepEvalScorer


class TaskCompletion(DeepEvalScorer):
    """
    Evaluates whether an agent successfully completes its assigned task.

    This metric assesses the agent's ability to fully accomplish the task it was given,
    measuring how well the final output aligns with the expected task completion criteria.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import TaskCompletion
        >>> scorer = TaskCompletion(threshold=0.7)
        >>> feedback = scorer(
        ...     inputs="Book a flight from NYC to SF for tomorrow",
        ...     outputs="Flight booked: UA123, departs 9am tomorrow",
        ...     trace=trace,
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="TaskCompletion",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class ToolCorrectness(DeepEvalScorer):
    """
    Evaluates whether an agent uses the correct tools for the task.

    This metric assesses if the agent selected and used the appropriate tools from its
    available toolset to accomplish the given task. It compares actual tool usage against
    expected tool selections.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import ToolCorrectness
        >>> scorer = ToolCorrectness(threshold=0.8)
        >>> feedback = scorer(
        ...     inputs="Search for Python tutorials",
        ...     trace=trace,  # trace contains tool calls
        ...     expectations={"expected_tool_calls": [{"name": "web_search", ...}]},
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="ToolCorrectness",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class ArgumentCorrectness(DeepEvalScorer):
    """
    Evaluates whether an agent provides correct arguments when calling tools.

    This metric assesses the accuracy of the arguments/parameters the agent passes to
    tools, ensuring the agent uses tools with appropriate and valid inputs.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import ArgumentCorrectness
        >>> scorer = ArgumentCorrectness(threshold=0.7)
        >>> feedback = scorer(
        ...     inputs="Calculate 15% tip on $50",
        ...     trace=trace,  # trace contains tool calls with arguments
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="ArgumentCorrectness",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class StepEfficiency(DeepEvalScorer):
    """
    Evaluates the efficiency of an agent's steps in completing a task.

    This metric measures whether the agent takes an optimal path to task completion,
    avoiding unnecessary steps or redundant tool calls. Higher scores indicate more
    efficient agent behavior.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import StepEfficiency
        >>> scorer = StepEfficiency(threshold=0.6)
        >>> feedback = scorer(
        ...     inputs="Get weather in San Francisco",
        ...     trace=trace,  # trace contains sequence of tool calls
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="StepEfficiency",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class PlanAdherence(DeepEvalScorer):
    """
    Evaluates whether an agent adheres to its planned approach.

    This metric assesses how well the agent follows the plan it generated for completing
    a task. It measures the consistency between the agent's stated plan and its actual
    execution steps.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import PlanAdherence
        >>> scorer = PlanAdherence(threshold=0.7)
        >>> feedback = scorer(
        ...     inputs="Research and book a hotel",
        ...     outputs="Followed plan: 1) searched hotels 2) compared prices 3) booked",
        ...     trace=trace,
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="PlanAdherence",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class PlanQuality(DeepEvalScorer):
    """
    Evaluates the quality of an agent's generated plan.

    This metric assesses whether the agent's plan is comprehensive, logical, and likely
    to achieve the desired task outcome. It evaluates plan structure before execution.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.agentic_metrics import PlanQuality
        >>> scorer = PlanQuality(threshold=0.7)
        >>> feedback = scorer(
        ...     inputs="Plan a trip to Paris",
        ...     outputs="Plan: 1) Book flights 2) Reserve hotel 3) Create itinerary",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="PlanQuality",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )
