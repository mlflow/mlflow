"""
Agent trace scorers for goal-plan-action alignment evaluation.

These scorers analyze agent execution traces to detect internal errors and
evaluate the quality of agent reasoning, planning, and tool usage.

Based on TruLens' benchmarked goal-plan-action alignment evaluations which achieve
95% error coverage against TRAIL (compared to 55% for standard LLM judges).
See: https://arxiv.org/abs/2510.08847
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.trulens.models import create_trulens_provider
from mlflow.genai.scorers.trulens.registry import get_feedback_method_name
from mlflow.genai.scorers.trulens.utils import format_rationale
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)


class TruLensAgentScorer(Scorer):
    """
    Base class for TruLens agent trace scorers.

    Agent trace scorers evaluate the quality of agent execution traces,
    analyzing reasoning, planning, and tool usage patterns.

    Args:
        model: Model to use for evaluation
    """

    metric_name: ClassVar[str]

    _provider: Any = PrivateAttr()
    _model: str = PrivateAttr()
    _method_name: str = PrivateAttr()

    def __init__(
        self,
        model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=self.metric_name)
        model = model or get_default_model()
        self._model = model
        self._provider = create_trulens_provider(model, **kwargs)
        self._method_name = get_feedback_method_name(self.metric_name)

    def _get_trace_string(self, trace: Trace | str | None) -> str:
        if trace is None:
            raise MlflowException.invalid_parameter_value(
                "Trace is required for agent trace evaluation."
            )
        if isinstance(trace, Trace):
            return trace.to_json()
        if isinstance(trace, str):
            return trace
        raise MlflowException.invalid_parameter_value(
            f"Invalid trace type: {type(trace)}. Expected Trace or str."
        )

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=self._model,
        )
        try:
            trace_str = self._get_trace_string(trace)
            feedback_method = getattr(self._provider, self._method_name)
            score, reasons = feedback_method(trace=trace_str)

            return Feedback(
                name=self.name,
                value=score,
                rationale=format_rationale(reasons),
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "trulens"},
            )
        except MlflowException:
            raise
        except Exception as e:
            _logger.error(f"Error evaluating TruLens agent trace metric {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "trulens"},
            )


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class LogicalConsistency(TruLensAgentScorer):
    """
    Evaluates logical consistency and reasoning quality of agent traces.

    Analyzes how coherent and logically sound the agent's decision-making process
    is throughout the execution trace.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import LogicalConsistency

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[LogicalConsistency()],
        )
    """

    metric_name: ClassVar[str] = "logical_consistency"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class ExecutionEfficiency(TruLensAgentScorer):
    """
    Evaluates execution efficiency of agent traces.

    Analyzes whether the agent takes an optimal path to achieve its goal
    without unnecessary steps or redundant operations.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import ExecutionEfficiency

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[ExecutionEfficiency()],
        )
    """

    metric_name: ClassVar[str] = "execution_efficiency"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class PlanAdherence(TruLensAgentScorer):
    """
    Evaluates plan adherence of agent traces.

    Analyzes whether the agent follows its stated plan during execution
    or deviates from it.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import PlanAdherence

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[PlanAdherence()],
        )
    """

    metric_name: ClassVar[str] = "plan_adherence"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class PlanQuality(TruLensAgentScorer):
    """
    Evaluates plan quality of agent traces.

    Analyzes whether the agent's plan is well-structured, comprehensive,
    and appropriate for achieving the stated goal.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import PlanQuality

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[PlanQuality()],
        )
    """

    metric_name: ClassVar[str] = "plan_quality"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class ToolSelection(TruLensAgentScorer):
    """
    Evaluates tool selection quality of agent traces.

    Analyzes whether the agent chooses the appropriate tools for each
    step of the task.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import ToolSelection

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[ToolSelection()],
        )
    """

    metric_name: ClassVar[str] = "tool_selection"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class ToolCalling(TruLensAgentScorer):
    """
    Evaluates tool calling quality of agent traces.

    Analyzes whether the agent correctly invokes tools with appropriate
    parameters and handles tool responses properly.

    Args:
        model: {{ model }}

    Example:

    .. code-block:: python

        from mlflow.genai.scorers.trulens import ToolCalling

        traces = mlflow.search_traces(experiment_ids=["1"])
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[ToolCalling()],
        )
    """

    metric_name: ClassVar[str] = "tool_calling"
