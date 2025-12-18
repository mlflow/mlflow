"""
TruLens agent trace scorers for goal-plan-action alignment evaluation.

These scorers analyze agent execution traces to detect internal errors and
evaluate the quality of agent reasoning, planning, and tool usage.

The evaluators are based on TruLens' benchmarked goal-plan-action alignment
evaluations which achieve 95% error coverage against TRAIL (compared to 55%
for standard LLM judges). See: https://arxiv.org/abs/2510.08847
"""

from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


def _check_trulens_installed():
    """Check if trulens is installed and raise a helpful error if not."""
    try:
        import trulens.providers.openai  # noqa: F401

        return True
    except ImportError:
        raise MlflowException(
            "TruLens scorers require the 'trulens' package. "
            "Install it with: pip install trulens trulens-providers-openai",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _convert_mlflow_trace_to_string(trace: Trace) -> str:
    """
    Convert an MLflow Trace object to a string representation for TruLens evaluation.

    TruLens evaluators accept either a Trace object or a string representation.
    Since MLflow uses its own Trace format, we convert to JSON string which
    TruLens can process.

    Args:
        trace: An MLflow Trace object containing spans data.

    Returns:
        A JSON string representation of the trace suitable for TruLens evaluation.
    """
    return trace.to_json()


class _TruLensAgentTraceScorerBase(Scorer):
    """Base class for TruLens agent trace scorer wrappers."""

    name: str
    model_name: str | None = None
    model_provider: str = "openai"
    criteria: str | None = None
    custom_instructions: str | None = None
    temperature: float = 0.0
    enable_trace_compression: bool = True

    def _get_trulens_provider(self):
        """Get the appropriate TruLens provider instance."""
        _check_trulens_installed()

        match self.model_provider:
            case "openai":
                from trulens.providers.openai import OpenAI

                return OpenAI(model_engine=self.model_name or "gpt-4o-mini")
            case "litellm":
                try:
                    from trulens.providers.litellm import LiteLLM

                    return LiteLLM(model_engine=self.model_name or "gpt-4o-mini")
                except ImportError:
                    raise MlflowException(
                        "LiteLLM provider requires 'trulens-providers-litellm'. "
                        "Install it with: pip install trulens-providers-litellm",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            case _:
                raise MlflowException(
                    f"Unsupported model provider: {self.model_provider}. "
                    "Currently supported: 'openai', 'litellm'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    def _format_rationale(self, reasons: dict[str, Any] | None) -> str | None:
        """Format TruLens reasons dict into a readable rationale string."""
        if not reasons:
            return None

        parts = []
        for key, value in reasons.items():
            if isinstance(value, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {value}")
            else:
                parts.append(f"{key}: {value}")

        return " | ".join(parts) if parts else None

    def _get_trace_string(self, trace: Trace | str | None) -> str:
        """Convert trace to string format expected by TruLens."""
        if trace is None:
            raise MlflowException(
                "Trace is required for agent trace evaluation. "
                "Ensure you are passing trace data to the scorer.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if isinstance(trace, Trace):
            return _convert_mlflow_trace_to_string(trace)
        elif isinstance(trace, str):
            return trace
        else:
            raise MlflowException(
                f"Invalid trace type: {type(trace)}. Expected Trace or str.",
                error_code=INVALID_PARAMETER_VALUE,
            )


@experimental(version="3.9.0")
class TruLensLogicalConsistencyScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens logical consistency scorer for agent traces.

    Evaluates the quality of an agentic trace using a rubric focused on
    logical consistency and reasoning. This scorer analyzes how coherent
    and logically sound the agent's decision-making process is throughout
    the execution trace.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_logical_consistency".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensLogicalConsistencyScorer()],
        )
    """

    name: str = "trulens_logical_consistency"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate logical consistency of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with logical consistency score (0-1) and rationale.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.logical_consistency_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )


@experimental(version="3.9.0")
class TruLensExecutionEfficiencyScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens execution efficiency scorer for agent traces.

    Evaluates the quality of an agentic execution using a rubric focused on
    execution efficiency. This scorer analyzes whether the agent takes an
    optimal or near-optimal path to achieve its goal without unnecessary
    steps or redundant operations.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_execution_efficiency".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensExecutionEfficiencyScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensExecutionEfficiencyScorer()],
        )
    """

    name: str = "trulens_execution_efficiency"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate execution efficiency of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with execution efficiency score (0-1) and rationale.
            Score of 1.0 indicates highly streamlined/optimized workflow,
            0.0 indicates highly inefficient workflow.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.execution_efficiency_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )


@experimental(version="3.9.0")
class TruLensPlanAdherenceScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens plan adherence scorer for agent traces.

    Evaluates the quality of an agentic trace using a rubric focused on
    execution adherence to the plan. This scorer analyzes whether the agent
    follows its stated plan during execution or deviates from it.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_plan_adherence".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensPlanAdherenceScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensPlanAdherenceScorer()],
        )
    """

    name: str = "trulens_plan_adherence"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate plan adherence of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with plan adherence score (0-1) and rationale.
            Score of 1.0 indicates execution followed plan exactly,
            0.0 indicates execution did not follow plan.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.plan_adherence_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )


@experimental(version="3.9.0")
class TruLensPlanQualityScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens plan quality scorer for agent traces.

    Evaluates the quality of an agentic system's plan. This scorer analyzes
    whether the agent's plan is well-structured, comprehensive, and appropriate
    for achieving the stated goal.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_plan_quality".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensPlanQualityScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensPlanQualityScorer()],
        )
    """

    name: str = "trulens_plan_quality"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate plan quality of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with plan quality score (0-1) and rationale.
            Score of 1.0 indicates excellent plan quality,
            0.0 indicates poor plan quality.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.plan_quality_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )


@experimental(version="3.9.0")
class TruLensToolSelectionScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens tool selection scorer for agent traces.

    Evaluates the quality of an agentic trace using a rubric focused on
    tool selection. This scorer analyzes whether the agent chooses the
    appropriate tools for each step of the task.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_tool_selection".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensToolSelectionScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensToolSelectionScorer()],
        )
    """

    name: str = "trulens_tool_selection"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate tool selection of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with tool selection score (0-1) and rationale.
            Score of 1.0 indicates excellent tool selection,
            0.0 indicates poor tool selection.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.tool_selection_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )


@experimental(version="3.9.0")
class TruLensToolCallingScorer(_TruLensAgentTraceScorerBase):
    """
    TruLens tool calling scorer for agent traces.

    Evaluates the quality of an agentic trace using a rubric focused on
    tool calling. This scorer analyzes whether the agent correctly invokes
    tools with appropriate parameters and handles tool responses properly.

    This is part of TruLens' goal-plan-action alignment evaluations which
    achieve 95% error coverage against TRAIL benchmarks.

    Args:
        name: The name of the scorer. Defaults to "trulens_tool_calling".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".
        criteria: Optional custom criteria for evaluation.
        custom_instructions: Optional custom instructions for evaluation.
        temperature: LLM temperature for evaluation. Defaults to 0.0 for determinism.
        enable_trace_compression: Whether to compress trace data. Defaults to True.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import TruLensToolCallingScorer

        # Get traces from your agent
        traces = mlflow.search_traces(experiment_ids=["1"])

        # Evaluate traces
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[TruLensToolCallingScorer()],
        )
    """

    name: str = "trulens_tool_calling"

    def __call__(
        self,
        *,
        trace: Trace | str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate tool calling of an agent trace.

        Args:
            trace: The agent trace to evaluate (MLflow Trace object or JSON string).

        Returns:
            Feedback with tool calling score (0-1) and rationale.
            Score of 1.0 indicates excellent tool calling,
            0.0 indicates poor tool calling.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()
        trace_str = self._get_trace_string(trace)

        score, reasons = provider.tool_calling_with_cot_reasons(
            trace=trace_str,
            criteria=self.criteria,
            custom_instructions=self.custom_instructions,
            temperature=self.temperature,
        )

        return Feedback(
            name=self.name,
            value=score,
            rationale=self._format_rationale(reasons),
        )
