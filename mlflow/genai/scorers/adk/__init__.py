"""
ADK (Agent Development Kit) integration for MLflow.

This module provides integration with Google ADK evaluation metrics, allowing
them to be used with MLflow's scorer interface.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.adk import ToolTrajectory, Hallucinations

    result = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[
            ToolTrajectory(match_type="in_order"),
            Hallucinations(model="gemini-2.5-flash"),
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.adk._mapping import map_adk_result_to_feedback, map_to_adk_invocations
from mlflow.genai.scorers.adk._registry import (
    ADK_NOT_INSTALLED_ERROR_MESSAGE,
    get_adk_metric_name,
    get_evaluator_class,
    is_deterministic_metric,
)
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.8.0")
class AdkScorer(Scorer):
    """Base scorer class for ADK evaluation metrics.

    Wraps Google ADK evaluators for use with MLflow's scorer interface.

    Args:
        metric_name: Name of the ADK metric (e.g., "ToolTrajectory").
            If not provided, uses the class-level metric_name attribute.
        model: Model name for LLM-based evaluators (e.g., "gemini-2.5-flash").
        **kwargs: Additional metric-specific parameters passed to the ADK
            evaluator's criterion.
    """

    _evaluator: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        if metric_name is None:
            metric_name = getattr(self, "metric_name", None)
            if metric_name is None:
                raise MlflowException.invalid_parameter_value(
                    "metric_name must be provided either as an argument or as a "
                    "class attribute."
                )

        super().__init__(name=metric_name)

        self._is_deterministic = is_deterministic_metric(metric_name)
        adk_metric_name = get_adk_metric_name(metric_name)

        if self._is_deterministic:
            self._model_uri = None
        else:
            self._model_uri = model

        self._evaluator = self._create_evaluator(
            adk_metric_name, model, **kwargs
        )

    @staticmethod
    def _create_evaluator(adk_metric_name: str, model: str | None, **kwargs):
        """Create an ADK evaluator instance with the given configuration."""
        try:
            from google.adk.evaluation.eval_metrics import EvalMetric
        except ImportError as e:
            raise MlflowException.invalid_parameter_value(
                ADK_NOT_INSTALLED_ERROR_MESSAGE
            ) from e

        evaluator_class = get_evaluator_class(adk_metric_name)
        criterion = _build_criterion(adk_metric_name, model, **kwargs)

        eval_metric = EvalMetric(
            metric_name=adk_metric_name,
            criterion=criterion,
        )

        return evaluator_class(eval_metric=eval_metric)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    @property
    def is_session_level_scorer(self) -> bool:
        return False

    def _raise_registration_not_supported(self, method_name: str):
        raise MlflowException.invalid_parameter_value(
            f"'{method_name}()' is not supported for third-party scorers like ADK. "
            f"Third-party scorers cannot be registered, started, updated, or stopped. "
            f"Use them directly in mlflow.genai.evaluate() instead."
        )

    def register(self, **kwargs):
        self._raise_registration_not_supported("register")

    def start(self, **kwargs):
        self._raise_registration_not_supported("start")

    def update(self, **kwargs):
        self._raise_registration_not_supported("update")

    def stop(self, **kwargs):
        self._raise_registration_not_supported("stop")

    def align(self, **kwargs):
        raise MlflowException.invalid_parameter_value(
            "'align()' is not supported for third-party scorers like ADK. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """Evaluate using the wrapped ADK evaluator.

        Args:
            inputs: The input to evaluate.
            outputs: The output to evaluate.
            expectations: Expected values for evaluation (expected_response,
                expected_tool_calls, rubrics).
            trace: MLflow trace for extracting tool call information.
            session: Not used for ADK scorers (per-invocation evaluation).

        Returns:
            Feedback object with score, rationale, and ADK metadata.
        """
        if self._is_deterministic:
            source_type = AssessmentSourceType.CODE
            source_id = None
        else:
            source_type = AssessmentSourceType.LLM_JUDGE
            source_id = self._model_uri

        assessment_source = AssessmentSource(
            source_type=source_type,
            source_id=source_id,
        )

        try:
            actual_invocation, expected_invocation = map_to_adk_invocations(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

            actual_list = [actual_invocation]
            expected_list = [expected_invocation] if expected_invocation else None

            eval_result = self._evaluator.evaluate_invocations(
                actual_invocations=actual_list,
                expected_invocations=expected_list,
            )

            return map_adk_result_to_feedback(
                name=self.name,
                eval_result=eval_result,
                source=assessment_source,
                is_deterministic=self._is_deterministic,
            )
        except Exception as e:
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
            )


@experimental(version="3.8.0")
def get_scorer(
    metric_name: str,
    model: str | None = None,
    **kwargs: Any,
) -> AdkScorer:
    """Get an ADK evaluation metric as an MLflow scorer.

    Args:
        metric_name: Name of the ADK metric. Available metrics:
            - "ToolTrajectory": Tool call trajectory matching
            - "FinalResponseMatch": Final response matching (LLM-based)
            - "ResponseMatch": Response similarity (ROUGE-based)
            - "RubricBasedResponseQuality": Rubric-based response quality
            - "RubricBasedToolUseQuality": Rubric-based tool use quality
            - "Hallucinations": Hallucination detection
        model: Model name for LLM-based metrics (e.g., "gemini-2.5-flash").
        **kwargs: Additional metric-specific parameters.

    Returns:
        AdkScorer instance that can be used with mlflow.genai.evaluate().

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import get_scorer

        scorer = get_scorer("ToolTrajectory", match_type="in_order")
        feedback = scorer(
            inputs="Book a flight",
            outputs="Flight booked",
            expectations={"expected_tool_calls": ["search_flights", "book_flight"]},
        )
    """
    return AdkScorer(
        metric_name=metric_name,
        model=model,
        **kwargs,
    )


def _build_criterion(adk_metric_name: str, model: str | None, **kwargs):
    """Build an ADK criterion for the given metric.

    Creates the appropriate criterion subclass based on the metric type.
    """
    from google.adk.evaluation.eval_metrics import BaseCriterion

    threshold = kwargs.pop("threshold", 0.5)

    if adk_metric_name == "tool_trajectory_avg_score":
        from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion

        match_type = kwargs.pop("match_type", "exact")
        return ToolTrajectoryCriterion(
            threshold=threshold,
            match_type=match_type,
            **kwargs,
        )

    if adk_metric_name in (
        "final_response_match_v2",
        "rubric_based_final_response_quality_v1",
        "rubric_based_tool_use_quality_v1",
    ):
        from google.adk.evaluation.eval_metrics import (
            JudgeModelOptions,
            RubricsBasedCriterion,
        )

        judge_opts = JudgeModelOptions()
        if model:
            judge_opts = JudgeModelOptions(judge_model=model)

        rubrics = kwargs.pop("rubrics", [])

        if adk_metric_name == "final_response_match_v2":
            from google.adk.evaluation.eval_metrics import LlmAsAJudgeCriterion

            return LlmAsAJudgeCriterion(
                threshold=threshold,
                judge_model_options=judge_opts,
                **kwargs,
            )

        return RubricsBasedCriterion(
            threshold=threshold,
            judge_model_options=judge_opts,
            rubrics=rubrics,
            **kwargs,
        )

    if adk_metric_name == "hallucinations_v1":
        from google.adk.evaluation.eval_metrics import (
            HallucinationsCriterion,
            JudgeModelOptions,
        )

        judge_opts = JudgeModelOptions()
        if model:
            judge_opts = JudgeModelOptions(judge_model=model)

        evaluate_intermediate = kwargs.pop(
            "evaluate_intermediate_nl_responses", False
        )
        return HallucinationsCriterion(
            threshold=threshold,
            judge_model_options=judge_opts,
            evaluate_intermediate_nl_responses=evaluate_intermediate,
            **kwargs,
        )

    if adk_metric_name == "response_match_score":
        return BaseCriterion(threshold=threshold, **kwargs)

    # Fallback: basic criterion
    return BaseCriterion(threshold=threshold, **kwargs)


# Named scorer subclasses for convenience


@experimental(version="3.8.0")
class ToolTrajectory(AdkScorer):
    """Evaluates tool call trajectory accuracy.

    Compares the sequence of tools called by the agent against expected
    tool calls using one of three match types: EXACT, IN_ORDER, or ANY_ORDER.

    Args:
        match_type: Match strategy - "exact", "in_order", or "any_order".
            Defaults to "exact".
        threshold: Minimum score for passing (default: 0.5).
        model: Not used for this deterministic metric.

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import ToolTrajectory

        scorer = ToolTrajectory(match_type="in_order")
        feedback = scorer(
            inputs="Book a flight to NYC",
            outputs="Flight booked",
            expectations={
                "expected_tool_calls": ["search_flights", "book_flight"],
            },
        )
    """

    metric_name: ClassVar[str] = "ToolTrajectory"


@experimental(version="3.8.0")
class FinalResponseMatch(AdkScorer):
    """Evaluates if the agent's final response matches the expected response.

    Uses an LLM judge to determine semantic equivalence between the agent's
    response and the expected golden response.

    Args:
        model: LLM model for judging (e.g., "gemini-2.5-flash").
        threshold: Minimum score for passing (default: 0.5).

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import FinalResponseMatch

        scorer = FinalResponseMatch(model="gemini-2.5-flash")
        feedback = scorer(
            inputs="What is Python?",
            outputs="Python is a programming language.",
            expectations={"expected_response": "Python is a programming language."},
        )
    """

    metric_name: ClassVar[str] = "FinalResponseMatch"


@experimental(version="3.8.0")
class ResponseMatch(AdkScorer):
    """Evaluates response similarity using ROUGE-based matching.

    A deterministic metric that computes text similarity between the
    agent's response and the expected response without requiring an LLM.

    Args:
        threshold: Minimum score for passing (default: 0.5).

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import ResponseMatch

        scorer = ResponseMatch(threshold=0.8)
        feedback = scorer(
            inputs="What is Python?",
            outputs="Python is a programming language.",
            expectations={"expected_response": "Python is a programming language."},
        )
    """

    metric_name: ClassVar[str] = "ResponseMatch"


@experimental(version="3.8.0")
class RubricBasedResponseQuality(AdkScorer):
    """Evaluates response quality using custom rubrics.

    Uses an LLM judge to assess the agent's response against a set of
    user-defined rubrics.

    Args:
        model: LLM model for judging (e.g., "gemini-2.5-flash").
        rubrics: List of rubric strings or dicts for evaluation.
        threshold: Minimum score for passing (default: 0.5).

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import RubricBasedResponseQuality

        scorer = RubricBasedResponseQuality(
            model="gemini-2.5-flash",
            rubrics=["Response is helpful and accurate"],
        )
    """

    metric_name: ClassVar[str] = "RubricBasedResponseQuality"


@experimental(version="3.8.0")
class RubricBasedToolUseQuality(AdkScorer):
    """Evaluates tool use quality using custom rubrics.

    Uses an LLM judge to assess the agent's tool usage against a set of
    user-defined rubrics.

    Args:
        model: LLM model for judging (e.g., "gemini-2.5-flash").
        rubrics: List of rubric strings or dicts for evaluation.
        threshold: Minimum score for passing (default: 0.5).

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import RubricBasedToolUseQuality

        scorer = RubricBasedToolUseQuality(
            model="gemini-2.5-flash",
            rubrics=["Agent uses the correct tools for the task"],
        )
    """

    metric_name: ClassVar[str] = "RubricBasedToolUseQuality"


@experimental(version="3.8.0")
class Hallucinations(AdkScorer):
    """Detects hallucinations in agent responses.

    Uses an LLM judge to identify fabricated information in the agent's
    response. Supports evaluating both final and intermediate responses.

    Args:
        model: LLM model for judging (e.g., "gemini-2.5-flash").
        evaluate_intermediate_nl_responses: Whether to also evaluate
            intermediate responses (default: False).
        threshold: Minimum score for passing (default: 0.5).

    Examples:

    .. code-block:: python

        from mlflow.genai.scorers.adk import Hallucinations

        scorer = Hallucinations(model="gemini-2.5-flash")
        feedback = scorer(
            inputs="What is the capital of France?",
            outputs="The capital of France is Paris.",
        )
    """

    metric_name: ClassVar[str] = "Hallucinations"


__all__ = [
    "AdkScorer",
    "get_scorer",
    "ToolTrajectory",
    "FinalResponseMatch",
    "ResponseMatch",
    "RubricBasedResponseQuality",
    "RubricBasedToolUseQuality",
    "Hallucinations",
]
