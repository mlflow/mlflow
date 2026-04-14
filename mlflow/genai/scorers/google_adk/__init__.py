"""
Google ADK integration for MLflow.

This module provides integration with Google Agent Development Kit (ADK) evaluators,
allowing them to be used with MLflow's scorer interface for agent evaluation.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.google_adk import ToolTrajectory, ResponseMatch

    scorer = ToolTrajectory(threshold=0.5)
    feedback = scorer(
        outputs="I found 3 flights to Paris.",
        expectations={
            "expected_tool_calls": [
                {"name": "search_flights", "args": {"destination": "Paris"}},
            ]
        },
    )

    rouge_scorer = ResponseMatch(threshold=0.5)
    feedback = rouge_scorer(
        outputs="MLflow is a platform for ML.",
        expectations={"expected_response": "MLflow is an ML lifecycle platform."},
    )

Model routing notes for LLM judge scorers (``ResponseEvaluation``, ``Safety``,
``Hallucination``):

    These scorers delegate model invocation to ADK's internal LLM registry
    (``google.adk.models.LlmRegistry``), which does not go through MLflow's
    gateway adapters (``call_chat_completions`` / LiteLLM). Only models that
    ADK natively supports (Gemini, etc.) can be used as the judge. Passing
    MLflow model URIs like ``databricks`` or ``openai:/gpt-4o`` will fail at
    evaluation time because ADK cannot resolve them. This is a known limitation
    and will be revisited once ADK exposes a pluggable judge-model interface.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.google_adk.utils import (
    check_adk_installed,
    map_scorer_inputs_to_invocation,
    run_async,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_FRAMEWORK_NAME = "google_adk"

_DEFAULT_THRESHOLD = 0.5

# ADK's LLM judge criteria require a Google-native model name (the judge runs
# through ADK's own LlmRegistry, not through MLflow's gateway adapters).
# Gemini is the only model ADK resolves out of the box, so it is the default.
_DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


@experimental(version="3.11.0")
class GoogleADKScorer(Scorer):
    """
    Base class for Google ADK evaluator scorers.

    Google ADK provides deterministic evaluators for agent assessment
    including tool trajectory matching and response similarity scoring.

    Args:
        metric_name: Name of the ADK metric
        threshold: Score threshold for pass/fail determination (default: 0.5)
    """

    _evaluator: Any = PrivateAttr()
    _threshold: float = PrivateAttr()
    _model: str | None = PrivateAttr(default=None)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def _raise_registration_not_supported(self, method_name: str):
        raise MlflowException.invalid_parameter_value(
            f"'{method_name}()' is not supported for third-party scorers like Google ADK. "
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
            "'align()' is not supported for third-party scorers like Google ADK. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def _evaluate_judge(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        is_async: bool = False,
        require_reference: bool = False,
    ) -> Feedback:
        is_llm = self._model is not None
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE if is_llm else AssessmentSourceType.CODE,
            source_id=self._model if is_llm else f"google_adk/{self.name}",
        )

        try:
            if require_reference:
                reference = None
                if expectations:
                    reference = (
                        expectations.get("expected_response")
                        or expectations.get("context")
                        or expectations.get("reference")
                        or expectations.get("expected_output")
                    )
                if reference is None:
                    raise MlflowException.invalid_parameter_value(
                        f"{self.name} scorer requires 'expected_response', 'context', or "
                        "'reference' in expectations."
                    )

            actual_inv, expected_inv = map_scorer_inputs_to_invocation(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

            call = self._evaluator.evaluate_invocations(
                actual_invocations=[actual_inv],
                expected_invocations=[expected_inv],
            )
            result = run_async(call) if is_async else call

            score = result.overall_score if result.overall_score is not None else 0.0
            passed = score >= self._threshold

            return Feedback(
                name=self.name,
                value=CategoricalRating.YES if passed else CategoricalRating.NO,
                rationale=f"{self.name} score: {score:.2f} (threshold: {self._threshold})",
                source=assessment_source,
                metadata={
                    "score": score,
                    "threshold": self._threshold,
                    FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME,
                },
            )
        except Exception as e:
            _logger.error("Error evaluating Google ADK metric %s: %s", self.name, e)
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME},
            )


@experimental(version="3.11.0")
class ToolTrajectory(GoogleADKScorer):
    """
    Evaluates agent tool call trajectories against expected tool calls.

    Wraps ADK's ``TrajectoryEvaluator`` to compare the sequence of tools
    called by an agent against a reference list of expected calls.  Three
    matching strategies are supported: ``EXACT``, ``IN_ORDER``, and
    ``ANY_ORDER``.

    Args:
        match_type: How to compare tool calls. One of "EXACT", "IN_ORDER",
            or "ANY_ORDER" (default: "EXACT").
        threshold: Score threshold for pass/fail (default: 0.5).
            Each invocation scores 1.0 (match) or 0.0 (no match).

    Examples:
        .. code-block:: python

            scorer = ToolTrajectory(match_type="EXACT", threshold=0.5)
            feedback = scorer(
                inputs="Book a flight to Paris",
                outputs="Booked flight AA123 to Paris",
                expectations={
                    "expected_tool_calls": [
                        {"name": "search_flights", "args": {"destination": "Paris"}},
                        {"name": "book_flight", "args": {"flight_id": "AA123"}},
                    ],
                    "actual_tool_calls": [
                        {"name": "search_flights", "args": {"destination": "Paris"}},
                        {"name": "book_flight", "args": {"flight_id": "AA123"}},
                    ],
                },
            )
    """

    metric_name: ClassVar[str] = "ToolTrajectory"

    def __init__(
        self,
        match_type: Literal["EXACT", "IN_ORDER", "ANY_ORDER"] = "EXACT",
        threshold: float = _DEFAULT_THRESHOLD,
        **kwargs: Any,
    ):
        check_adk_installed()
        super().__init__(name=self.metric_name)
        self._threshold = threshold
        self._evaluator = _create_trajectory_evaluator(threshold, match_type)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        if not expectations or "expected_tool_calls" not in expectations:
            return Feedback(
                name=self.name,
                error=MlflowException.invalid_parameter_value(
                    "ToolTrajectory scorer requires 'expected_tool_calls' in expectations."
                ),
                source=AssessmentSource(
                    source_type=AssessmentSourceType.CODE,
                    source_id=f"google_adk/{self.metric_name}",
                ),
                metadata={FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME},
            )

        return self._evaluate_judge(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )


@experimental(version="3.11.0")
class ResponseMatch(GoogleADKScorer):
    """
    Evaluates response similarity using ROUGE-1 scoring.

    Wraps ADK's ``RougeEvaluator`` to compute ROUGE-1 F-measure between
    the agent's output and a reference (expected) response.  Scores range
    from 0.0 to 1.0, with values closer to 1.0 indicating higher overlap.

    Args:
        threshold: Score threshold for pass/fail (default: 0.5).

    Examples:
        .. code-block:: python

            scorer = ResponseMatch(threshold=0.6)
            feedback = scorer(
                outputs="MLflow is an open-source ML platform.",
                expectations={
                    "expected_response": "MLflow is an open-source platform for ML lifecycle."
                },
            )
    """

    metric_name: ClassVar[str] = "ResponseMatch"

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        **kwargs: Any,
    ):
        check_adk_installed()
        super().__init__(name=self.metric_name)
        self._threshold = threshold
        self._evaluator = _create_rouge_evaluator(threshold)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        return self._evaluate_judge(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
            require_reference=True,
        )


@experimental(version="3.12.0")
class ResponseEvaluation(GoogleADKScorer):
    """
    Evaluates response quality using an LLM judge.

    Wraps ADK's ``FinalResponseMatchV2Evaluator`` which uses a judge model
    to assess whether the agent's response matches the expected response.
    Requires expected_response in expectations.

    .. note::
        The judge model must be resolvable by ADK's ``LlmRegistry`` (e.g.
        ``gemini-2.5-flash``). MLflow model URIs such as ``databricks`` or
        ``openai:/gpt-4o`` are NOT supported for the LLM judge scorers in
        this integration.

    Args:
        model: Judge model ID (default: ``"gemini-2.5-flash"``). Must be a
            model ADK's ``LlmRegistry`` can resolve.
        threshold: Score threshold for pass/fail (default: 0.5).
        num_samples: Number of judge samples for majority voting (default: 5).
    """

    metric_name: ClassVar[str] = "ResponseEvaluation"

    def __init__(
        self,
        model: str = _DEFAULT_JUDGE_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        num_samples: int = 5,
        **kwargs: Any,
    ):
        check_adk_installed()
        super().__init__(name=self.metric_name)
        self._threshold = threshold
        self._model = model
        self._evaluator = _create_response_evaluation_evaluator(threshold, model, num_samples)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        return self._evaluate_judge(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
            is_async=True,
            require_reference=True,
        )


@experimental(version="3.12.0")
class Safety(GoogleADKScorer):
    """
    Evaluates response safety using an LLM judge.

    Wraps ADK's ``SafetyEvaluatorV1`` which uses a judge model to assess
    whether the agent's response is safe and free of harmful content.

    .. note::
        The judge model must be resolvable by ADK's ``LlmRegistry`` (e.g.
        ``gemini-2.5-flash``). MLflow model URIs such as ``databricks`` or
        ``openai:/gpt-4o`` are NOT supported.

    Args:
        model: Judge model ID (default: ``"gemini-2.5-flash"``).
        threshold: Score threshold for pass/fail (default: 0.5).
        num_samples: Number of judge samples for majority voting (default: 5).
    """

    metric_name: ClassVar[str] = "Safety"

    def __init__(
        self,
        model: str = _DEFAULT_JUDGE_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        num_samples: int = 5,
        **kwargs: Any,
    ):
        check_adk_installed()
        super().__init__(name=self.metric_name)
        self._threshold = threshold
        self._model = model
        self._evaluator = _create_safety_evaluator(threshold, model, num_samples)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        return self._evaluate_judge(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )


@experimental(version="3.12.0")
class Hallucination(GoogleADKScorer):
    """
    Evaluates response factual grounding using an LLM judge.

    Wraps ADK's ``HallucinationsV1Evaluator`` which uses a judge model to
    assess whether the agent's response contains hallucinated content.

    .. note::
        The judge model must be resolvable by ADK's ``LlmRegistry`` (e.g.
        ``gemini-2.5-flash``). MLflow model URIs such as ``databricks`` or
        ``openai:/gpt-4o`` are NOT supported.

    Args:
        model: Judge model ID (default: ``"gemini-2.5-flash"``).
        threshold: Score threshold for pass/fail (default: 0.5).
        num_samples: Number of judge samples for majority voting (default: 5).
    """

    metric_name: ClassVar[str] = "Hallucination"

    def __init__(
        self,
        model: str = _DEFAULT_JUDGE_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        num_samples: int = 5,
        **kwargs: Any,
    ):
        check_adk_installed()
        super().__init__(name=self.metric_name)
        self._threshold = threshold
        self._model = model
        self._evaluator = _create_hallucination_evaluator(threshold, model, num_samples)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        return self._evaluate_judge(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
            is_async=True,
        )


def _create_llm_judge_criterion(threshold: float, model: str, num_samples: int):
    from google.adk.evaluation.eval_metrics import JudgeModelOptions, LlmAsAJudgeCriterion

    judge_opts = JudgeModelOptions(judge_model=model, num_samples=num_samples)
    return LlmAsAJudgeCriterion(threshold=threshold, judge_model_options=judge_opts)


def _create_response_evaluation_evaluator(threshold: float, model: str, num_samples: int):
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.final_response_match_v2 import FinalResponseMatchV2Evaluator

    criterion = _create_llm_judge_criterion(threshold, model, num_samples)
    eval_metric = EvalMetric(
        metric_name="final_response_match_v2",
        threshold=threshold,
        criterion=criterion,
    )
    return FinalResponseMatchV2Evaluator(eval_metric=eval_metric)


def _create_safety_evaluator(threshold: float, model: str, num_samples: int):
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.safety_evaluator import SafetyEvaluatorV1

    criterion = _create_llm_judge_criterion(threshold, model, num_samples)
    eval_metric = EvalMetric(
        metric_name="safety_evaluator_v1",
        threshold=threshold,
        criterion=criterion,
    )
    return SafetyEvaluatorV1(eval_metric=eval_metric)


def _create_hallucination_evaluator(threshold: float, model: str, num_samples: int):
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.hallucinations_v1 import HallucinationsV1Evaluator

    criterion = _create_llm_judge_criterion(threshold, model, num_samples)
    eval_metric = EvalMetric(
        metric_name="hallucinations_v1",
        threshold=threshold,
        criterion=criterion,
    )
    return HallucinationsV1Evaluator(eval_metric=eval_metric)


def _create_trajectory_evaluator(threshold: float, match_type: str):
    from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
    from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator

    match match_type.upper():
        case "EXACT":
            mt = ToolTrajectoryCriterion.MatchType.EXACT
        case "IN_ORDER":
            mt = ToolTrajectoryCriterion.MatchType.IN_ORDER
        case "ANY_ORDER":
            mt = ToolTrajectoryCriterion.MatchType.ANY_ORDER
        case _:
            raise MlflowException.invalid_parameter_value(
                f"Invalid match_type '{match_type}'. "
                f"Must be one of: 'EXACT', 'IN_ORDER', 'ANY_ORDER'."
            )

    criterion = ToolTrajectoryCriterion(match_type=mt)
    eval_metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        threshold=threshold,
        criterion=criterion,
    )
    return TrajectoryEvaluator(eval_metric=eval_metric)


def _create_rouge_evaluator(threshold: float):
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.final_response_match_v1 import RougeEvaluator

    eval_metric = EvalMetric(
        metric_name="response_match_score",
        threshold=threshold,
    )
    return RougeEvaluator(eval_metric=eval_metric)


@experimental(version="3.11.0")
def get_scorer(
    metric_name: str,
    **kwargs: Any,
) -> GoogleADKScorer:
    """
    Get a Google ADK evaluator as an MLflow scorer.

    Args:
        metric_name: Name of the ADK metric. Available: "ToolTrajectory",
            "ResponseMatch", "ResponseEvaluation", "Safety", "Hallucination".
        kwargs: Additional keyword arguments passed to the scorer constructor
            (e.g., threshold, match_type, model, num_samples).

    Returns:
        GoogleADKScorer instance that can be called with MLflow's scorer interface

    Examples:
        .. code-block:: python

            scorer = get_scorer("ToolTrajectory", match_type="IN_ORDER", threshold=0.5)
            scorer = get_scorer("ResponseMatch", threshold=0.7)
    """
    from mlflow.genai.scorers.google_adk.registry import get_scorer_class

    return get_scorer_class(metric_name)(**kwargs)


__all__ = [
    "GoogleADKScorer",
    # Deterministic scorers
    "ResponseMatch",
    "ToolTrajectory",
    # LLM judge scorers
    "Hallucination",
    "ResponseEvaluation",
    "Safety",
    "get_scorer",
]
