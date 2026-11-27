"""
Inspect AI integration for MLflow.

This module wraps `Inspect AI <https://inspect.aisi.org.uk/>`_ scorers so they run through
MLflow's scorer interface and inside ``mlflow.genai.evaluate()``.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.inspect_ai import ExactMatch, ModelGradedFact, get_scorer

    scorer = ModelGradedFact(model="openai:/gpt-4o-mini")
    feedback = scorer(
        inputs="What is the capital of France?",
        outputs="Paris is the capital of France.",
        expectations={"expected_response": "Paris"},
    )

    exact = ExactMatch()
    feedback = exact(outputs="Paris", expectations={"expected_response": "Paris"})

    # Any other scorer factory shipped by Inspect can be used by name.
    choice = get_scorer("match", location="any")
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import AssessmentError, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import DEFAULT_SCORER_TIMEOUT, Scorer, ScorerKind
from mlflow.genai.scorers.inspect_ai.models import create_inspect_model
from mlflow.genai.scorers.inspect_ai.registry import get_scorer_factory, is_model_graded
from mlflow.genai.scorers.inspect_ai.utils import (
    TARGET_EXPECTATION_KEYS,
    TERMINAL_STATE_KEY,
    build_task_state,
    check_inspect_ai_installed,
    run_scorer,
    score_to_feedback_fields,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)

_FRAMEWORK_NAME = "inspect_ai"
_DEFAULT_THRESHOLD = 0.5
_NOT_RUN_ERROR_CODE = "INSPECT_AI_SCORER_NOT_RUN"


@experimental(version="3.19.0")
@format_docstring(_MODEL_API_DOC)
class InspectAIScorer(Scorer):
    """
    MLflow scorer that runs an Inspect AI scorer.

    The wrapper builds an Inspect ``TaskState`` from the MLflow inputs, outputs and trace,
    reads the ``Target`` from the expectations, awaits the Inspect scorer with a timeout and
    converts the resulting ``Score`` into a ``Feedback``. The target comes from the first of
    these expectation keys that is set: ``expected_response``, ``expected_output``,
    ``reference``, ``expected_facts`` or ``target``. Lists become multi-value targets.

    Args:
        metric_name: Name of an ``inspect_ai.scorer`` factory such as ``"exact"``,
            ``"includes"``, ``"match"``, ``"pattern"``, ``"f1"``, ``"answer"``,
            ``"model_graded_fact"`` or ``"model_graded_qa"``. Any other factory shipped by
            Inspect works too. The CamelCase names of the concrete classes in this module
            are accepted as aliases. Required unless the concrete class pins the name.
        scorer: An already-built Inspect scorer to wrap instead of a factory name. Then
            ``metric_name`` is only the Feedback name and ``model`` is ignored. Custom
            scorers are recorded with a code source and cannot be serialized.
        model: {{ model }}
            Only used by model-graded scorers. The grader calls go through MLflow's judge
            backends, so Databricks, the native providers and the LiteLLM fallback all work.
        model_kwargs: Parameters for the grader LLM, for example ``temperature``. Ignored
            by deterministic scorers.
        threshold: Minimum numeric score for a ``yes`` verdict (default 0.5). Inspect grades
            map to numbers the way Inspect maps them: ``C`` is 1.0, ``P`` is 0.5, ``I`` and
            ``N`` are 0.0.
        timeout: Per-call timeout in seconds for the Inspect scorer coroutine. ``None`` uses
            MLflow's default scorer timeout and ``0`` disables it.
        metric_kwargs: Keyword arguments forwarded to the Inspect scorer factory, for
            example ``ignore_case`` for ``match`` or ``partial_credit`` for
            ``model_graded_fact``.
    """

    _scorer: Any = PrivateAttr()
    _metric_name: str = PrivateAttr(default="")
    _metric_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _model: str | None = PrivateAttr(default=None)
    _threshold: float = PrivateAttr(default=_DEFAULT_THRESHOLD)
    _is_model_graded: bool = PrivateAttr(default=False)
    _is_custom: bool = PrivateAttr(default=False)

    def __init__(
        self,
        metric_name: str | None = None,
        scorer: Any = None,
        model: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        timeout: int | float | None = None,
        **metric_kwargs: Any,
    ):
        check_inspect_ai_installed()

        if metric_name is None:
            metric_name = getattr(type(self), "metric_name", None)
        if metric_name is None:
            raise MlflowException.invalid_parameter_value(
                "InspectAIScorer requires 'metric_name'. Pass the name of an Inspect scorer "
                "factory, or a name for the Feedback together with a custom 'scorer'."
            )

        super().__init__(name=metric_name, timeout=timeout)

        self._metric_name = metric_name
        self._threshold = threshold
        self._is_custom = scorer is not None

        if scorer is not None:
            if model is not None:
                _logger.warning(
                    "'model' is ignored when wrapping a custom Inspect scorer. Build the "
                    "scorer with the grader model you want instead."
                )
            self._scorer = scorer
            self._model = None
            self._is_model_graded = False
            self._metric_kwargs = {}
            return

        factory = get_scorer_factory(metric_name)
        self._is_model_graded = is_model_graded(metric_name)
        factory_kwargs = dict(metric_kwargs)
        if self._is_model_graded:
            self._model = model or get_default_model()
            factory_kwargs["model"] = create_inspect_model(self._model, model_kwargs=model_kwargs)
        else:
            self._model = None
        self._scorer = factory(**factory_kwargs)

        # Everything needed to rebuild this scorer through the generic third-party
        # deserialization path, which calls `cls(metric_name=..., model=..., **kwargs)`.
        self._metric_kwargs = {**metric_kwargs, "threshold": threshold}
        if model_kwargs:
            self._metric_kwargs["model_kwargs"] = dict(model_kwargs)
        if timeout is not None:
            self._metric_kwargs["timeout"] = timeout

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def align(self, **kwargs):
        raise MlflowException.invalid_parameter_value(
            "'align()' is not supported for third-party scorers like Inspect AI. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        if self._is_custom:
            raise MlflowException.invalid_parameter_value(
                f"Inspect AI scorer '{self.name}' wraps a custom scorer object and cannot be "
                "serialized. Use a scorer factory name (for example 'model_graded_fact') "
                "when the scorer needs to be registered or stored."
            )
        return super().model_dump(**kwargs)

    def _effective_timeout(self) -> float | None:
        timeout = DEFAULT_SCORER_TIMEOUT if self.timeout is None else self.timeout
        return timeout or None

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped Inspect AI scorer.

        Args:
            inputs: The input to evaluate. A dict with a ``messages`` list becomes the
                Inspect message history.
            outputs: The output to evaluate.
            expectations: Expected values. The Inspect target is read from
                ``expected_response``, ``expected_output``, ``reference``,
                ``expected_facts`` or ``target``.
            trace: MLflow trace used to resolve inputs, outputs and expectations.
            session: List of MLflow traces for multi-turn evaluation. Each trace becomes a
                user and an assistant message.

        Returns:
            Feedback with a ``yes``/``no`` value, the Inspect explanation as rationale and
            the numeric score, raw Inspect value and terminal state in metadata.
        """
        if self._is_model_graded:
            assessment_source = AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=self._model,
            )
        else:
            assessment_source = AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id=f"{_FRAMEWORK_NAME}/{self._metric_name}",
            )

        try:
            state, target = build_task_state(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
                session=session,
            )
            if target is None:
                if not self._is_custom:
                    raise MlflowException.invalid_parameter_value(
                        f"Inspect AI scorer '{self.name}' needs a target. Provide one of "
                        f"{list(TARGET_EXPECTATION_KEYS)} in expectations."
                    )
                from inspect_ai.scorer import Target

                target = Target("")

            score = run_scorer(self._scorer, state, target, timeout=self._effective_timeout())

            if score is None:
                return Feedback(
                    name=self.name,
                    error=AssessmentError(
                        error_code=_NOT_RUN_ERROR_CODE,
                        error_message="The Inspect AI scorer returned no score for this sample.",
                    ),
                    source=assessment_source,
                    metadata={
                        FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME,
                        TERMINAL_STATE_KEY: "not_run",
                    },
                )

            fields = score_to_feedback_fields(score, threshold=self._threshold)
            return Feedback(
                name=self.name,
                value=fields.value,
                rationale=fields.rationale,
                source=assessment_source,
                metadata={
                    FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME,
                    TERMINAL_STATE_KEY: "scored",
                    "threshold": self._threshold,
                    **fields.metadata,
                },
            )
        except Exception as e:
            _logger.error("Error evaluating Inspect AI scorer %s: %s", self.name, e)
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME, TERMINAL_STATE_KEY: "error"},
            )


@experimental(version="3.19.0")
@format_docstring(_MODEL_API_DOC)
def get_scorer(
    metric_name: str,
    model: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
    timeout: int | float | None = None,
    **metric_kwargs: Any,
) -> InspectAIScorer:
    """
    Get an Inspect AI scorer as an MLflow scorer.

    Args:
        metric_name: Name of an ``inspect_ai.scorer`` factory (e.g., ``"exact"``,
            ``"model_graded_fact"``).
        model: {{ model }}
        model_kwargs: Parameters for the grader LLM, for example ``temperature``.
        threshold: Minimum numeric score for a ``yes`` verdict (default 0.5).
        timeout: Per-call timeout in seconds. ``None`` uses MLflow's default scorer
            timeout and ``0`` disables it.
        metric_kwargs: Keyword arguments forwarded to the Inspect scorer factory.

    Returns:
        InspectAIScorer instance that can be called with MLflow's scorer interface.

    Examples:

    .. code-block:: python

        scorer = get_scorer("model_graded_fact", model="openai:/gpt-4o-mini")
        feedback = scorer(
            inputs="What is the capital of France?",
            outputs="Paris is the capital of France.",
            expectations={"expected_response": "Paris"},
        )

        scorer = get_scorer("match", location="any", ignore_case=True)
        feedback = scorer(outputs="The answer is Paris.", expectations={"target": "paris"})
    """
    return InspectAIScorer(
        metric_name=metric_name,
        model=model,
        model_kwargs=model_kwargs,
        threshold=threshold,
        timeout=timeout,
        **metric_kwargs,
    )


@experimental(version="3.19.0")
class ExactMatch(InspectAIScorer):
    """
    Wraps Inspect's ``exact`` scorer.

    Passes when the output matches the target exactly after Inspect's normalization
    (whitespace, case and punctuation are ignored).

    Examples:
        .. code-block:: python

            scorer = ExactMatch()
            feedback = scorer(outputs="Paris", expectations={"expected_response": "Paris"})
    """

    metric_name: ClassVar[str] = "ExactMatch"


@experimental(version="3.19.0")
class Includes(InspectAIScorer):
    """
    Wraps Inspect's ``includes`` scorer.

    Passes when the target text appears anywhere in the output.

    Args:
        ignore_case: Compare case-insensitively (default True).

    Examples:
        .. code-block:: python

            scorer = Includes()
            feedback = scorer(
                outputs="The capital of France is Paris.",
                expectations={"expected_response": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "Includes"


@experimental(version="3.19.0")
class Match(InspectAIScorer):
    """
    Wraps Inspect's ``match`` scorer.

    Passes when the target appears at the beginning or the end of the output, or anywhere
    when ``location="any"``.

    Args:
        location: Where to look for the target, one of ``"begin"``, ``"end"``, ``"any"`` or
            ``"exact"`` (default ``"end"``).
        ignore_case: Compare case-insensitively (default True).
        numeric: Compare numerically, so ``"1.0"`` matches ``"1"`` (default False).

    Examples:
        .. code-block:: python

            scorer = Match(location="any")
            feedback = scorer(outputs="It is Paris, I think.", expectations={"target": "Paris"})
    """

    metric_name: ClassVar[str] = "Match"


@experimental(version="3.19.0")
class Pattern(InspectAIScorer):
    """
    Wraps Inspect's ``pattern`` scorer.

    Extracts the answer from the output with a regular expression and compares the capture
    group against the target.

    Args:
        pattern: Regular expression with a capture group for the answer (required).
        ignore_case: Compare case-insensitively (default True).
        match_all: Require every capture group to match the target (default False).

    Examples:
        .. code-block:: python

            scorer = Pattern(pattern=r"Answer: (\\w+)")
            feedback = scorer(outputs="Answer: Paris", expectations={"target": "Paris"})
    """

    metric_name: ClassVar[str] = "Pattern"


@experimental(version="3.19.0")
class F1(InspectAIScorer):
    """
    Wraps Inspect's ``f1`` scorer.

    Scores the token-level F1 overlap between the output and the target (or the best of
    several targets). The score is a number between 0 and 1 and is compared against
    ``threshold`` for the verdict.

    Examples:
        .. code-block:: python

            scorer = F1(threshold=0.6)
            feedback = scorer(
                outputs="Paris is the capital",
                expectations={"expected_facts": ["Paris", "capital of France"]},
            )
    """

    metric_name: ClassVar[str] = "F1"


@experimental(version="3.19.0")
class Answer(InspectAIScorer):
    """
    Wraps Inspect's ``answer`` scorer.

    Looks for an ``ANSWER:`` line in the output, extracts the letter, word or line after it
    and compares it against the target.

    Args:
        pattern: What follows ``ANSWER:``, one of ``"letter"``, ``"word"`` or ``"line"``
            (required).

    Examples:
        .. code-block:: python

            scorer = Answer(pattern="letter")
            feedback = scorer(outputs="ANSWER: B", expectations={"target": "B"})
    """

    metric_name: ClassVar[str] = "Answer"


@experimental(version="3.19.0")
@format_docstring(_MODEL_API_DOC)
class ModelGradedFact(InspectAIScorer):
    """
    Wraps Inspect's ``model_graded_fact`` scorer.

    An LLM grader checks whether the facts in the target appear in the output. The grader
    answers with ``GRADE: C`` or ``GRADE: I`` (and ``GRADE: P`` with ``partial_credit``).

    Args:
        model: {{ model }}
        model_kwargs: Parameters for the grader LLM, for example ``temperature``.
        template: Custom grading template (see the Inspect docs for the placeholders).
        instructions: Extra grading instructions appended to the template.
        grade_pattern: Regular expression used to extract the grade.
        include_history: Include the full message history in the grading prompt.
        partial_credit: Allow the ``P`` grade, which maps to a score of 0.5.

    Examples:
        .. code-block:: python

            scorer = ModelGradedFact(model="openai:/gpt-4o-mini", partial_credit=True)
            feedback = scorer(
                inputs="What is the capital of France?",
                outputs="Paris is the capital of France.",
                expectations={"expected_response": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "ModelGradedFact"


@experimental(version="3.19.0")
@format_docstring(_MODEL_API_DOC)
class ModelGradedQA(InspectAIScorer):
    """
    Wraps Inspect's ``model_graded_qa`` scorer.

    An LLM grader decides whether the output answers the question according to the
    grading rubric in the target. The grader answers with ``GRADE: C``, ``GRADE: I`` or,
    with ``partial_credit``, ``GRADE: P``.

    Args:
        model: {{ model }}
        model_kwargs: Parameters for the grader LLM, for example ``temperature``.
        template: Custom grading template (see the Inspect docs for the placeholders).
        instructions: Extra grading instructions appended to the template.
        grade_pattern: Regular expression used to extract the grade.
        include_history: Include the full message history in the grading prompt.
        partial_credit: Allow the ``P`` grade, which maps to a score of 0.5.

    Examples:
        .. code-block:: python

            scorer = ModelGradedQA(model="openai:/gpt-4o-mini")
            feedback = scorer(
                inputs="Explain why the sky is blue.",
                outputs="Rayleigh scattering favors shorter wavelengths.",
                expectations={"reference": "Mentions Rayleigh scattering of sunlight."},
            )
    """

    metric_name: ClassVar[str] = "ModelGradedQA"


__all__ = [
    "InspectAIScorer",
    "get_scorer",
    "ExactMatch",
    "Includes",
    "Match",
    "Pattern",
    "F1",
    "Answer",
    "ModelGradedFact",
    "ModelGradedQA",
]
