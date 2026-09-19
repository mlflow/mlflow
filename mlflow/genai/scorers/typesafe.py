"""TypeSafe System One integration for MLflow scorers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.utils.trace_utils import (
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)

if TYPE_CHECKING:
    from typesafe_sdk import Choice, Noul, Score

__all__ = ["make_typesafe_scorer"]

_logger = logging.getLogger(__name__)
_QUESTION_TYPES = {"noul", "choice", "score"}


def _normalize_question(
    question: Noul | Choice | Score | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(question, dict):
        value = question
    else:
        try:
            from typesafe_sdk import Choice, Noul, Score
        except ImportError as e:
            raise MlflowException(
                "The `typesafe-sdk` package is required to use TypeSafe question objects. "
                "Install it with `pip install typesafe-sdk`."
            ) from e

        if not isinstance(question, (Noul, Choice, Score)):
            raise MlflowException.invalid_parameter_value(
                "`question` must be a TypeSafe Noul, Choice, or Score question, or a dictionary."
            )

        if isinstance(question, Noul):
            question_type = "noul"
        elif isinstance(question, Choice):
            question_type = "choice"
        else:
            question_type = "score"
        value = {
            "type": question_type,
            "instructions": question.instructions,
        }
        if question.criteria is not None:
            value["criteria"] = question.criteria

    question_type = value.get("type")
    if question_type not in _QUESTION_TYPES:
        raise MlflowException.invalid_parameter_value(
            f"`question.type` must be one of {sorted(_QUESTION_TYPES)}, got {question_type!r}."
        )

    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as e:
        raise MlflowException.invalid_parameter_value(f"`question` must be JSON serializable: {e}")


def _typesafe_answer_to_feedback(
    *,
    name: str,
    answer: Any,
    requested_model: str,
    resolved_model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Feedback:
    from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

    metadata = {FRAMEWORK_METADATA_KEY: "typesafe"}
    if input_tokens is not None:
        metadata["typesafe.input_tokens"] = str(input_tokens)
    if output_tokens is not None:
        metadata["typesafe.output_tokens"] = str(output_tokens)
    if resolved_model != requested_model:
        metadata["typesafe.requested_model"] = requested_model

    if isinstance(answer, NoulAnswer):
        answer_type = "noul"
        value = answer.noul
    elif isinstance(answer, ChoiceAnswer):
        answer_type = "choice"
        value = answer.choice
        metadata["typesafe.confidence"] = str(answer.confidence)
        metadata["typesafe.probabilities"] = json.dumps(answer.probabilities, sort_keys=True)
    elif isinstance(answer, ScoreAnswer):
        answer_type = "score"
        value = answer.score
        metadata["typesafe.confidence"] = str(answer.confidence)
        metadata["typesafe.probabilities"] = json.dumps(answer.probabilities, sort_keys=True)
        metadata["typesafe.legend"] = json.dumps(answer.legend, sort_keys=True)
    else:
        raise MlflowException.invalid_parameter_value(
            f"TypeSafe returned an unsupported answer type: {type(answer).__name__}."
        )

    metadata["typesafe.answer_type"] = answer_type
    return Feedback(
        name=name,
        value=value,
        rationale=None,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=f"typesafe:/{resolved_model}",
        ),
        metadata=metadata,
    )


class _TypeSafeScorer(Scorer):
    _metric_name: str = PrivateAttr()
    _metric_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _model: str = PrivateAttr()
    _question: dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        metric_name: str,
        question: Noul | Choice | Score | dict[str, Any],
        model: str = "jev-latest",
        description: str | None = None,
    ):
        normalized_question = _normalize_question(question)
        super().__init__(name=metric_name, description=description)
        self._metric_name = metric_name
        self._metric_kwargs = {"question": normalized_question}
        self._model = model
        self._question = normalized_question

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=f"typesafe:/{self._model}",
        )
        try:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as e:
                raise MlflowException(
                    "The `typesafe-sdk` package is required to use TypeSafe scorers. "
                    "Install it with `pip install typesafe-sdk`."
                ) from e

            if trace is not None:
                inputs = resolve_inputs_from_trace(inputs, trace)
                outputs = resolve_outputs_from_trace(outputs, trace)
                expectations = resolve_expectations_from_trace(expectations, trace)

            state = {
                key: value
                for key, value in {
                    "inputs": inputs,
                    "outputs": outputs,
                    "expectations": expectations,
                }.items()
                if value is not None
            }
            with TypeSafeClient() as client:
                response = client.system_one(
                    state=state,
                    questions={self.name: self._question},
                    model=self._model,
                )

            return _typesafe_answer_to_feedback(
                name=self.name,
                answer=response.answers[self.name],
                requested_model=self._model,
                resolved_model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except Exception as e:
            _logger.error("Error evaluating TypeSafe scorer %s: %s", self.name, e)
            return Feedback(name=self.name, error=e, source=source)


def make_typesafe_scorer(
    name: str,
    question: Noul | Choice | Score | dict[str, Any],
    *,
    model: str = "jev-latest",
    description: str | None = None,
) -> Scorer:
    """Create an MLflow scorer backed by one TypeSafe System One question.

    Args:
        name: Name used for the scorer and its TypeSafe question.
        question: A TypeSafe Noul, Choice, or Score question, or its dictionary form.
        model: TypeSafe model used for inference.
        description: Optional scorer description.

    Returns:
        A TypeSafe scorer that maps the native typed answer to MLflow ``Feedback``.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers.typesafe import make_typesafe_scorer

            scorer = make_typesafe_scorer(
                "relevant",
                {"type": "noul", "instructions": "Is `outputs` relevant to `inputs`?"},
            )
    """
    return _TypeSafeScorer(
        metric_name=name,
        question=question,
        model=model,
        description=description,
    )
