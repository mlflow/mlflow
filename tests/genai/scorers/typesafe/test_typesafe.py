import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mlflow
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.typesafe import make_typesafe_scorer
from mlflow.telemetry.events import _get_scorer_class_name_for_tracking


class Noul:
    def __init__(self, *, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class Choice:
    def __init__(self, *, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class Score:
    def __init__(self, *, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class NoulAnswer:
    def __init__(self, *, noul):
        self.noul = noul


class ChoiceAnswer:
    def __init__(self, *, choice, confidence, probabilities):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities


class ScoreAnswer:
    def __init__(self, *, score, confidence, probabilities, legend):
        self.score = score
        self.confidence = confidence
        self.probabilities = probabilities
        self.legend = legend


class FakeTypeSafeClient:
    response = None
    instance = None

    def __init__(self):
        self.system_one = Mock(return_value=self.response)
        type(self).instance = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


@pytest.fixture(autouse=True)
def mock_typesafe_sdk(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(
            TypeSafeClient=FakeTypeSafeClient,
            Noul=Noul,
            Choice=Choice,
            Score=Score,
            NoulAnswer=NoulAnswer,
            ChoiceAnswer=ChoiceAnswer,
            ScoreAnswer=ScoreAnswer,
        ),
    )


def make_response(answer, input_tokens=12, output_tokens=3, model="jev-latest"):
    return SimpleNamespace(
        answers={"quality": answer},
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
    )


@pytest.mark.parametrize(
    ("answer", "question_type", "expected_value", "expected_metadata"),
    [
        (
            NoulAnswer(noul=0.84),
            "noul",
            0.84,
            {"typesafe.answer_type": "noul"},
        ),
        (
            ChoiceAnswer(
                choice="good",
                confidence=0.9,
                probabilities={"bad": 0.1, "good": 0.9},
            ),
            "choice",
            "good",
            {
                "typesafe.answer_type": "choice",
                "typesafe.confidence": "0.9",
                "typesafe.probabilities": '{"bad": 0.1, "good": 0.9}',
            },
        ),
        (
            ScoreAnswer(
                score=1.7,
                confidence=0.75,
                probabilities={0: 0.1, 1: 0.2, 2: 0.7},
                legend={0: "poor", 1: "ok", 2: "great"},
            ),
            "score",
            1.7,
            {
                "typesafe.answer_type": "score",
                "typesafe.confidence": "0.75",
                "typesafe.probabilities": '{"0": 0.1, "1": 0.2, "2": 0.7}',
                "typesafe.legend": '{"0": "poor", "1": "ok", "2": "great"}',
            },
        ),
    ],
)
def test_typesafe_scorer_maps_answers(answer, question_type, expected_value, expected_metadata):
    FakeTypeSafeClient.response = make_response(answer)
    scorer = make_typesafe_scorer(
        "quality", {"type": question_type, "instructions": "Judge the output."}
    )

    feedback = scorer(
        inputs={"question": "What is MLflow?"},
        outputs="An ML platform.",
        expectations={"expected_response": "An ML platform."},
    )

    assert feedback.value == expected_value
    assert feedback.rationale is None
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "typesafe:/jev-latest"
    assert feedback.metadata == {
        "mlflow.scorer.framework": "typesafe",
        "typesafe.input_tokens": "12",
        "typesafe.output_tokens": "3",
        **expected_metadata,
    }
    FakeTypeSafeClient.instance.system_one.assert_called_once_with(
        state={
            "inputs": {"question": "What is MLflow?"},
            "outputs": "An ML platform.",
            "expectations": {"expected_response": "An ML platform."},
        },
        questions={"quality": {"type": question_type, "instructions": "Judge the output."}},
        model="jev-latest",
    )


def test_typesafe_scorer_uses_resolved_model():
    FakeTypeSafeClient.response = make_response(NoulAnswer(noul=0.8), model="jev-2026-09-01")

    feedback = make_typesafe_scorer(
        "quality", {"type": "noul", "instructions": "Relevant?"}, model="jev-latest"
    )(outputs="Yes")

    assert feedback.source.source_id == "typesafe:/jev-2026-09-01"
    assert feedback.metadata["typesafe.requested_model"] == "jev-latest"


def test_typesafe_scorer_omits_none_state_fields():
    FakeTypeSafeClient.response = make_response(NoulAnswer(noul=0.2))

    make_typesafe_scorer("quality", {"type": "noul", "instructions": "Relevant?"})(outputs="No")

    assert FakeTypeSafeClient.instance.system_one.call_args.kwargs["state"] == {"outputs": "No"}


def test_typesafe_scorer_extracts_state_from_trace():
    @mlflow.trace(name="answer", span_type=SpanType.CHAIN)
    def answer(question):
        return f"Answer to: {question}"

    answer(question="Why?")
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    FakeTypeSafeClient.response = make_response(NoulAnswer(noul=0.9))

    make_typesafe_scorer("quality", {"type": "noul", "instructions": "Relevant?"})(trace=trace)

    assert FakeTypeSafeClient.instance.system_one.call_args.kwargs["state"] == {
        "inputs": {"question": "Why?"},
        "outputs": "Answer to: Why?",
    }


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (Noul(instructions="Relevant?"), {"type": "noul", "instructions": "Relevant?"}),
        (
            Choice(instructions="Quality?", criteria={"good": None, "bad": None}),
            {
                "type": "choice",
                "instructions": "Quality?",
                "criteria": {"good": None, "bad": None},
            },
        ),
        (
            Score(instructions="Quality?", criteria=["bad", "good"]),
            {"type": "score", "instructions": "Quality?", "criteria": ["bad", "good"]},
        ),
    ],
)
def test_typesafe_scorer_normalizes_sdk_questions(question, expected):
    scorer = make_typesafe_scorer("quality", question)

    assert scorer._question == expected


def test_typesafe_scorer_rejects_unrelated_model_object():
    question = Mock()
    question.model_dump.return_value = {"type": "noul", "instructions": "Relevant?"}

    with pytest.raises(MlflowException, match="must be a TypeSafe"):
        make_typesafe_scorer("quality", question)

    question.model_dump.assert_not_called()


def test_typesafe_scorer_validates_dictionary_question_type():
    with pytest.raises(MlflowException, match="question.type"):
        make_typesafe_scorer("quality", {"type": "unknown", "instructions": "Relevant?"})


def test_typesafe_scorer_returns_error_feedback(monkeypatch):
    FakeTypeSafeClient.response = RuntimeError("service unavailable")
    scorer = make_typesafe_scorer("quality", {"type": "noul", "instructions": "Relevant?"})
    monkeypatch.setattr(
        FakeTypeSafeClient,
        "__enter__",
        Mock(side_effect=RuntimeError("service unavailable")),
    )

    feedback = scorer(outputs="test")

    assert feedback.error is not None
    assert "service unavailable" in str(feedback.error)


def test_typesafe_scorer_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    scorer = make_typesafe_scorer("quality", {"type": "noul", "instructions": "Relevant?"})

    feedback = scorer(outputs="test")

    assert feedback.error is not None
    assert "pip install typesafe-sdk" in str(feedback.error)


def test_typesafe_scorer_factory_and_serialization_round_trip():
    scorer = make_typesafe_scorer(
        "quality",
        {"type": "choice", "instructions": "Quality?", "criteria": {"good": None}},
        model="jev-preview",
        description="Output quality",
    )

    assert scorer.kind == ScorerKind.THIRD_PARTY
    dump = scorer.model_dump()
    assert dump["third_party_scorer_data"] == {
        "module": "mlflow.genai.scorers.typesafe",
        "class": "_TypeSafeScorer",
        "metric_name": "quality",
        "model": "jev-preview",
        "kwargs": {
            "question": {
                "type": "choice",
                "instructions": "Quality?",
                "criteria": {"good": None},
            }
        },
    }

    restored = Scorer.model_validate(dump)

    assert type(restored) is type(scorer)
    assert restored.name == "quality"
    assert restored.description == "Output quality"
    assert restored._question == scorer._question
    assert restored._model == "jev-preview"


def test_typesafe_scorer_tracking_class_name():
    scorer = make_typesafe_scorer("quality", {"type": "noul", "instructions": "Relevant?"})

    assert _get_scorer_class_name_for_tracking(scorer) == "TypeSafe:quality"


def test_typesafe_module_exports_only_factory():
    from mlflow.genai.scorers import typesafe

    assert typesafe.__all__ == ["make_typesafe_scorer"]
    assert not hasattr(typesafe, "TypeSafeScorer")
