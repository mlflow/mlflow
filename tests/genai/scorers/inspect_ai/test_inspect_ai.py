import asyncio
from unittest.mock import patch

import pytest
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, model_graded_fact

import mlflow
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.inspect_ai import (
    F1,
    Answer,
    ExactMatch,
    Includes,
    InspectAIScorer,
    Match,
    ModelGradedFact,
    ModelGradedQA,
    Pattern,
    get_scorer,
)

_COMPLETE = "mlflow.genai.scorers.llm_backend.ScorerLLMClient.complete"


def _make_trace(inputs, outputs):
    with mlflow.start_span(name="root") as span:
        span.set_inputs(inputs)
        span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


@pytest.mark.parametrize(
    ("scorer", "outputs", "expectations", "expected_value", "expected_score", "raw_value"),
    [
        (ExactMatch(), "Paris", {"expected_response": "Paris"}, CategoricalRating.YES, 1.0, "C"),
        (ExactMatch(), "Rome", {"expected_response": "Paris"}, CategoricalRating.NO, 0.0, "I"),
        (
            Includes(),
            "The capital of France is Paris.",
            {"expected_output": "Paris"},
            CategoricalRating.YES,
            1.0,
            "C",
        ),
        (
            Match(location="any"),
            "It is Paris, I think.",
            {"target": "paris"},
            CategoricalRating.YES,
            1.0,
            "C",
        ),
        (
            Pattern(pattern=r"Answer: (\w+)"),
            "Answer: Paris",
            {"reference": "Paris"},
            CategoricalRating.YES,
            1.0,
            "C",
        ),
        (Answer(pattern="letter"), "ANSWER: B", {"target": "B"}, CategoricalRating.YES, 1.0, "C"),
        (
            F1(threshold=0.4),
            "Paris is the capital",
            {"expected_facts": ["Paris"]},
            CategoricalRating.YES,
            0.5,
            0.5,
        ),
        (
            F1(threshold=0.6),
            "Paris is the capital",
            {"expected_facts": ["Paris"]},
            CategoricalRating.NO,
            0.5,
            0.5,
        ),
    ],
)
def test_deterministic_scorers_run_real_inspect_scorers(
    scorer, outputs, expectations, expected_value, expected_score, raw_value
):
    feedback = scorer(outputs=outputs, expectations=expectations)

    assert feedback.error is None
    assert feedback.name == scorer.name
    assert feedback.value == expected_value
    assert feedback.source.source_type == AssessmentSourceType.CODE
    assert feedback.source.source_id == f"inspect_ai/{scorer.name}"
    assert feedback.metadata[FRAMEWORK_METADATA_KEY] == "inspect_ai"
    assert feedback.metadata["terminal_state"] == "scored"
    assert feedback.metadata["threshold"] == scorer._threshold
    assert feedback.metadata["score"] == expected_score
    assert feedback.metadata["raw_value"] == raw_value


def test_exact_match_feedback_fields():
    feedback = ExactMatch()(outputs="Paris", expectations={"expected_response": "Paris"})

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale is None
    assert feedback.metadata == {
        FRAMEWORK_METADATA_KEY: "inspect_ai",
        "terminal_state": "scored",
        "threshold": 0.5,
        "raw_value": "C",
        "score": 1.0,
        "answer": "Paris",
    }


def test_get_scorer_uses_inspect_factory_names_and_kwargs():
    scorer = get_scorer("match", location="any", threshold=0.9)

    assert isinstance(scorer, InspectAIScorer)
    assert scorer.name == "match"
    assert scorer.kind == ScorerKind.THIRD_PARTY
    assert scorer._metric_kwargs == {"location": "any", "threshold": 0.9}

    feedback = scorer(outputs="Yes, Paris.", expectations={"target": "paris"})
    assert feedback.value == CategoricalRating.YES


def test_missing_target_returns_error_feedback():
    feedback = ExactMatch()(outputs="Paris")

    assert feedback.value is None
    assert "needs a target" in feedback.error.error_message
    assert feedback.error.error_code == "MlflowException"
    assert feedback.source.source_type == AssessmentSourceType.CODE
    assert feedback.metadata == {FRAMEWORK_METADATA_KEY: "inspect_ai", "terminal_state": "error"}


def test_metric_name_is_required_without_a_custom_scorer():
    with pytest.raises(MlflowException, match="requires 'metric_name'"):
        InspectAIScorer()


def test_unknown_scorer_name_raises():
    with pytest.raises(MlflowException, match="Unknown Inspect AI scorer: 'no_such_scorer'"):
        get_scorer("no_such_scorer")


@pytest.mark.parametrize(
    ("grader_output", "partial_credit", "threshold", "expected_value", "expected_score"),
    [
        ("Checked the facts.\nGRADE: C", False, 0.5, CategoricalRating.YES, 1.0),
        ("Missing the capital.\nGRADE: I", False, 0.5, CategoricalRating.NO, 0.0),
        ("Half right.\nGRADE: P", True, 0.5, CategoricalRating.YES, 0.5),
        ("Half right.\nGRADE: P", True, 0.6, CategoricalRating.NO, 0.5),
    ],
)
def test_model_graded_fact_routes_grader_through_mlflow_backend(
    grader_output, partial_credit, threshold, expected_value, expected_score
):
    with patch(_COMPLETE, return_value=grader_output) as mock_complete:
        scorer = ModelGradedFact(
            model="openai:/gpt-4o-mini",
            model_kwargs={"temperature": 0.0},
            partial_credit=partial_credit,
            threshold=threshold,
        )
        feedback = scorer(
            inputs="What is the capital of France?",
            outputs="Paris is the capital of France.",
            expectations={"expected_response": "Paris"},
        )

    mock_complete.assert_called_once()
    messages = mock_complete.call_args.args[0]
    assert [m["role"] for m in messages] == ["user"]
    assert "What is the capital of France?" in messages[0]["content"]
    assert "Paris is the capital of France." in messages[0]["content"]
    assert mock_complete.call_args.kwargs == {"temperature": 0.0}

    assert feedback.error is None
    assert feedback.value == expected_value
    assert feedback.rationale == grader_output
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4o-mini"
    assert feedback.metadata[FRAMEWORK_METADATA_KEY] == "inspect_ai"
    assert feedback.metadata["terminal_state"] == "scored"
    assert feedback.metadata["score"] == expected_score
    assert feedback.metadata["threshold"] == threshold
    assert feedback.metadata["answer"] == "Paris is the capital of France."


def test_model_graded_qa_uses_default_model_when_none_given():
    with (
        patch(
            "mlflow.genai.scorers.inspect_ai.get_default_model", return_value="openai:/gpt-4.1-mini"
        ),
        patch(_COMPLETE, return_value="GRADE: C") as mock_complete,
    ):
        scorer = ModelGradedQA()
        feedback = scorer(
            inputs="Why is the sky blue?",
            outputs="Rayleigh scattering.",
            expectations={"reference": "Mentions Rayleigh scattering."},
        )

    mock_complete.assert_called_once()
    assert scorer._model == "openai:/gpt-4.1-mini"
    assert feedback.value == CategoricalRating.YES
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"


def test_grader_failure_becomes_error_feedback():
    with patch(_COMPLETE, side_effect=RuntimeError("provider down")):
        feedback = ModelGradedFact(model="openai:/gpt-4o-mini")(
            inputs="q", outputs="a", expectations={"expected_response": "r"}
        )

    assert feedback.value is None
    assert "provider down" in feedback.error.error_message
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.metadata == {FRAMEWORK_METADATA_KEY: "inspect_ai", "terminal_state": "error"}


def test_custom_inspect_scorer_with_inspect_mock_provider():
    grader = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "Nope.\nGRADE: I")],
    )
    scorer = InspectAIScorer("facts", scorer=model_graded_fact(model=grader))

    feedback = scorer(inputs="q", outputs="a", expectations={"reference": "r"})

    assert feedback.value == CategoricalRating.NO
    assert feedback.rationale == "Nope.\nGRADE: I"
    assert feedback.source.source_type == AssessmentSourceType.CODE
    assert feedback.source.source_id == "inspect_ai/facts"
    assert feedback.metadata["raw_value"] == "I"
    assert feedback.metadata["score"] == 0.0
    # Inspect's grading transcript holds ChatMessage objects, which are dropped from metadata.
    assert "grading" not in feedback.metadata


def test_custom_scorer_without_target_runs_with_empty_target():
    async def length_scorer(state, target):
        return Score(value=len(state.output.completion), explanation=f"target={target.text!r}")

    feedback = InspectAIScorer("length", scorer=length_scorer, threshold=3)(outputs="Paris")

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "target=''"
    assert feedback.metadata["score"] == 5.0


def test_custom_scorer_ignores_model_with_warning(caplog):
    async def noop(state, target):
        return Score(value="C")

    with caplog.at_level("WARNING", logger="mlflow.genai.scorers.inspect_ai"):
        scorer = InspectAIScorer("noop", scorer=noop, model="openai:/gpt-4o-mini")

    assert scorer._model is None
    assert "'model' is ignored" in caplog.text


def test_scorer_returning_none_is_reported_as_not_run():
    async def silent(state, target):
        return None

    feedback = InspectAIScorer("silent", scorer=silent)(outputs="x")

    assert feedback.value is None
    assert feedback.error.error_code == "INSPECT_AI_SCORER_NOT_RUN"
    assert feedback.metadata == {FRAMEWORK_METADATA_KEY: "inspect_ai", "terminal_state": "not_run"}


def test_scorer_timeout_is_an_error_feedback():
    async def slow(state, target):
        await asyncio.sleep(2)
        return Score(value="C")

    feedback = InspectAIScorer("slow", scorer=slow, timeout=0.05)(outputs="x")

    assert feedback.value is None
    assert "timed out after 0.05 seconds" in feedback.error.error_message
    assert feedback.metadata["terminal_state"] == "error"


def test_list_values_pass_through_without_threshold():
    async def multi(state, target):
        return Score(value=[1, 0, 1])

    feedback = InspectAIScorer("multi", scorer=multi)(outputs="x")

    assert feedback.value == [1, 0, 1]
    assert "score" not in feedback.metadata
    assert feedback.metadata["raw_value"] == [1, 0, 1]


def test_trace_inputs_and_outputs_are_resolved():
    trace = _make_trace({"question": "What is the capital of France?"}, "Paris")

    feedback = ExactMatch()(trace=trace, expectations={"expected_response": "Paris"})

    assert feedback.value == CategoricalRating.YES


def test_session_turns_become_message_history():
    seen = {}

    async def capture(state, target):
        seen["messages"] = [(m.role, m.text) for m in state.messages]
        seen["input"] = state.input_text
        seen["output"] = state.output.completion
        return Score(value="C")

    session = [
        _make_trace({"messages": [{"role": "user", "content": "Hi"}]}, "Hello!"),
        _make_trace(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Capital of France?"},
                ]
            },
            "Paris",
        ),
    ]
    feedback = InspectAIScorer("history", scorer=capture)(session=session)

    assert feedback.value == CategoricalRating.YES
    assert seen["messages"] == [
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "Capital of France?"),
        ("assistant", "Paris"),
    ]
    assert seen["input"] == "Capital of France?"
    assert seen["output"] == "Paris"


@pytest.mark.parametrize(
    ("scorer", "expected_data"),
    [
        (
            ExactMatch(threshold=0.9),
            {
                "module": "mlflow.genai.scorers.inspect_ai",
                "class": "ExactMatch",
                "metric_name": "ExactMatch",
                "model": None,
                "kwargs": {"threshold": 0.9},
            },
        ),
        (
            get_scorer("match", location="any", timeout=30),
            {
                "module": "mlflow.genai.scorers.inspect_ai",
                "class": "InspectAIScorer",
                "metric_name": "match",
                "model": None,
                "kwargs": {"location": "any", "threshold": 0.5, "timeout": 30},
            },
        ),
    ],
)
def test_serialization_round_trip(scorer, expected_data):
    dump = scorer.model_dump()

    assert dump["third_party_scorer_data"] == expected_data

    restored = Scorer.model_validate(dump)
    assert type(restored) is type(scorer)
    assert restored.name == scorer.name
    assert restored._metric_kwargs == scorer._metric_kwargs
    assert restored.timeout == scorer.timeout
    assert (
        restored(outputs="Paris", expectations={"target": "paris"}).value
        == scorer(outputs="Paris", expectations={"target": "paris"}).value
    )


def test_model_graded_serialization_keeps_model_and_model_kwargs():
    scorer = ModelGradedFact(model="openai:/gpt-4o-mini", model_kwargs={"temperature": 0.0})

    dump = scorer.model_dump()
    assert dump["third_party_scorer_data"] == {
        "module": "mlflow.genai.scorers.inspect_ai",
        "class": "ModelGradedFact",
        "metric_name": "ModelGradedFact",
        "model": "openai:/gpt-4o-mini",
        "kwargs": {"threshold": 0.5, "model_kwargs": {"temperature": 0.0}},
    }

    restored = Scorer.model_validate(dump)
    assert isinstance(restored, ModelGradedFact)
    assert restored._model == "openai:/gpt-4o-mini"
    assert restored._is_model_graded is True


def test_custom_scorer_cannot_be_serialized():
    async def noop(state, target):
        return Score(value="C")

    with pytest.raises(MlflowException, match="cannot be serialized"):
        InspectAIScorer("noop", scorer=noop).model_dump()


def test_align_is_not_supported():
    with pytest.raises(MlflowException, match="'align\\(\\)' is not supported"):
        ExactMatch().align()
