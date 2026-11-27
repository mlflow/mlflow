import asyncio

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.inspect_ai.utils import (
    build_task_state,
    resolve_target,
    run_scorer,
    score_to_feedback_fields,
)


def _make_trace(inputs, outputs):
    with mlflow.start_span(name="root") as span:
        span.set_inputs(inputs)
        span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


@pytest.mark.parametrize(
    ("expectations", "expected"),
    [
        ({"expected_response": "Paris"}, ["Paris"]),
        ({"expected_output": "Paris", "reference": "Rome"}, ["Paris"]),
        ({"reference": "Rome", "target": "Lyon"}, ["Rome"]),
        ({"expected_facts": ["Paris", "France"]}, ["Paris", "France"]),
        ({"target": ("a", 2)}, ["a", "2"]),
        ({"expected_response": "", "target": "Lyon"}, ["Lyon"]),
        ({"expected_facts": [], "reference": "Rome"}, ["Rome"]),
        ({"expected_response": {"city": "Paris"}}, ['{"city": "Paris"}']),
    ],
)
def test_resolve_target_priority(expectations, expected):
    assert resolve_target(expectations).target == expected


@pytest.mark.parametrize("expectations", [None, {}, {"other": "x"}, {"expected_response": None}])
def test_resolve_target_without_target_key(expectations):
    assert resolve_target(expectations) is None


def test_build_task_state_from_strings():
    state, target = build_task_state(
        inputs="What is the capital of France?",
        outputs="Paris",
        expectations={"expected_response": "Paris"},
    )

    assert state.input_text == "What is the capital of France?"
    assert state.output.completion == "Paris"
    assert [(m.role, m.text) for m in state.messages] == [
        ("user", "What is the capital of France?"),
        ("assistant", "Paris"),
    ]
    assert state.sample_id == "mlflow-sample"
    assert state.epoch == 1
    assert str(state.model) == "mlflow/evaluated-app"
    assert target.text == "Paris"


def test_build_task_state_honors_message_history_and_drops_unknown_roles():
    inputs = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]},
            {"role": "tool", "content": "ignored"},
        ]
    }
    state, target = build_task_state(inputs=inputs, outputs="Paris")

    assert [(m.role, m.text) for m in state.messages] == [
        ("system", "Be brief."),
        ("user", "Capital of France?"),
        ("assistant", "Paris"),
    ]
    assert target is None


def test_build_task_state_without_inputs_or_outputs():
    state, target = build_task_state()

    assert state.input_text == ""
    assert state.output.completion == ""
    assert state.messages == []
    assert target is None


def test_build_task_state_resolves_trace():
    trace = _make_trace({"question": "Capital of France?"}, {"answer": "Paris"})

    state, target = build_task_state(trace=trace, expectations={"target": "Paris"})

    assert "Capital of France?" in state.input_text
    assert state.output.completion == '{"answer": "Paris"}'
    assert state.sample_id == trace.info.trace_id
    assert target.text == "Paris"


def test_build_task_state_from_session():
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

    state, target = build_task_state(session=session, expectations={"reference": "Paris"})

    assert [(m.role, m.text) for m in state.messages] == [
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "Capital of France?"),
        ("assistant", "Paris"),
    ]
    assert state.input_text == "Capital of France?"
    assert state.output.completion == "Paris"
    assert state.sample_id == session[-1].info.trace_id
    assert target.text == "Paris"


@pytest.mark.parametrize(
    ("raw_value", "threshold", "expected_value", "expected_score"),
    [
        ("C", 0.5, CategoricalRating.YES, 1.0),
        ("I", 0.5, CategoricalRating.NO, 0.0),
        ("P", 0.5, CategoricalRating.YES, 0.5),
        ("P", 0.6, CategoricalRating.NO, 0.5),
        ("N", 0.5, CategoricalRating.NO, 0.0),
        (0.7, 0.5, CategoricalRating.YES, 0.7),
        (0, 0.5, CategoricalRating.NO, 0.0),
        (True, 0.5, CategoricalRating.YES, 1.0),
        ("0.75", 0.5, CategoricalRating.YES, 0.75),
    ],
)
def test_score_to_feedback_fields_scalars(raw_value, threshold, expected_value, expected_score):
    fields = score_to_feedback_fields(Score(value=raw_value), threshold=threshold)

    assert fields.value == expected_value
    assert fields.rationale is None
    assert fields.metadata == {"raw_value": raw_value, "score": expected_score}


@pytest.mark.parametrize("raw_value", [[1, 0, 1], {"precision": 0.5, "recall": 1.0}])
def test_score_to_feedback_fields_passes_collections_through(raw_value):
    fields = score_to_feedback_fields(Score(value=raw_value), threshold=0.5)

    assert fields.value == raw_value
    assert fields.metadata == {"raw_value": raw_value}


def test_score_to_feedback_fields_keeps_explanation_answer_and_json_metadata():
    score = Score(
        value="C",
        answer="Paris",
        explanation="Matches the target.",
        metadata={"confidence": 0.9, "grading": [ChatMessageUser(content="prompt")]},
    )

    fields = score_to_feedback_fields(score, threshold=0.5)

    assert fields.rationale == "Matches the target."
    assert fields.metadata == {
        "raw_value": "C",
        "score": 1.0,
        "answer": "Paris",
        "confidence": 0.9,
    }


def test_run_scorer_awaits_the_scorer():
    async def scorer(state, target):
        return Score(value="C", explanation=target.text)

    state, _ = build_task_state(outputs="x")

    assert run_scorer(scorer, state, Target("t"), timeout=None).explanation == "t"


def test_run_scorer_times_out():
    async def slow(state, target):
        await asyncio.sleep(2)
        return Score(value="C")

    state, _ = build_task_state(outputs="x")

    with pytest.raises(MlflowException, match="timed out after 0.05 seconds"):
        run_scorer(slow, state, Target("t"), timeout=0.05)


def test_run_scorer_inside_a_running_event_loop_uses_a_worker_thread():
    async def scorer(state, target):
        return Score(value="C")

    state, _ = build_task_state(outputs="x")

    async def main():
        return run_scorer(scorer, state, Target("t"), timeout=None)

    assert asyncio.run(main()).value == "C"
