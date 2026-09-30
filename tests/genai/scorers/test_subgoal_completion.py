from unittest import mock

import pydantic
import pytest

import mlflow
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer, SubGoalCompletion
from mlflow.genai.scorers.builtin_scorers import _SubGoalResult, _SubGoalResults


def test_subgoal_completion_returns_named_feedback_in_goal_order():
    scorer = SubGoalCompletion(
        subgoals={
            "searched_docs": "The agent searched the docs.",
            "filed_ticket": "The agent filed a ticket.",
        },
        model="openai:/gpt-4o-mini",
    )
    judge_response = _SubGoalResults(
        results=[
            _SubGoalResult(goal_id="filed_ticket", result="no", rationale="No ticket was filed."),
            _SubGoalResult(
                goal_id="searched_docs", result="yes", rationale="The search tool was called."
            ),
        ]
    )
    trace = mock.Mock(spec=Trace)

    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output",
        return_value=judge_response,
    ) as judge:
        feedbacks = scorer.run(trace=trace)

    judge.assert_called_once()
    assert judge.call_args.kwargs["trace"] is trace
    assert judge.call_args.kwargs["output_schema"] is _SubGoalResults
    assert "searched_docs" in judge.call_args.kwargs["messages"][1].content
    assert [f.name for f in feedbacks] == [
        "subgoal_completion/searched_docs",
        "subgoal_completion/filed_ticket",
    ]
    assert [f.value for f in feedbacks] == ["yes", "no"]
    assert [f.rationale for f in feedbacks] == [
        "The search tool was called.",
        "No ticket was filed.",
    ]
    assert [f.metadata for f in feedbacks] == [
        {"subgoal": "The agent searched the docs."},
        {"subgoal": "The agent filed a ticket."},
    ]
    assert all(f.source.source_type == AssessmentSourceType.LLM_JUDGE for f in feedbacks)


@pytest.mark.parametrize(
    "goal_ids",
    [
        ["first"],
        ["first", "first"],
        ["first", "second", "extra"],
    ],
)
def test_subgoal_completion_rejects_incomplete_or_duplicate_judge_results(goal_ids):
    scorer = SubGoalCompletion(
        subgoals={"first": "First goal", "second": "Second goal"},
        model="openai:/gpt-4o-mini",
    )
    response = _SubGoalResults(
        results=[
            _SubGoalResult(goal_id=goal_id, result="yes", rationale="Found") for goal_id in goal_ids
        ]
    )

    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output",
        return_value=response,
    ):
        with pytest.raises(MlflowException, match="must return each goal ID once"):
            scorer(trace=mock.Mock(spec=Trace))


@pytest.mark.parametrize(
    ("subgoals", "error"),
    [
        ({}, "subgoals must contain at least one goal"),
        ({"bad/id": "A goal"}, "Goal ID"),
        ({"goal": "  "}, "must have a description"),
    ],
)
def test_subgoal_completion_rejects_invalid_goals(subgoals, error):
    with pytest.raises(pydantic.ValidationError, match=error):
        SubGoalCompletion(subgoals=subgoals)


@pytest.mark.parametrize("trace", [None, "not a trace"])
def test_subgoal_completion_requires_a_trace(trace):
    scorer = SubGoalCompletion(subgoals={"searched_docs": "Search the docs"})

    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"
    ) as judge:
        with pytest.raises(MlflowException, match="needs an MLflow trace"):
            scorer(trace=trace)

    judge.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("aggregations", ["max"]),
        ("extra_headers", {"X-Test": "value"}),
    ],
)
def test_subgoal_completion_rejects_unsupported_options(option, value):
    with pytest.raises(pydantic.ValidationError, match=option):
        SubGoalCompletion(subgoals={"searched_docs": "Search the docs"}, **{option: value})


def test_subgoal_completion_serialization_keeps_goal_ids_and_descriptions():
    scorer = SubGoalCompletion(
        name="support_goals",
        subgoals={"searched_docs": "The agent searched the docs."},
        model="openai:/gpt-4o-mini",
    )

    restored = Scorer.model_validate(scorer.model_dump())

    assert isinstance(restored, SubGoalCompletion)
    assert restored.name == scorer.name
    assert restored.subgoals == scorer.subgoals
    assert restored.model == scorer.model


def test_subgoal_completion_evaluate_keeps_each_goal_result(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        with mlflow.start_span(name="support_request") as span:
            span.set_inputs({"question": "How do I open a ticket?"})
            span.set_outputs("I searched the docs. Here is how to file a ticket.")
        trace = mlflow.get_trace(span.trace_id, flush=True)

        scorer = SubGoalCompletion(
            subgoals={"searched_docs": "Search the docs", "filed_ticket": "File a ticket"},
            model="openai:/gpt-4o-mini",
        )
        response = _SubGoalResults(
            results=[
                _SubGoalResult(goal_id="searched_docs", result="yes", rationale="Docs searched"),
                _SubGoalResult(goal_id="filed_ticket", result="no", rationale="No ticket call"),
            ]
        )
        with mock.patch(
            "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output",
            return_value=response,
        ):
            evaluation = mlflow.genai.evaluate(data=[{"trace": trace}], scorers=[scorer])

        assert evaluation.metrics["subgoal_completion/searched_docs/mean"] == 1.0
        assert evaluation.metrics["subgoal_completion/filed_ticket/mean"] == 0.0
    finally:
        mlflow.set_tracking_uri(previous_uri)
