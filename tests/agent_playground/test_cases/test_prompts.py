import json
from types import SimpleNamespace
from unittest import mock

import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import prompts, store
from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestSpec,
)
from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException


@pytest.fixture
def client(db_uri):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    yield MlflowClient(tracking_uri=db_uri)
    mlflow.set_tracking_uri(original)


@pytest.fixture
def experiment_id(client):
    return client.create_experiment("agent_playground_prompts_test")


# --- build_test_gen_prompt -----------------------------------------------


def _fake_assessment(rationale: str, anchor: dict[str, object] | None) -> Feedback:
    metadata = {"anchor": json.dumps(anchor)} if anchor else None
    return Feedback(
        name="agent_playground.feedback",
        value="quality",
        rationale=rationale,
        metadata=metadata,
    )


def _fake_trace(messages: list[dict[str, object]]) -> SimpleNamespace:
    root_span = SimpleNamespace(inputs={"messages": messages})
    return SimpleNamespace(data=SimpleNamespace(spans=[root_span]))


def test_test_gen_prompt_includes_rationale():
    assessment = _fake_assessment("agent should cite docs", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment) as get_a,
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace) as get_t,
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
        get_a.assert_called_once_with(trace_id="tr-1", assessment_id="fb-1")
        get_t.assert_called_once_with("tr-1")
    assert "agent should cite docs" in prompt


def test_test_gen_prompt_includes_anchored_substring():
    selected = "Use INFO for general advice"
    anchor = {
        "message_id": "msg-1",
        "start": 0,
        "end": len(selected),
        "selected_text": selected,
        "prefix": "",
        "suffix": "",
    }
    assessment = _fake_assessment("agent should cite docs", anchor=anchor)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "Use INFO for general advice" in prompt


def test_test_gen_prompt_includes_conversation_messages():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([
        {"role": "user", "content": "how do I set up logging?"},
        {"role": "assistant", "content": "Set the log level to INFO."},
    ])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "how do I set up logging?" in prompt
    assert "Set the log level to INFO." in prompt


def test_test_gen_prompt_degrades_when_anchor_missing():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no anchored substring" in prompt


def test_test_gen_prompt_degrades_when_anchor_malformed():
    assessment = Feedback(
        name="agent_playground.feedback",
        value="quality",
        rationale="x",
        metadata={"anchor": "not-json"},
    )
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no anchored substring" in prompt


def test_test_gen_prompt_degrades_when_no_conversation():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no prior conversation context" in prompt


def test_test_gen_prompt_includes_persona_guidance():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.MlflowClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    # The instructions must call out the persona block and tie it to the
    # ConversationSimulator dict shape so Claude knows what to emit for
    # multi-turn cases.
    assert "persona" in prompt
    assert "ConversationSimulator" in prompt


def test_test_gen_prompt_propagates_assessment_error():
    with mock.patch.object(prompts, "get_assessment", side_effect=MlflowException("nope")):
        with pytest.raises(MlflowException, match="nope"):
            prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")


# --- build_fix_prompt ----------------------------------------------------


def test_fix_prompt_raises_when_case_missing(experiment_id):
    with pytest.raises(MlflowException, match="not found"):
        prompts.build_fix_prompt(experiment_id, "tc-nonexistent")


def test_fix_prompt_includes_test_case_id_and_verify_command(experiment_id):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="agent must cite docs",
        assertion=AssertionSpec(must_contain=["docs"]),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert test_case_id in prompt
    assert "mlflow agent test run --test-case" in prompt


def test_fix_prompt_includes_rationale_summary(experiment_id):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="agent must cite docs on log levels",
        assertion=AssertionSpec(must_contain=["docs"]),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert "agent must cite docs on log levels" in prompt


def test_fix_prompt_assertion_strategy_renders_clauses(experiment_id):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="x",
        assertion=AssertionSpec(
            must_contain=["docs"],
            must_call_tool=["search_docs"],
            must_not_call_tool=["delete_record"],
        ),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert "must_contain" in prompt
    assert "docs" in prompt
    assert "must_call_tool" in prompt
    assert "search_docs" in prompt
    assert "must_not_call_tool" in prompt


def test_fix_prompt_judge_strategy_renders_criteria(experiment_id):
    spec = TestSpec(
        strategy="judge",
        rationale_summary="x",
        judge=JudgeSpec(criteria="response is friendly", expected_response="hi friend"),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert "response is friendly" in prompt
    assert "hi friend" in prompt


def test_fix_prompt_persona_renders_goal_and_guidelines(experiment_id):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="x",
        max_turns=3,
        assertion=AssertionSpec(must_contain=["docs"]),
        persona=PersonaSpec(
            goal="learn about logging",
            persona="terse Python developer",
            simulation_guidelines=["ask one question at a time", "stay technical"],
        ),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert "learn about logging" in prompt
    assert "terse Python developer" in prompt
    assert "ask one question at a time" in prompt
    assert "stay technical" in prompt
    assert "max_turns" in prompt


def test_fix_prompt_no_persona_indicates_single_turn(experiment_id):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="x",
        assertion=AssertionSpec(must_contain=["docs"]),
    )
    test_case_id = store.insert_case(experiment_id, spec)
    prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    assert "single-turn" in prompt
