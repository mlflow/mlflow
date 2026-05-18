import json
from types import SimpleNamespace
from unittest import mock

import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import prompts, store
from mlflow.agent_playground.test_cases.entities import (
    AssertionExpectations,
    JudgeExpectations,
    PersonaSpec,
    TestCaseRow,
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


def _assertion_row(**overrides) -> TestCaseRow:
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": AssertionExpectations(must_contain=["docs"]),
        "rationale_summary": "agent must cite docs",
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)


def _judge_row(**overrides) -> TestCaseRow:
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": JudgeExpectations(
            instructions="response is friendly", expected_response="hi friend"
        ),
        "rationale_summary": "agent should sound friendlier",
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)


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
    # The root-span finder walks `parent_id is None`; mirror that on the
    # fake so the production extractor (which calls
    # `trace.data._get_root_span()`) picks this span as root.
    root_span = SimpleNamespace(parent_id=None, inputs={"messages": messages})
    return SimpleNamespace(
        data=SimpleNamespace(
            spans=[root_span],
            _get_root_span=lambda: root_span,
        )
    )


def test_test_gen_prompt_includes_rationale():
    assessment = _fake_assessment("agent should cite docs", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment) as get_a,
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace) as get_t,
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
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
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
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "how do I set up logging?" in prompt
    assert "Set the log level to INFO." in prompt


def test_test_gen_prompt_degrades_when_anchor_missing():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
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
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no anchored substring" in prompt


def test_test_gen_prompt_degrades_when_anchor_fails_validation():
    # Anchor JSON is well-formed but selected_text length disagrees
    # with end-start; the range validator rejects construction and the
    # builder degrades to no-anchor.
    anchor = {
        "message_id": "msg-1",
        "start": 0,
        "end": 5,
        "selected_text": "way longer than five chars",
        "prefix": "",
        "suffix": "",
    }
    assessment = _fake_assessment("x", anchor=anchor)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no anchored substring" in prompt


def test_test_gen_prompt_degrades_when_rationale_missing():
    # ``Feedback.rationale`` may be ``None`` (the widget allows
    # rationale-less feedback); the builder falls back to a placeholder
    # rather than emitting "None" verbatim.
    assessment = Feedback(
        name="agent_playground.feedback",
        value="quality",
        rationale=None,
        metadata=None,
    )
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no rationale recorded" in prompt
    assert "None" not in prompt.split("# Anchored substring")[0]


def test_test_gen_prompt_degrades_when_no_conversation():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "no prior conversation context" in prompt


def test_test_gen_prompt_documents_discriminated_union_shape():
    # Pin the specific instruction phrases — looser substrings like
    # "expectations" or "kind" would pass even if the schema-instruction
    # copy was removed entirely.
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    assert "GeneratedTestCase" in prompt
    assert "discriminated union on ``kind``" in prompt
    assert 'kind="assertion"' in prompt
    assert 'kind="judge"' in prompt
    assert "``instructions``" in prompt


def test_test_gen_prompt_includes_persona_block_instruction():
    assessment = _fake_assessment("x", anchor=None)
    trace = _fake_trace([])
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", return_value=trace),
    ):
        prompt = prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")
    # Specific instruction phrases — looser substrings like "persona" or
    # "ConversationSimulator" would pass even if the instructions copy
    # was removed entirely.
    assert "populate the\n   ``persona`` field" in prompt
    assert "test-case dict" in prompt
    assert "mlflow.genai.simulators.ConversationSimulator" in prompt


def test_test_gen_prompt_propagates_assessment_error():
    with mock.patch.object(prompts, "get_assessment", side_effect=MlflowException("nope")):
        with pytest.raises(MlflowException, match="nope"):
            prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")


def test_test_gen_prompt_wraps_non_mlflow_assessment_error():
    # Non-MlflowException from the assessment loader gets wrapped with
    # debuggable context.
    with mock.patch.object(prompts, "get_assessment", side_effect=RuntimeError("boom")):
        with pytest.raises(MlflowException, match="Failed to load assessment"):
            prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")


def test_test_gen_prompt_wraps_non_mlflow_trace_error():
    assessment = _fake_assessment("x", anchor=None)
    with (
        mock.patch.object(prompts, "get_assessment", return_value=assessment),
        mock.patch.object(prompts.TracingClient, "get_trace", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(MlflowException, match="Failed to load trace"):
            prompts.build_test_gen_prompt(trace_id="tr-1", assessment_id="fb-1")


# --- build_fix_prompt ----------------------------------------------------


def test_fix_prompt_raises_when_case_missing(experiment_id):
    with pytest.raises(MlflowException, match="not found"):
        prompts.build_fix_prompt(experiment_id, "tc-nonexistent")


def test_fix_prompt_includes_test_case_id_and_verify_command(experiment_id):
    row = _assertion_row()
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert row.test_case_id in prompt
    assert "mlflow agent test run --test-case" in prompt


def test_fix_prompt_includes_rationale_summary(experiment_id):
    row = _assertion_row(rationale_summary="agent must cite docs on log levels")
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "agent must cite docs on log levels" in prompt


def test_fix_prompt_assertion_strategy_renders_clauses(experiment_id):
    row = _assertion_row(
        expectations=AssertionExpectations(
            must_contain=["docs"],
            must_call_tool=["search_docs"],
            must_not_call_tool=["delete_record"],
        ),
        rationale_summary="x",
    )
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "must_contain" in prompt
    assert "docs" in prompt
    assert "must_call_tool" in prompt
    assert "search_docs" in prompt
    assert "must_not_call_tool" in prompt


def test_fix_prompt_judge_strategy_renders_instructions(experiment_id):
    row = _judge_row(rationale_summary="x")
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "instructions" in prompt
    assert "response is friendly" in prompt
    assert "hi friend" in prompt


def test_fix_prompt_persona_renders_goal_and_guidelines(experiment_id):
    row = _assertion_row(
        max_turns=3,
        persona=PersonaSpec(
            goal="learn about logging",
            persona="terse Python developer",
            simulation_guidelines=["ask one question at a time", "stay technical"],
        ),
        rationale_summary="x",
    )
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "learn about logging" in prompt
    assert "terse Python developer" in prompt
    assert "ask one question at a time" in prompt
    assert "stay technical" in prompt
    assert "max_turns" in prompt


def test_fix_prompt_persona_includes_context(experiment_id):
    row = _assertion_row(
        persona=PersonaSpec(
            goal="learn about logging",
            context={"user_id": "123", "tenant": "acme"},
        ),
        rationale_summary="x",
    )
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "context" in prompt
    assert "user_id" in prompt
    assert "tenant" in prompt


def test_fix_prompt_no_persona_indicates_single_turn(experiment_id):
    row = _assertion_row(rationale_summary="x")
    store.insert_case(experiment_id, row)
    prompt = prompts.build_fix_prompt(experiment_id, row.test_case_id)
    assert "single-turn" in prompt
