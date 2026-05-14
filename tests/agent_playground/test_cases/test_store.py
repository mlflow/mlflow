import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestSpec,
)


@pytest.fixture
def client(db_uri):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    yield MlflowClient(tracking_uri=db_uri)
    mlflow.set_tracking_uri(original)


@pytest.fixture
def experiment_id(client):
    return client.create_experiment("agent_playground_store_test")


@pytest.fixture
def assertion_spec():
    return TestSpec(
        strategy="assertion",
        rationale_summary="agent must cite docs",
        assertion=AssertionSpec(must_contain=["docs"], must_call_tool=["search_docs"]),
    )


@pytest.fixture
def judge_spec():
    return TestSpec(
        strategy="judge",
        rationale_summary="agent should sound friendlier",
        judge=JudgeSpec(criteria="response is friendly", expected_response="hi friend"),
    )


@pytest.fixture
def persona_spec(assertion_spec):
    return assertion_spec.model_copy(
        update={
            "persona": PersonaSpec(
                goal="learn about logging",
                persona="terse Python developer",
                simulation_guidelines=["ask one question at a time"],
            ),
            "max_turns": 3,
        }
    )


def test_regression_dataset_name_uses_experiment_id():
    assert store.regression_dataset_name("123") == "regression_suite_123"


def test_get_or_create_creates_dataset_first_call(experiment_id):
    dataset = store.get_or_create_regression_dataset(experiment_id)
    assert dataset.name == store.regression_dataset_name(experiment_id)


def test_get_or_create_is_idempotent(experiment_id):
    first = store.get_or_create_regression_dataset(experiment_id)
    second = store.get_or_create_regression_dataset(experiment_id)
    assert first.dataset_id == second.dataset_id


def test_list_cases_returns_empty_when_no_dataset(experiment_id):
    assert store.list_cases(experiment_id) == []


def test_insert_returns_test_case_id(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    assert test_case_id.startswith("tc-")


def test_insert_then_list_returns_the_case(experiment_id, assertion_spec):
    test_case_id = store.insert_case(
        experiment_id,
        assertion_spec,
        conversation_messages=[{"role": "user", "content": "hi"}],
        source_feedback_id="fb-001",
        source_trace_id="tr-001",
        source_assistant_message_id="msg-001",
    )
    cases = store.list_cases(experiment_id)
    assert len(cases) == 1
    case = cases[0]
    assert case.test_case_id == test_case_id
    assert case.spec.strategy == "assertion"
    assert case.spec.assertion is not None
    assert case.spec.assertion.must_contain == ["docs"]
    assert case.source_feedback_id == "fb-001"
    assert case.source_feedback_ids == ["fb-001"]
    assert case.source_trace_id == "tr-001"
    assert case.source_assistant_message_id == "msg-001"
    assert case.conversation_messages == [{"role": "user", "content": "hi"}]
    assert not case.promoted


def test_get_case_finds_by_id(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    case = store.get_case(experiment_id, test_case_id)
    assert case is not None
    assert case.test_case_id == test_case_id


def test_get_case_returns_none_for_unknown_id(experiment_id, assertion_spec):
    store.insert_case(experiment_id, assertion_spec)
    assert store.get_case(experiment_id, "tc-nonexistent") is None


def test_judge_strategy_roundtrips(experiment_id, judge_spec):
    test_case_id = store.insert_case(experiment_id, judge_spec)
    case = store.get_case(experiment_id, test_case_id)
    assert case is not None
    assert case.spec.strategy == "judge"
    assert case.spec.judge is not None
    assert case.spec.judge.criteria == "response is friendly"
    assert case.spec.judge.expected_response == "hi friend"
    assert case.spec.assertion is None


def test_persona_roundtrips(experiment_id, persona_spec):
    test_case_id = store.insert_case(experiment_id, persona_spec)
    case = store.get_case(experiment_id, test_case_id)
    assert case is not None
    assert case.spec.persona is not None
    assert case.spec.persona.goal == "learn about logging"
    assert case.spec.persona.simulation_guidelines == ["ask one question at a time"]
    assert case.spec.max_turns == 3


def test_delete_case_removes_row(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    assert store.delete_case(experiment_id, test_case_id)
    assert store.get_case(experiment_id, test_case_id) is None


def test_delete_case_returns_false_for_missing(experiment_id, assertion_spec):
    store.insert_case(experiment_id, assertion_spec)
    assert not store.delete_case(experiment_id, "tc-nonexistent")


def test_delete_case_no_dataset_returns_false(experiment_id):
    assert not store.delete_case(experiment_id, "tc-anything")


def test_update_replaces_spec(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    new_spec = assertion_spec.model_copy(
        update={"assertion": AssertionSpec(must_contain=["log level", "docs"])}
    )
    updated = store.update_case(experiment_id, test_case_id, spec=new_spec)
    assert updated is not None
    assert updated.spec.assertion is not None
    assert updated.spec.assertion.must_contain == ["log level", "docs"]


def test_update_keeps_unchanged_fields(experiment_id, assertion_spec):
    test_case_id = store.insert_case(
        experiment_id,
        assertion_spec,
        source_feedback_id="fb-001",
        source_trace_id="tr-001",
    )
    updated = store.update_case(experiment_id, test_case_id, promoted=True)
    assert updated is not None
    assert updated.promoted
    assert updated.source_feedback_id == "fb-001"
    assert updated.source_trace_id == "tr-001"
    assert updated.spec.assertion is not None
    assert updated.spec.assertion.must_contain == ["docs"]


def test_update_attach_source_feedback_id(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec, source_feedback_id="fb-001")
    updated = store.update_case(experiment_id, test_case_id, attach_source_feedback_id="fb-002")
    assert updated is not None
    assert updated.source_feedback_ids == ["fb-001", "fb-002"]


def test_update_attach_source_feedback_id_is_idempotent(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec, source_feedback_id="fb-001")
    store.update_case(experiment_id, test_case_id, attach_source_feedback_id="fb-002")
    store.update_case(experiment_id, test_case_id, attach_source_feedback_id="fb-002")
    case = store.get_case(experiment_id, test_case_id)
    assert case is not None
    assert case.source_feedback_ids == ["fb-001", "fb-002"]


def test_update_returns_none_for_missing_case(experiment_id):
    assert store.update_case(experiment_id, "tc-nope") is None


def test_list_anchors_includes_message_ids(experiment_id, assertion_spec):
    tc1 = store.insert_case(experiment_id, assertion_spec, source_assistant_message_id="msg-1")
    tc2 = store.insert_case(experiment_id, assertion_spec, source_assistant_message_id="msg-2")
    anchors = store.list_anchors(experiment_id)
    assert ("msg-1", tc1) in anchors
    assert ("msg-2", tc2) in anchors


def test_list_anchors_none_when_unanchored(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    anchors = store.list_anchors(experiment_id)
    assert (None, test_case_id) in anchors


def test_list_summaries_returns_rationales(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    summaries = store.list_summaries(experiment_id)
    assert summaries == [("agent must cite docs", test_case_id)]


def test_multiple_inserts_get_unique_ids(experiment_id, assertion_spec):
    ids = {store.insert_case(experiment_id, assertion_spec) for _ in range(5)}
    assert len(ids) == 5


def test_explicit_test_case_id_is_respected(experiment_id, assertion_spec):
    test_case_id = store.insert_case(experiment_id, assertion_spec, test_case_id="tc-custom")
    assert test_case_id == "tc-custom"
    assert store.get_case(experiment_id, "tc-custom") is not None


def test_get_or_create_recovers_for_other_experiment(client, assertion_spec):
    exp_a = client.create_experiment("ap_store_test_a")
    exp_b = client.create_experiment("ap_store_test_b")
    store.insert_case(exp_a, assertion_spec)
    # Different experiment, different dataset: should not see exp_a's case.
    assert store.list_cases(exp_b) == []
