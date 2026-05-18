from unittest import mock

import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssertionExpectations,
    JudgeExpectations,
    PersonaSpec,
    TestCaseRow,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode


@pytest.fixture
def client(db_uri):
    # Sets the fluent tracking URI for the store layer (which uses the
    # global URI), and also yields a client used by `experiment_id` /
    # the cross-experiment isolation test to create experiments.
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    yield MlflowClient(tracking_uri=db_uri)
    mlflow.set_tracking_uri(original)


@pytest.fixture
def experiment_id(client):
    return client.create_experiment("agent_playground_store_test")


def _assertion_row(**overrides) -> TestCaseRow:
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": AssertionExpectations(
            must_contain=["docs"], must_call_tool=["search_docs"]
        ),
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


def _persona_row(**overrides) -> TestCaseRow:
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": AssertionExpectations(
            must_contain=["docs"], must_call_tool=["search_docs"]
        ),
        "rationale_summary": "agent must cite docs",
        "persona": PersonaSpec(
            goal="learn about logging",
            persona="terse Python developer",
            simulation_guidelines=["ask one question at a time"],
        ),
        "max_turns": 3,
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)


@pytest.fixture
def assertion_row():
    return _assertion_row()


@pytest.fixture
def judge_row():
    return _judge_row()


@pytest.fixture
def persona_row():
    return _persona_row()


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


def test_insert_returns_test_case_id(experiment_id, assertion_row):
    returned = store.insert_case(experiment_id, assertion_row)
    assert returned == assertion_row.test_case_id


def test_new_test_case_id_uses_expected_prefix():
    assert store.new_test_case_id().startswith("tc-")


def test_insert_then_list_returns_the_case(experiment_id):
    row = _assertion_row(
        conversation_messages=[{"role": "user", "content": "hi"}],
        source_feedback_ids=["fb-001"],
        source_trace_id="tr-001",
        source_assistant_message_id="msg-001",
    )
    store.insert_case(experiment_id, row)
    cases = store.list_cases(experiment_id)
    assert len(cases) == 1
    case = cases[0]
    assert case.test_case_id == row.test_case_id
    assert isinstance(case.expectations, AssertionExpectations)
    assert case.expectations.must_contain == ["docs"]
    assert case.source_feedback_ids == ["fb-001"]
    assert case.source_trace_id == "tr-001"
    assert case.source_assistant_message_id == "msg-001"
    assert case.conversation_messages == [{"role": "user", "content": "hi"}]
    assert not case.promoted


def test_get_case_finds_by_id(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    case = store.get_case(experiment_id, assertion_row.test_case_id)
    assert case is not None
    assert case.test_case_id == assertion_row.test_case_id


def test_get_case_returns_none_for_unknown_id(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    assert store.get_case(experiment_id, "tc-nonexistent") is None


@pytest.mark.parametrize(
    "row_factory",
    [_assertion_row, _judge_row, _persona_row],
    ids=["assertion", "judge", "persona"],
)
def test_row_roundtrips_through_store(experiment_id, row_factory):
    row = row_factory()
    store.insert_case(experiment_id, row)
    case = store.get_case(experiment_id, row.test_case_id)
    assert case is not None
    assert case == row


def test_delete_case_removes_row(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    assert store.delete_case(experiment_id, assertion_row.test_case_id)
    assert store.get_case(experiment_id, assertion_row.test_case_id) is None


def test_delete_case_returns_false_for_missing(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    assert not store.delete_case(experiment_id, "tc-nonexistent")


def test_delete_case_no_dataset_returns_false(experiment_id):
    assert not store.delete_case(experiment_id, "tc-anything")


def test_update_replaces_expectations(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    new_expectations = AssertionExpectations(must_contain=["log level", "docs"])
    updated = store.update_case(
        experiment_id, assertion_row.test_case_id, expectations=new_expectations
    )
    assert updated is not None
    assert isinstance(updated.expectations, AssertionExpectations)
    assert updated.expectations.must_contain == ["log level", "docs"]


def test_update_can_switch_strategy(experiment_id, assertion_row):
    # Discriminated union: assertion -> judge swap is a clean replacement,
    # no orphan-payload concern.
    store.insert_case(experiment_id, assertion_row)
    new_expectations = JudgeExpectations(instructions="be polite")
    updated = store.update_case(
        experiment_id, assertion_row.test_case_id, expectations=new_expectations
    )
    assert updated is not None
    assert isinstance(updated.expectations, JudgeExpectations)
    assert updated.expectations.instructions == "be polite"


def test_update_keeps_unchanged_fields(experiment_id):
    row = _assertion_row(
        source_feedback_ids=["fb-001"],
        source_trace_id="tr-001",
    )
    store.insert_case(experiment_id, row)
    updated = store.update_case(experiment_id, row.test_case_id, promoted=True)
    assert updated is not None
    assert updated.promoted
    assert updated.source_feedback_ids == ["fb-001"]
    assert updated.source_trace_id == "tr-001"
    assert isinstance(updated.expectations, AssertionExpectations)
    assert updated.expectations.must_contain == ["docs"]


def test_update_attach_source_feedback_id(experiment_id):
    row = _assertion_row(source_feedback_ids=["fb-001"])
    store.insert_case(experiment_id, row)
    updated = store.update_case(experiment_id, row.test_case_id, attach_source_feedback_id="fb-002")
    assert updated is not None
    assert updated.source_feedback_ids == ["fb-001", "fb-002"]


def test_update_attach_source_feedback_id_is_idempotent(experiment_id):
    row = _assertion_row(source_feedback_ids=["fb-001"])
    store.insert_case(experiment_id, row)
    store.update_case(experiment_id, row.test_case_id, attach_source_feedback_id="fb-002")
    store.update_case(experiment_id, row.test_case_id, attach_source_feedback_id="fb-002")
    case = store.get_case(experiment_id, row.test_case_id)
    assert case is not None
    assert case.source_feedback_ids == ["fb-001", "fb-002"]


def test_update_returns_none_for_missing_case(experiment_id):
    assert store.update_case(experiment_id, "tc-nope") is None


def test_list_anchors_includes_message_ids(experiment_id):
    row1 = _assertion_row(source_assistant_message_id="msg-1")
    row2 = _assertion_row(source_assistant_message_id="msg-2")
    store.insert_case(experiment_id, row1)
    store.insert_case(experiment_id, row2)
    anchors = store.list_anchors(experiment_id)
    assert (row1.source_assistant_message_id, row1.test_case_id) in anchors
    assert (row2.source_assistant_message_id, row2.test_case_id) in anchors


def test_list_anchors_none_when_unanchored(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    anchors = store.list_anchors(experiment_id)
    assert (None, assertion_row.test_case_id) in anchors


def test_list_summaries_returns_rationales(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    summaries = store.list_summaries(experiment_id)
    assert summaries == [("agent must cite docs", assertion_row.test_case_id)]


def test_multiple_inserts_get_unique_ids(experiment_id):
    ids = {store.insert_case(experiment_id, _assertion_row()) for _ in range(5)}
    assert len(ids) == 5


def test_explicit_test_case_id_is_respected(experiment_id):
    row = _assertion_row(test_case_id="tc-custom")
    returned = store.insert_case(experiment_id, row)
    assert returned == "tc-custom"
    assert store.get_case(experiment_id, "tc-custom") is not None


def test_explicit_test_case_id_collision_rejected(experiment_id):
    store.insert_case(experiment_id, _assertion_row(test_case_id="tc-collision"))
    with pytest.raises(MlflowException, match="already exists") as exc_info:
        store.insert_case(experiment_id, _assertion_row(test_case_id="tc-collision"))
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_explicit_test_case_id_collision_rejected_even_when_messages_differ(experiment_id):
    store.insert_case(
        experiment_id,
        _assertion_row(
            test_case_id="tc-collision",
            conversation_messages=[{"role": "user", "content": "first"}],
        ),
    )
    with pytest.raises(Exception, match="already exists"):
        store.insert_case(
            experiment_id,
            _assertion_row(
                test_case_id="tc-collision",
                conversation_messages=[{"role": "user", "content": "second"}],
            ),
        )


def test_datasets_are_isolated_between_experiments(client):
    exp_a = client.create_experiment("ap_store_test_a")
    exp_b = client.create_experiment("ap_store_test_b")
    store.insert_case(exp_a, _assertion_row())
    # Different experiment, different dataset: should not see exp_a's case.
    assert store.list_cases(exp_b) == []


def test_list_cases_skips_malformed_rows(experiment_id, assertion_row):
    # Insert one well-formed case, then write a malformed record
    # directly through the underlying dataset so it bypasses our
    # entity validation. ``list_cases`` should log a warning (via
    # the module logger; mlflow disables logger propagation so we
    # patch the logger directly rather than using caplog) and return
    # only the good row.
    store.insert_case(experiment_id, assertion_row)
    dataset = store.get_or_create_regression_dataset(experiment_id)
    dataset.merge_records([
        {
            "inputs": {"messages": [], "record_id": "tc-bad"},
            "expectations": {"kind": "judge"},  # missing required ``instructions``
            "tags": {"rationale_summary": "broken row"},
        }
    ])
    with mock.patch.object(store, "_logger") as mock_logger:
        cases = store.list_cases(experiment_id)
    assert [c.test_case_id for c in cases] == [assertion_row.test_case_id]
    mock_logger.warning.assert_called_once()
    assert "tc-bad" in str(mock_logger.warning.call_args)


def test_list_cases_skips_row_with_non_numeric_max_turns(experiment_id, assertion_row):
    # ``tags.max_turns`` is written as ``str(int)``; a corrupted row
    # with a non-numeric value should fall back to the entity default
    # rather than poison the listing.
    store.insert_case(experiment_id, assertion_row)
    dataset = store.get_or_create_regression_dataset(experiment_id)
    dataset.merge_records([
        {
            "inputs": {"messages": [], "record_id": "tc-bad-max-turns"},
            "expectations": {
                "kind": "assertion",
                "must_contain": ["x"],
                "must_not_contain": [],
                "must_call_tool": [],
                "must_not_call_tool": [],
            },
            "tags": {
                "rationale_summary": "ok",
                "max_turns": "not-a-number",
            },
        }
    ])
    cases = store.list_cases(experiment_id)
    bad = next((c for c in cases if c.test_case_id == "tc-bad-max-turns"), None)
    assert bad is not None
    assert bad.max_turns == TestCaseRow.model_fields["max_turns"].default


def test_update_case_restores_on_merge_failure(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    original = store.get_case(experiment_id, assertion_row.test_case_id)
    assert original is not None

    real_merge = EvaluationDataset.merge_records
    call_count = {"n": 0}

    def flaky_merge(self, records):
        call_count["n"] += 1
        # Fail the first merge (the new record); allow the restore
        # merge (the second call) to proceed normally.
        if call_count["n"] == 1:
            raise MlflowException("synthetic merge failure")
        return real_merge(self, records)

    with mock.patch.object(EvaluationDataset, "merge_records", flaky_merge):
        with pytest.raises(MlflowException, match="synthetic merge failure"):
            store.update_case(experiment_id, assertion_row.test_case_id, promoted=True)

    # Both merges happened: the failing new-record merge and the
    # restore merge that re-inserted the original record.
    assert call_count["n"] == 2
    restored = store.get_case(experiment_id, assertion_row.test_case_id)
    assert restored == original


def test_update_case_logs_when_restore_also_fails(experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)

    def always_fails(self, records):
        raise MlflowException("synthetic merge failure")

    with (
        mock.patch.object(EvaluationDataset, "merge_records", always_fails),
        mock.patch.object(store, "_logger") as mock_logger,
        pytest.raises(MlflowException, match="synthetic merge failure"),
    ):
        store.update_case(experiment_id, assertion_row.test_case_id, promoted=True)
    mock_logger.error.assert_called_once()
    assert "failed to restore" in str(mock_logger.error.call_args)
