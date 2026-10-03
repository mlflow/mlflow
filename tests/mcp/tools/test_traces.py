import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.mcp.tools.traces import (
    delete_trace_assessment,
    delete_trace_tag,
    delete_traces,
    get_trace,
    get_trace_assessment,
    log_trace_expectation,
    log_trace_feedback,
    search_traces,
    set_trace_tag,
    update_trace_assessment,
)


@pytest.fixture
def experiment_id():
    experiment_id = MlflowClient().create_experiment("exp")
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _log_trace(name: str = "span") -> str:
    with mlflow.start_span(name) as span:
        span.set_inputs({"q": "hi"})
        span.set_outputs({"a": "hello"})
    return span.trace_id


def test_search_traces_returns_full_traces(experiment_id):
    trace_id = _log_trace()
    page = search_traces(experiment_id)
    (trace,) = page.traces
    assert trace["info"]["trace_id"] == trace_id
    assert [span["name"] for span in trace["data"]["spans"]] == ["span"]
    assert page.next_page_token is None

    (trace,) = search_traces(experiment_id, include_spans=False).traces
    assert trace["data"]["spans"] == []


def test_search_traces_projects_fields_and_pages(experiment_id):
    trace_ids = [_log_trace(f"span-{i}") for i in range(3)]

    collected = []
    page_token = None
    while True:
        page = search_traces(
            experiment_id,
            max_results=2,
            page_token=page_token,
            order_by=["timestamp_ms ASC"],
            extract_fields=["info.trace_id"],
        )
        assert all(
            trace == {"info": {"trace_id": trace["info"]["trace_id"]}} for trace in page.traces
        )
        collected.extend(trace["info"]["trace_id"] for trace in page.traces)
        if not (page_token := page.next_page_token):
            break
    assert collected == trace_ids


def test_search_traces_rejects_unknown_fields(experiment_id):
    _log_trace()
    with pytest.raises(MlflowException, match="info.nope"):
        search_traces(experiment_id, extract_fields=["info.nope"])


def test_get_trace_with_and_without_projection(experiment_id):
    trace_id = _log_trace()
    assert get_trace(trace_id).trace["info"]["trace_id"] == trace_id
    projected = get_trace(trace_id, extract_fields=["info.trace_id", "info.state"])
    assert projected.trace == {"info": {"trace_id": trace_id, "state": "OK"}}


def test_delete_traces_by_id(experiment_id):
    keep = _log_trace()
    drop = _log_trace()
    result = delete_traces(experiment_id, trace_ids=[drop])
    assert result.deleted_count == 1
    assert [t["info"]["trace_id"] for t in search_traces(experiment_id).traces] == [keep]


def test_trace_tags(experiment_id):
    trace_id = _log_trace()
    assert set_trace_tag(trace_id, "env", "prod").value == "prod"
    assert get_trace(trace_id).trace["info"]["tags"]["env"] == "prod"

    delete_trace_tag(trace_id, "env")
    assert "env" not in get_trace(trace_id).trace["info"]["tags"]


def test_feedback_lifecycle(experiment_id):
    trace_id = _log_trace()
    logged = log_trace_feedback(
        trace_id,
        "relevance",
        value=0.9,
        source_type="HUMAN",
        source_id="reviewer@example.com",
        rationale="on topic",
        metadata={"round": "1"},
    )
    assert logged.kind == "feedback"
    assert logged.value == 0.9
    assert logged.source.source_type == "HUMAN"
    assert logged.source.source_id == "reviewer@example.com"
    assert logged.metadata == {"round": "1"}
    assert get_trace_assessment(trace_id, logged.assessment_id) == logged

    # Falsy values and an omitted rationale: the value changes, the rationale is kept.
    updated = update_trace_assessment(trace_id, logged.assessment_id, value=0)
    assert updated.value == 0
    assert updated.rationale == "on topic"

    ref = delete_trace_assessment(trace_id, logged.assessment_id)
    assert ref.assessment_id == logged.assessment_id
    with pytest.raises(MlflowException, match=logged.assessment_id):
        get_trace_assessment(trace_id, logged.assessment_id)


def test_expectation_lifecycle(experiment_id):
    trace_id = _log_trace()
    logged = log_trace_expectation(trace_id, "expected", value={"answer": "Paris"})
    assert logged.kind == "expectation"
    assert logged.value == {"answer": "Paris"}

    updated = update_trace_assessment(trace_id, logged.assessment_id, value=False)
    assert updated.kind == "expectation"
    assert updated.value is False


def test_feedback_source_needs_both_type_and_id(experiment_id):
    trace_id = _log_trace()
    logged = log_trace_feedback(trace_id, "q", value="good", source_type="HUMAN")
    assert logged.source.source_id != "HUMAN"
    assert logged.value == "good"


def test_assessment_metadata_keeps_non_string_values(experiment_id):
    trace_id = _log_trace()
    metadata = {"confidence": 0.9, "round": 1, "flags": ["a", "b"]}
    logged = log_trace_feedback(trace_id, "q", value=1, metadata=metadata)
    assert logged.metadata == metadata
    assert get_trace_assessment(trace_id, logged.assessment_id).metadata == metadata
    updated = update_trace_assessment(trace_id, logged.assessment_id, metadata={"round": 2})
    assert updated.metadata["round"] == 2


def test_invalid_metadata_is_rejected(experiment_id):
    trace_id = _log_trace()
    with pytest.raises(MlflowException, match="`metadata` must be a JSON object"):
        log_trace_feedback(trace_id, "q", value=1, metadata="[1, 2]")
