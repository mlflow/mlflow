import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    PersonaSpec,
)
from mlflow.server.agent_playground.test_cases_router import (
    test_cases_router as _router,
)

from tests.agent_playground._factories import make_assertion_row


@pytest.fixture
def client(db_uri):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    yield MlflowClient(tracking_uri=db_uri)
    mlflow.set_tracking_uri(original)


@pytest.fixture
def experiment_id(client):
    return client.create_experiment("agent_playground_router_test")


@pytest.fixture
def assertion_row():
    return make_assertion_row()


@pytest.fixture
def http():
    app = FastAPI()
    app.include_router(_router)
    return TestClient(app)


@pytest.fixture
def seeded_case(experiment_id):
    row = make_assertion_row(
        conversation_messages=[{"role": "user", "content": "hi"}],
        source_feedback_ids=["fb-001"],
        source_trace_id="tr-001",
        source_assistant_message_id="msg-001",
    )
    return store.insert_case(experiment_id, row)


_PREFIX = "/ajax-api/3.0/mlflow/agent-playground"


def test_list_returns_empty_when_no_dataset(http, experiment_id):
    resp = http.get(f"{_PREFIX}/test-cases", params={"experiment_id": experiment_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["test_cases"] == []
    assert body["next_page_token"] is None


def test_list_returns_seeded_case(http, experiment_id, seeded_case):
    resp = http.get(f"{_PREFIX}/test-cases", params={"experiment_id": experiment_id})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["test_cases"]) == 1
    assert body["test_cases"][0]["test_case_id"] == seeded_case


def test_list_paginates(http, experiment_id):
    ids = [store.insert_case(experiment_id, make_assertion_row()) for _ in range(3)]
    resp = http.get(
        f"{_PREFIX}/test-cases",
        params={"experiment_id": experiment_id, "max_results": 2},
    )
    body = resp.json()
    assert len(body["test_cases"]) == 2
    assert body["next_page_token"] is not None

    resp2 = http.get(
        f"{_PREFIX}/test-cases",
        params={
            "experiment_id": experiment_id,
            "max_results": 2,
            "page_token": body["next_page_token"],
        },
    )
    body2 = resp2.json()
    assert len(body2["test_cases"]) == 1
    assert body2["next_page_token"] is None
    returned_ids = {tc["test_case_id"] for tc in body["test_cases"] + body2["test_cases"]}
    assert returned_ids == set(ids)


def test_list_rejects_invalid_page_token(http, experiment_id):
    resp = http.get(
        f"{_PREFIX}/test-cases",
        params={"experiment_id": experiment_id, "page_token": "not-base64-int"},
    )
    assert resp.status_code == 400


def test_list_rejects_zero_max_results(http, experiment_id):
    resp = http.get(
        f"{_PREFIX}/test-cases",
        params={"experiment_id": experiment_id, "max_results": 0},
    )
    assert resp.status_code == 422


def test_get_returns_seeded_case(http, experiment_id, seeded_case):
    resp = http.get(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
    )
    assert resp.status_code == 200
    assert resp.json()["test_case_id"] == seeded_case


def test_patch_updates_rationale_summary_only(http, experiment_id, seeded_case):
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
        json={"rationale_summary": "updated reason"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Verify "only" rationale_summary changed; all other fields preserved.
    assert body["rationale_summary"] == "updated reason"
    assert body["expectations"]["kind"] == "assertion"
    assert body["max_turns"] == 5
    assert body["persona"] is None
    assert body["expectations"] == {
        "kind": "assertion",
        "must_contain": ["docs"],
        "must_not_contain": [],
        "must_call_tool": ["search_docs"],
        "must_not_call_tool": [],
    }
    assert body["promoted"] is False
    assert body["source_feedback_ids"] == ["fb-001"]
    # ``conversation_messages`` round-trips through the
    # delete-then-insert path in ``store.update_case``; assert
    # preservation explicitly since a regression there is exactly
    # what this test is meant to guard against.
    assert body["conversation_messages"] == [{"role": "user", "content": "hi"}]


def test_patch_rejects_empty_body(http, experiment_id, seeded_case):
    # An empty PATCH body would otherwise trigger the
    # delete-then-insert path with nothing to update (resetting
    # ``created_time``/``source`` and exposing the read-side race);
    # the ``_reject_empty_patch`` model validator rejects at the wire
    # layer.
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
        json={},
    )
    assert resp.status_code == 422


def test_patch_switches_strategy_atomic_expectations(http, experiment_id, seeded_case):
    # Atomic expectations replacement: client sends the full new
    # ``JudgeExpectations`` payload to switch strategies. No orphan
    # ``assertion`` payload survives.
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
        json={
            "expectations": {
                "kind": "judge",
                "instructions": "response is friendly",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["expectations"]["kind"] == "judge"
    assert body["expectations"]["instructions"] == "response is friendly"


def test_patch_rejects_malformed_expectations(http, experiment_id, seeded_case):
    # ``kind="judge"`` requires ``instructions``; the discriminated
    # union rejects at the wire layer (FastAPI maps to 422).
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
        json={"expectations": {"kind": "judge"}},
    )
    assert resp.status_code == 422


def test_patch_clear_persona_removes_persona(http, experiment_id):
    row = make_assertion_row(persona=PersonaSpec(goal="g", persona="p"))
    store.insert_case(experiment_id, row)
    resp = http.patch(
        f"{_PREFIX}/test-cases/{row.test_case_id}",
        params={"experiment_id": experiment_id},
        json={"clear_persona": True},
    )
    assert resp.status_code == 200
    assert resp.json()["persona"] is None


def test_patch_clear_persona_with_persona_payload_returns_400(http, experiment_id):
    # Sending both ``persona`` and ``clear_persona=True`` is ambiguous;
    # ``store.update_case`` raises ``MlflowException(INVALID_PARAMETER_VALUE)``
    # which the router maps to 400.
    row = make_assertion_row(persona=PersonaSpec(goal="g", persona="p"))
    store.insert_case(experiment_id, row)
    resp = http.patch(
        f"{_PREFIX}/test-cases/{row.test_case_id}",
        params={"experiment_id": experiment_id},
        json={
            "clear_persona": True,
            "persona": {"goal": "new-goal"},
        },
    )
    assert resp.status_code == 400


def test_patch_returns_404_for_missing(http, experiment_id):
    resp = http.patch(
        f"{_PREFIX}/test-cases/tc-missing",
        params={"experiment_id": experiment_id},
        json={"rationale_summary": "x"},
    )
    assert resp.status_code == 404


def test_patch_judge_strategy_renders_instructions(http, experiment_id):
    row = make_assertion_row()
    store.insert_case(experiment_id, row)
    resp = http.patch(
        f"{_PREFIX}/test-cases/{row.test_case_id}",
        params={"experiment_id": experiment_id},
        json={
            "expectations": {
                "kind": "judge",
                "instructions": "be polite",
                "expected_response": "hi friend",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["expectations"]["kind"] == "judge"
    assert body["expectations"]["instructions"] == "be polite"
    assert body["expectations"]["expected_response"] == "hi friend"


def test_delete_removes_case(http, experiment_id, seeded_case):
    resp = http.delete(
        f"{_PREFIX}/test-cases/{seeded_case}",
        params={"experiment_id": experiment_id},
    )
    assert resp.status_code == 204
    assert store.get_case(experiment_id, seeded_case) is None


def test_delete_returns_404_for_missing(http, experiment_id):
    resp = http.delete(
        f"{_PREFIX}/test-cases/tc-missing",
        params={"experiment_id": experiment_id},
    )
    assert resp.status_code == 404


def test_prompt_for_fix_renders(http, experiment_id, seeded_case):
    resp = http.post(
        f"{_PREFIX}/test-cases/prompt-for-fix",
        params={"experiment_id": experiment_id, "test_case_id": seeded_case},
    )
    assert resp.status_code == 200
    prompt = resp.json()["prompt"]
    assert seeded_case in prompt


def test_prompt_for_fix_returns_404_for_missing(http, experiment_id):
    resp = http.post(
        f"{_PREFIX}/test-cases/prompt-for-fix",
        params={"experiment_id": experiment_id, "test_case_id": "tc-missing"},
    )
    assert resp.status_code == 404


def test_endpoints_reject_empty_experiment_id(http):
    # ``Query(..., min_length=1)`` on every endpoint surfaces an empty
    # ``experiment_id`` as a 422 at the wire boundary rather than
    # silently routing to the placeholder ``regression_suite_``
    # dataset name.
    assert http.get(f"{_PREFIX}/test-cases", params={"experiment_id": ""}).status_code == 422
    assert http.get(f"{_PREFIX}/test-cases/tc-x", params={"experiment_id": ""}).status_code == 422
    assert (
        http.patch(
            f"{_PREFIX}/test-cases/tc-x",
            params={"experiment_id": ""},
            json={"rationale_summary": "x"},
        ).status_code
        == 422
    )
    assert (
        http.delete(f"{_PREFIX}/test-cases/tc-x", params={"experiment_id": ""}).status_code == 422
    )
