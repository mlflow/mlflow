import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestSpec,
)
from mlflow.server.agent_playground.test_cases_router import (
    test_cases_router as _router,
)


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
        judge=JudgeSpec(criteria="response is friendly"),
    )


@pytest.fixture
def http():
    app = FastAPI()
    app.include_router(_router)
    return TestClient(app)


@pytest.fixture
def seeded_case(experiment_id, assertion_spec):
    return store.insert_case(
        experiment_id,
        assertion_spec,
        conversation_messages=[{"role": "user", "content": "hi"}],
        source_feedback_id="fb-001",
        source_trace_id="tr-001",
        source_assistant_message_id="msg-001",
    )


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


def test_list_paginates(http, experiment_id, assertion_spec):
    ids = [store.insert_case(experiment_id, assertion_spec) for _ in range(3)]
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
        json={"experiment_id": experiment_id, "rationale_summary": "updated reason"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Verify "only" rationale_summary changed; all other fields preserved.
    assert body["spec"]["rationale_summary"] == "updated reason"
    assert body["spec"]["strategy"] == "assertion"
    assert body["spec"]["max_turns"] == 5
    assert body["spec"]["persona"] is None
    assert body["spec"]["judge"] is None
    assert body["spec"]["assertion"] == {
        "must_contain": ["docs"],
        "must_not_contain": [],
        "must_call_tool": ["search_docs"],
        "must_not_call_tool": [],
    }
    assert body["promoted"] is False
    assert body["source_feedback_ids"] == ["fb-001"]


def test_patch_switches_strategy_with_matching_payload(
    http, experiment_id, seeded_case, judge_spec
):
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        json={
            "experiment_id": experiment_id,
            "strategy": "judge",
            "judge": judge_spec.judge.model_dump(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["spec"]["strategy"] == "judge"
    # Strategy-switch auto-clears the inactive payload; verify the
    # previous assertion clauses dropped.
    assert body["spec"]["assertion"] is None
    assert body["spec"]["judge"]["criteria"] == "response is friendly"


def test_patch_rejects_strategy_swap_without_payload(http, experiment_id, seeded_case):
    resp = http.patch(
        f"{_PREFIX}/test-cases/{seeded_case}",
        json={"experiment_id": experiment_id, "strategy": "judge"},
    )
    assert resp.status_code == 422


def test_patch_clear_persona_removes_persona(http, experiment_id, assertion_spec):
    spec = assertion_spec.model_copy(update={"persona": PersonaSpec(goal="g", persona="p")})
    test_case_id = store.insert_case(experiment_id, spec)
    resp = http.patch(
        f"{_PREFIX}/test-cases/{test_case_id}",
        json={"experiment_id": experiment_id, "clear_persona": True},
    )
    assert resp.status_code == 200
    assert resp.json()["spec"]["persona"] is None


def test_patch_returns_404_for_missing(http, experiment_id):
    resp = http.patch(
        f"{_PREFIX}/test-cases/tc-missing",
        json={"experiment_id": experiment_id, "rationale_summary": "x"},
    )
    assert resp.status_code == 404


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
        json={"experiment_id": experiment_id, "test_case_id": seeded_case},
    )
    assert resp.status_code == 200
    prompt = resp.json()["prompt"]
    assert seeded_case in prompt


def test_prompt_for_fix_returns_404_for_missing(http, experiment_id):
    resp = http.post(
        f"{_PREFIX}/test-cases/prompt-for-fix",
        json={"experiment_id": experiment_id, "test_case_id": "tc-missing"},
    )
    assert resp.status_code == 404
