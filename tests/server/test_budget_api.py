"""Tests for the budget windows FastAPI endpoint."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mlflow.gateway.budget_tracker as _bt_module
from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker.in_memory import InMemoryBudgetTracker
from mlflow.server.gateway_api import budget_router


@pytest.fixture(autouse=True)
def _reset_budget_tracker():
    _bt_module._budget_tracker = None
    yield
    _bt_module._budget_tracker = None


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(budget_router)
    return TestClient(app)


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    duration_unit=BudgetDurationUnit.DAYS,
    duration_value=1,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration_unit=duration_unit,
        duration_value=duration_value,
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=BudgetAction.ALERT,
        created_at=0,
        last_updated_at=0,
    )


def test_list_budget_windows_empty(client):
    response = client.get("/ajax-api/3.0/mlflow/gateway/budgets/windows")
    assert response.status_code == 200
    assert response.json() == {"windows": {}}


def test_list_budget_windows_returns_window_data(client):
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(budget_policy_id="bp-1", budget_amount=50.0)
    tracker.refresh_policies([policy])
    tracker.record_cost(12.5)

    with patch("mlflow.server.gateway_api.get_budget_tracker", return_value=tracker):
        response = client.get("/ajax-api/3.0/mlflow/gateway/budgets/windows")

    assert response.status_code == 200
    data = response.json()
    assert "windows" in data
    assert "bp-1" in data["windows"]
    window = data["windows"]["bp-1"]
    assert window["current_spend"] == 12.5
    assert "window_start_ms" in window
    assert "window_end_ms" in window
    assert window["window_end_ms"] > window["window_start_ms"]


def test_list_budget_windows_multiple_policies(client):
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])
    tracker.record_cost(30.0)

    with patch("mlflow.server.gateway_api.get_budget_tracker", return_value=tracker):
        response = client.get("/ajax-api/3.0/mlflow/gateway/budgets/windows")

    assert response.status_code == 200
    data = response.json()
    assert set(data["windows"].keys()) == {"bp-1", "bp-2"}
    assert data["windows"]["bp-1"]["current_spend"] == 30.0
    assert data["windows"]["bp-2"]["current_spend"] == 30.0


def test_list_budget_windows_timestamps_are_milliseconds(client):
    tracker = InMemoryBudgetTracker()
    policy = _make_policy()
    tracker.refresh_policies([policy])

    with patch("mlflow.server.gateway_api.get_budget_tracker", return_value=tracker):
        response = client.get("/ajax-api/3.0/mlflow/gateway/budgets/windows")

    assert response.status_code == 200
    window = response.json()["windows"]["bp-test"]
    # Timestamps should be reasonable millisecond values (after year 2000)
    min_ms = int(datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert window["window_start_ms"] >= min_ms
    assert window["window_end_ms"] >= min_ms


def test_list_budget_windows_zero_spend(client):
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(budget_amount=100.0)
    tracker.refresh_policies([policy])
    # No cost recorded

    with patch("mlflow.server.gateway_api.get_budget_tracker", return_value=tracker):
        response = client.get("/ajax-api/3.0/mlflow/gateway/budgets/windows")

    assert response.status_code == 200
    assert response.json()["windows"]["bp-test"]["current_spend"] == 0.0
