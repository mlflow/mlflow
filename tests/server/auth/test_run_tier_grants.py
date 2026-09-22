"""End-to-end tests for the run tier (RFC 0000 re-pointing).

``UpdateRun`` was gated on the run's experiment. It is now gated on the run tier with the
experiment as fallback, which adds three behaviours over master:

* escalation  -- a run grant authorizes with no experiment grant at all
* restriction -- a run DENY refuses even with experiment MANAGE
* override    -- a run grant decides outright, so a lower one refuses

Inheritance (no run grant -> the experiment decides) must be unchanged, and the existing
run tests in ``test_auth.py`` cover that side.
"""

import pytest
import requests

from mlflow import MlflowClient
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    READ,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_RUN,
    RESOURCE_TYPE_TRACE,
)
from mlflow.utils.os import is_windows

from tests.server.auth.auth_test_utils import (
    User,
    create_user,
    grant_role_permission,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def client(tmp_path):
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        },
        server_type="flask",
    ) as url:
        yield MlflowClient(url)


def _update_run(tracking_uri, run_id, auth):
    return requests.post(
        f"{tracking_uri}/api/2.0/mlflow/runs/update",
        json={"run_id": run_id, "status": "FINISHED"},
        auth=auth,
    )


@pytest.fixture
def run_fixture(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    """An experiment with one run, owned by a user the test does not act as."""
    owner, owner_password = create_user(client.tracking_uri)
    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("run-tier-grants")
        run = client.create_run(experiment_id)
    return experiment_id, run.info.run_id


def test_a_run_grant_authorizes_without_any_experiment_grant(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """Escalation: the run tier is authoritative when present, so it stands alone."""
    experiment_id, run_id = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_RUN, "*", EDIT.name)

    assert _update_run(client.tracking_uri, run_id, (user, password)).status_code == 200


def test_a_run_deny_refuses_despite_experiment_manage(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """Restriction: a DENY is absolute within its tier and is never rescued upward."""
    experiment_id, run_id = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, user, RESOURCE_TYPE_EXPERIMENT, experiment_id, MANAGE.name
    )
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_RUN, "*", DENY.name)

    assert _update_run(client.tracking_uri, run_id, (user, password)).status_code == 403


def test_a_lower_run_grant_overrides_the_experiment_downward(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """Tier override applies in both directions: the nearer tier decides, full stop.

    This is the behaviour that makes a positive requirement unsafe on a tier the
    pre-existing check did not consult -- see ACTION_NOT_DENIED.
    """
    experiment_id, run_id = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, user, RESOURCE_TYPE_EXPERIMENT, experiment_id, EDIT.name
    )
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_RUN, "*", READ.name)

    assert _update_run(client.tracking_uri, run_id, (user, password)).status_code == 403


def test_an_absent_run_grant_still_inherits_the_experiment(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """Master parity: with no run grant the experiment decides, exactly as before."""
    experiment_id, run_id = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, user, RESOURCE_TYPE_EXPERIMENT, experiment_id, EDIT.name
    )

    assert _update_run(client.tracking_uri, run_id, (user, password)).status_code == 200


def test_a_nonexistent_run_is_denied_not_defaulted(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch
):
    """Fail closed: an unresolvable run must not fall through to default_permission."""
    user, password = create_user(client.tracking_uri)
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_RUN, "*", MANAGE.name)

    response = _update_run(client.tracking_uri, "no-such-run-id", (user, password))
    assert response.status_code == 403


def test_a_per_id_run_grant_is_rejected_at_the_source(client: MlflowClient):
    """Sub-resources are wildcard-only grain: a per-id run grant must not be writable,
    since it could not be enforced in list/search paths.
    """
    user, _ = create_user(client.tracking_uri)
    with pytest.raises(requests.HTTPError, match="400"):
        grant_role_permission(
            client.tracking_uri, user, RESOURCE_TYPE_RUN, "some-run-id", EDIT.name
        )


# --------------------------------------------------------------------- CreateRun


def _create_run(tracking_uri, experiment_id, auth):
    return requests.post(
        f"{tracking_uri}/api/2.0/mlflow/runs/create",
        json={"experiment_id": experiment_id},
        auth=auth,
    )


def test_create_run_is_gated_on_the_run_tier(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """CreateRun has no run id, so the run tier is addressed at its wildcard grain."""
    experiment_id, _ = run_fixture
    denied, denied_password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, denied, RESOURCE_TYPE_EXPERIMENT, experiment_id, EDIT.name
    )
    grant_role_permission(client.tracking_uri, denied, RESOURCE_TYPE_RUN, "*", DENY.name)
    assert (
        _create_run(client.tracking_uri, experiment_id, (denied, denied_password)).status_code
        == 403
    )

    allowed, allowed_password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, allowed, RESOURCE_TYPE_EXPERIMENT, experiment_id, EDIT.name
    )
    assert (
        _create_run(client.tracking_uri, experiment_id, (allowed, allowed_password)).status_code
        == 200
    )


# --------------------------------------------------------------------- trace tier


def test_start_trace_is_gated_on_the_trace_tier(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    experiment_id, _ = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(
        client.tracking_uri, user, RESOURCE_TYPE_EXPERIMENT, experiment_id, MANAGE.name
    )
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_TRACE, "*", DENY.name)

    response = requests.post(
        f"{client.tracking_uri}/api/2.0/mlflow/traces",
        json={
            "experiment_id": experiment_id,
            "timestamp_ms": 1,
            "request_metadata": [],
            "tags": [],
        },
        auth=(user, password),
    )
    assert response.status_code == 403


def test_a_trace_grant_authorizes_start_trace_without_an_experiment_grant(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, run_fixture
):
    """Escalation on the trace tier."""
    experiment_id, _ = run_fixture
    user, password = create_user(client.tracking_uri)
    grant_role_permission(client.tracking_uri, user, RESOURCE_TYPE_TRACE, "*", EDIT.name)

    response = requests.post(
        f"{client.tracking_uri}/api/2.0/mlflow/traces",
        json={
            "experiment_id": experiment_id,
            "timestamp_ms": 1,
            "request_metadata": [],
            "tags": [],
        },
        auth=(user, password),
    )
    assert response.status_code == 200
