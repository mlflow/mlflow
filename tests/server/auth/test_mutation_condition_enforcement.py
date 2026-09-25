# End-to-end enforcement tests for mutation conditions.
#
# Phase 4's reviewable claim is that a condition actually gates a mutating route, so
# these tests drive a real server over HTTP: an admin authors the condition, a non-admin
# holding a role grant attempts the mutation, and the response is the assertion.
#
# The first wired route is ``SetRegisteredModelTag`` -- §7.1 case 1, and the RFC's
# primary use case. It exercises the whole stack in one request: request-value
# extraction, the condition load, request-condition evaluation, resource fetch through
# the request-scoped cache, and resource-condition evaluation.

import pytest

from mlflow import MlflowClient, MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server

_WORKSPACE = "default"


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def server(tmp_path):
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
        yield url


@pytest.fixture
def auth_client(server):
    return AuthServiceClient(server)


def _conditioned_user(auth_client, monkeypatch, *, value_condition=None, target_condition=None):
    """A non-admin who can update every registered model, narrowed by a condition.

    The grant is deliberately broad (``EDIT`` on ``*``) so that any denial in these
    tests can only come from the condition -- the invariant under test is that
    conditions subtract from what a grant allows.
    """
    username, password = random_str(), random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "registered_model", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_conditions(
                role.id,
                "registered_model",
                value_condition=value_condition,
                target_condition=target_condition,
            )
    return username, password


def _model_with_tags(server, monkeypatch, tags=None):
    name = f"model-{random_str()}"
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).create_registered_model(name, tags=tags)
    return name


def _set_tag(server, username, password, monkeypatch, name, key, value):
    with User(username, password, monkeypatch):
        MlflowClient(server).set_registered_model_tag(name, key, value)


def _assert_denied(fn):
    with pytest.raises(MlflowException, match=r"Permission denied") as ctx:
        fn()
    assert ctx.value.error_code == ErrorCode.Name(PERMISSION_DENIED)


# ---- Request conditions ----------------------------------------------------


def test_request_condition_denies_the_disallowed_tag_key(server, auth_client, monkeypatch):
    """The RFC's headline case: a role may set tags, but not the ones an admin reserves."""
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch)

    _assert_denied(
        lambda: _set_tag(server, username, password, monkeypatch, name, "lifecycle", "x")
    )


def test_request_condition_allows_an_unrelated_tag_key(server, auth_client, monkeypatch):
    """The other half of the same condition -- it must subtract only what it names."""
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch)

    _set_tag(server, username, password, monkeypatch, name, "team", "analytics")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        model = MlflowClient(server).get_registered_model(name)
    assert model.tags["team"] == "analytics"


def test_request_condition_constrains_the_value_as_well_as_the_key(
    server, auth_client, monkeypatch
):
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_value != 'prod'"
    )
    name = _model_with_tags(server, monkeypatch)

    _assert_denied(lambda: _set_tag(server, username, password, monkeypatch, name, "env", "prod"))
    _set_tag(server, username, password, monkeypatch, name, "env", "dev")


# ---- Resource conditions ---------------------------------------------------


def test_resource_condition_denies_a_model_whose_state_does_not_match(
    server, auth_client, monkeypatch
):
    """The resource condition reads *current* state, so the same request is allowed or
    denied depending on the model it targets.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    dev = _model_with_tags(server, monkeypatch, {"lifecycle": "dev"})
    prod = _model_with_tags(server, monkeypatch, {"lifecycle": "prod"})

    _set_tag(server, username, password, monkeypatch, dev, "team", "analytics")
    _assert_denied(lambda: _set_tag(server, username, password, monkeypatch, prod, "team", "x"))


def test_resource_condition_denies_a_model_missing_the_tag_entirely(
    server, auth_client, monkeypatch
):
    """D20's documented footgun, asserted so it cannot regress silently: absence fails
    on the resource side, so an untagged model is denied by a condition naming that tag.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    untagged = _model_with_tags(server, monkeypatch)

    _assert_denied(lambda: _set_tag(server, username, password, monkeypatch, untagged, "t", "v"))


# ---- The core invariants ---------------------------------------------------


def test_no_conditions_configured_behaves_exactly_as_before(server, auth_client, monkeypatch):
    """An empty table must reproduce today's server. This is the compatibility claim."""
    username, password = _conditioned_user(auth_client, monkeypatch)
    name = _model_with_tags(server, monkeypatch)

    _set_tag(server, username, password, monkeypatch, name, "lifecycle", "prod")


def test_admin_bypasses_conditions(server, auth_client, monkeypatch):
    """Conditions restrict delegated authority; they are not a way to constrain admins."""
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.update_user_admin(username, True)

    _set_tag(server, username, password, monkeypatch, name, "lifecycle", "prod")


def test_conditions_never_confer_access(server, auth_client, monkeypatch):
    """A condition that would *pass* cannot substitute for a missing grant: the base
    check runs first and conditions only ever subtract.
    """
    username, password = random_str(), random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.assign_role(username, role.id)
        # A permissive condition, and deliberately no grant at all.
        auth_client.add_mutation_conditions(
            role.id, "registered_model", value_condition="tag_key != 'nothing-matches-this'"
        )
    name = _model_with_tags(server, monkeypatch)

    _assert_denied(lambda: _set_tag(server, username, password, monkeypatch, name, "team", "x"))


def test_both_conditions_must_pass(server, auth_client, monkeypatch):
    username, password = _conditioned_user(
        auth_client,
        monkeypatch,
        value_condition="tag_key != 'lifecycle'",
        target_condition="tags.lifecycle = 'dev'",
    )
    dev = _model_with_tags(server, monkeypatch, {"lifecycle": "dev"})

    # Resource passes, request fails.
    _assert_denied(lambda: _set_tag(server, username, password, monkeypatch, dev, "lifecycle", "x"))
    # Both pass.
    _set_tag(server, username, password, monkeypatch, dev, "team", "analytics")


def test_a_second_role_cannot_lift_the_first_roles_restriction(server, auth_client, monkeypatch):
    """Non-monotonicity, which is the property that makes conditions safe to reason
    about: capability is the union of grants, but every applicable condition must pass.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        unrestricted = auth_client.create_role(
            workspace=_WORKSPACE, name=f"unrestricted-{random_str()}"
        )
        auth_client.add_role_permission(unrestricted.id, "registered_model", "*", "MANAGE")
        auth_client.assign_role(username, unrestricted.id)
    name = _model_with_tags(server, monkeypatch)

    _assert_denied(
        lambda: _set_tag(server, username, password, monkeypatch, name, "lifecycle", "x")
    )


# ---- Reads are never gated -------------------------------------------------


def test_a_read_is_not_gated_by_a_condition(server, auth_client, monkeypatch):
    """Conditions gate mutations only. A resource condition that the model fails must
    not stop the same user reading it.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    prod = _model_with_tags(server, monkeypatch, {"lifecycle": "prod"})

    with User(username, password, monkeypatch):
        model = MlflowClient(server).get_registered_model(prod)
    assert model.name == prod
