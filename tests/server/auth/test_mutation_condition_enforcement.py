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
import requests

from mlflow import MlflowClient, MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
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


# ---- Create scope ----------------------------------------------------------


def _create_model(server, username, password, monkeypatch, name, tags=None):
    with User(username, password, monkeypatch):
        MlflowClient(server).create_registered_model(name, tags=tags)


def test_create_condition_denies_a_disallowed_tag_in_the_body(server, auth_client, monkeypatch):
    """§7.1 case 3. A create carries a repeated tags field, so the condition has to be
    evaluated against every tag the body will store.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )

    with pytest.raises(MlflowException, match=r"Permission denied"):
        _create_model(
            server, username, password, monkeypatch, f"m-{random_str()}", {"lifecycle": "prod"}
        )


def test_create_allows_a_body_whose_tags_all_pass(server, auth_client, monkeypatch):
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = f"m-{random_str()}"

    _create_model(server, username, password, monkeypatch, name, {"team": "analytics"})

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_registered_model(name).tags["team"] == "analytics"


def test_create_denies_when_any_one_tag_of_several_fails(server, auth_client, monkeypatch):
    """A bulk body must not be a way around a restriction that holds for a single tag."""
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )

    with pytest.raises(MlflowException, match=r"Permission denied"):
        _create_model(
            server,
            username,
            password,
            monkeypatch,
            f"m-{random_str()}",
            {"team": "analytics", "lifecycle": "prod", "owner": "me"},
        )


def test_create_with_no_tags_is_not_denied_by_a_tag_condition(server, auth_client, monkeypatch):
    """D20 on the request side: a body that names no tag has nothing for a tag_key clause
    to object to, so the clause is vacuous rather than denying.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = f"m-{random_str()}"

    _create_model(server, username, password, monkeypatch, name)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_registered_model(name).name == name


def test_a_resource_condition_does_not_gate_a_create(server, auth_client, monkeypatch):
    """A create has no prior state, so a resource condition cannot apply to it. If it
    did, a target condition would make creation impossible for the role.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = f"m-{random_str()}"

    _create_model(server, username, password, monkeypatch, name)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_registered_model(name).name == name


def test_a_registered_model_condition_does_not_gate_a_prompt_create(
    server, auth_client, monkeypatch
):
    """D2. The create route is shared, and the family comes from the body, so a
    condition on `registered_model` must not govern a prompt create.
    """
    username, password = random_str(), random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        # Broad grants on both families, restricted on only one.
        auth_client.add_role_permission(role.id, "registered_model", "*", "EDIT")
        auth_client.add_role_permission(role.id, "prompt", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_conditions(
            role.id, "registered_model", value_condition="tag_key != 'lifecycle'"
        )

    # The same disallowed tag key: denied for a registered model, allowed for a prompt,
    # because the condition is scoped to the registered_model family alone.
    with pytest.raises(MlflowException, match=r"Permission denied"):
        _create_model(
            server, username, password, monkeypatch, f"m-{random_str()}", {"lifecycle": "prod"}
        )

    # Posted directly: the client refuses to register a prompt through the model API, but
    # the route is shared and the server must classify from the body.
    response = requests.post(
        f"{server}/api/2.0/mlflow/registered-models/create",
        json={
            "name": f"p-{random_str()}",
            "tags": [
                {"key": "lifecycle", "value": "prod"},
                {"key": IS_PROMPT_TAG_KEY, "value": "true"},
            ],
        },
        auth=(username, password),
        timeout=60,
    )
    assert response.status_code == 200, response.text


# ---- Delete gating (D12) ---------------------------------------------------


def test_deleting_a_reserved_tag_is_denied(server, auth_client, monkeypatch):
    """D12. Removing a tag a condition reserves is a way of escaping the restriction it
    expresses, so a delete is gated by the same `tag_key` clause a set is.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch, {"lifecycle": "prod"})

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_registered_model_tag(name, "lifecycle")


def test_deleting_an_unreserved_tag_is_allowed(server, auth_client, monkeypatch):
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch, {"team": "analytics"})

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_registered_model_tag(name, "team")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert "team" not in MlflowClient(server).get_registered_model(name).tags


def test_a_value_clause_does_not_gate_a_delete(server, auth_client, monkeypatch):
    """A deletion names a key but no value, so a `tag_value` clause has nothing to
    constrain and stays vacuous.

    The clause is deliberately positive. A negative one like ``tag_value != 'prod'``
    would pass whether the absent value arrived as ``None`` or as an empty string, so it
    could not tell a correct implementation from one that invents a value; ``= 'dev'``
    passes only when the value is genuinely absent.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_value = 'dev'"
    )
    name = _model_with_tags(server, monkeypatch, {"lifecycle": "prod"})

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_registered_model_tag(name, "lifecycle")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert "lifecycle" not in MlflowClient(server).get_registered_model(name).tags


def test_a_rename_is_not_denied_by_a_tag_condition(server, auth_client, monkeypatch):
    """A rename carries no tag, so a tag clause is vacuous. This is the route that shares
    the update validator but has no request values at all.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch)
    renamed = f"{name}-renamed"

    with User(username, password, monkeypatch):
        MlflowClient(server).rename_registered_model(name, renamed)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_registered_model(renamed).name == renamed


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
