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

import time

import pytest
import requests

from mlflow import MlflowClient, MlflowException
from mlflow.entities import Metric, Param, RunTag
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


def _conditioned_user(
    auth_client, monkeypatch, *, value_condition=None, target_condition=None, permission="EDIT"
):
    """A non-admin who can update every registered model, narrowed by a condition.

    The grant is deliberately broad (``*``) so that any denial in these tests can only come
    from the condition -- the invariant under test is that conditions subtract from what a
    grant allows. ``permission`` is ``MANAGE`` for the routes that check ``can_delete``,
    which ``EDIT`` does not carry.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "registered_model", "*", permission)
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_condition(
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
    # The RFC's headline case: a role may set tags, but not the ones an admin reserves.
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_tags(server, monkeypatch)

    _assert_denied(
        lambda: _set_tag(server, username, password, monkeypatch, name, "lifecycle", "x")
    )


def test_request_condition_allows_an_unrelated_tag_key(server, auth_client, monkeypatch):
    # The other half of the same condition -- it must subtract only what it names.
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
    # An empty table must reproduce today's server. This is the compatibility claim.
    username, password = _conditioned_user(auth_client, monkeypatch)
    name = _model_with_tags(server, monkeypatch)

    _set_tag(server, username, password, monkeypatch, name, "lifecycle", "prod")


def test_admin_bypasses_conditions(server, auth_client, monkeypatch):
    # Conditions restrict delegated authority; they are not a way to constrain admins.
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
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.assign_role(username, role.id)
        # A permissive condition, and deliberately no grant at all.
        auth_client.add_mutation_condition(
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
    # A bulk body must not be a way around a restriction that holds for a single tag.
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


def _exact_name_conditioned_user(auth_client, monkeypatch, resource_type, name, value_condition):
    """A broad grant on ``resource_type``, narrowed by a condition naming exactly one resource."""
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, resource_type, "*", "EDIT")
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_condition(
            role.id,
            resource_type,
            resource_pattern=name,
            value_condition=value_condition,
        )
    return username, password


def test_an_exact_name_value_condition_gates_that_model_create(server, auth_client, monkeypatch):
    """A registry resource IS its name, so a create names the resource it will become.

    Scoping a value condition to one model name has to constrain creating that name.
    Otherwise the restriction is trivially avoidable: delete the model and recreate it
    with exactly the tags the condition forbids. Unlike an experiment or a run -- whose
    ids the server assigns, so a create genuinely names nothing -- the registry keys on
    a caller-supplied name that is known before the handler runs.
    """
    name = f"m-{random_str()}"
    sibling = f"m-{random_str()}"
    username, password = _exact_name_conditioned_user(
        auth_client, monkeypatch, "registered_model", name, "tag_key != 'lifecycle'"
    )

    _assert_denied(
        lambda: _create_model(server, username, password, monkeypatch, name, {"lifecycle": "prod"})
    )
    # The same forbidden tag on a name the row does not govern: the scope still narrows.
    _create_model(server, username, password, monkeypatch, sibling, {"lifecycle": "prod"})


def test_an_exact_name_value_condition_gates_a_recreate_of_that_model(
    server, auth_client, monkeypatch
):
    # The avoidance route the previous test names: delete, then recreate with the tag.
    name = f"m-{random_str()}"
    username, password = _exact_name_conditioned_user(
        auth_client, monkeypatch, "registered_model", name, "tag_key != 'lifecycle'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).create_registered_model(name)
        MlflowClient(server).delete_registered_model(name)

    _assert_denied(
        lambda: _create_model(server, username, password, monkeypatch, name, {"lifecycle": "prod"})
    )


def test_an_exact_name_value_condition_gates_that_prompt_create(server, auth_client, monkeypatch):
    # The same, on the other family the shared create route can produce.
    name = f"p-{random_str()}"
    username, password = _exact_name_conditioned_user(
        auth_client, monkeypatch, "prompt", name, "tag_key != 'lifecycle'"
    )

    def _create(target):
        return requests.post(
            f"{server}/api/2.0/mlflow/registered-models/create",
            json={
                "name": target,
                "tags": [
                    {"key": "lifecycle", "value": "prod"},
                    {"key": IS_PROMPT_TAG_KEY, "value": "true"},
                ],
            },
            auth=(username, password),
            timeout=60,
        )

    assert _create(name).status_code == 403
    assert _create(f"p-{random_str()}").status_code == 200


def test_a_registered_model_condition_does_not_gate_a_prompt_create(
    server, auth_client, monkeypatch
):
    """D2. The create route is shared, and the family comes from the body, so a
    condition on `registered_model` must not govern a prompt create.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        # Broad grants on both families, restricted on only one.
        auth_client.add_role_permission(role.id, "registered_model", "*", "EDIT")
        auth_client.add_role_permission(role.id, "prompt", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_condition(
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


# ---- Aliases (RFC use case 3) -----------------------------------------------


def _model_with_version(server, monkeypatch, tags=None):
    """A registered model with one version, so an alias has something to point at."""
    name = f"model-{random_str()}"
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        experiment_id = client.create_experiment(f"exp-{random_str()}")
        run_id = client.create_run(experiment_id).info.run_id
        client.create_registered_model(name, tags=tags)
        client.create_model_version(name, f"runs:/{run_id}/model", run_id=run_id)
    return name


def _set_alias(server, username, password, monkeypatch, name, alias, version="1"):
    with User(username, password, monkeypatch):
        MlflowClient(server).set_registered_model_alias(name, alias, version)


def test_setting_a_restricted_alias_is_denied(server, auth_client, monkeypatch):
    # The RFC's use case 3: an alias the condition reserves cannot be published.
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="alias != 'champion'"
    )
    name = _model_with_version(server, monkeypatch)

    _assert_denied(lambda: _set_alias(server, username, password, monkeypatch, name, "champion"))


def test_setting_an_unrestricted_alias_is_allowed(server, auth_client, monkeypatch):
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="alias != 'champion'"
    )
    name = _model_with_version(server, monkeypatch)

    _set_alias(server, username, password, monkeypatch, name, "candidate")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_model_version_by_alias(name, "candidate").version == "1"


def test_deleting_a_restricted_alias_is_denied(server, auth_client, monkeypatch):
    """D12 on the alias side. Removing a reserved alias is a way of escaping the
    restriction, so the delete is gated on the alias it names just as the set is.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="alias != 'champion'", permission="MANAGE"
    )
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_registered_model_alias(name, "champion", "1")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_registered_model_alias(name, "champion")


def test_deleting_an_unrestricted_alias_is_allowed(server, auth_client, monkeypatch):
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="alias != 'champion'", permission="MANAGE"
    )
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_registered_model_alias(name, "candidate", "1")

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_registered_model_alias(name, "candidate")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_registered_model(name).aliases == {}


def test_a_tag_clause_does_not_gate_an_alias_route(server, auth_client, monkeypatch):
    """D20 vacuity, in the direction that matters here: an alias route sets no tag, so a
    `tag_key` clause has nothing to judge and must not deny it.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'lifecycle'"
    )
    name = _model_with_version(server, monkeypatch)

    _set_alias(server, username, password, monkeypatch, name, "champion")


def test_an_alias_clause_does_not_gate_a_tag_route(server, auth_client, monkeypatch):
    # The mirror of the above: a tag route sets no alias.
    username, password = _conditioned_user(
        auth_client, monkeypatch, value_condition="alias != 'champion'"
    )
    name = _model_with_version(server, monkeypatch)

    _set_tag(server, username, password, monkeypatch, name, "lifecycle", "prod")


def test_a_resource_alias_condition_gates_the_alias_route(server, auth_client, monkeypatch):
    """A resource condition reads the entry's current alias map, so it can require that an
    alias already point somewhere before any alias may be moved.
    """
    username, password = _conditioned_user(
        auth_client, monkeypatch, target_condition="aliases.champion = '1'"
    )
    without = _model_with_version(server, monkeypatch)

    _assert_denied(
        lambda: _set_alias(server, username, password, monkeypatch, without, "candidate")
    )

    with_alias = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_registered_model_alias(with_alias, "champion", "1")

    _set_alias(server, username, password, monkeypatch, with_alias, "candidate")


# ---- Model versions (tags only, D18) ----------------------------------------


def _version_conditioned_user(
    auth_client, monkeypatch, *, value_condition=None, target_condition=None, permission="EDIT"
):
    """A user who can mutate every model version, narrowed by a condition on the version type.

    The registered_model grant is the container READ the version routes also require; the
    condition is attached to the version type alone, so any denial comes from it.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "registered_model", "*", "READ")
        auth_client.add_role_permission(role.id, "registered_model_version", "*", permission)
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_condition(
                role.id,
                "registered_model_version",
                value_condition=value_condition,
                target_condition=target_condition,
            )
    return username, password


def test_a_restricted_version_tag_is_denied(server, auth_client, monkeypatch):
    # §7.1 case 8.
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'validated'"
    )
    name = _model_with_version(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_model_version_tag(name, "1", "validated", "yes")


def test_an_unrestricted_version_tag_is_allowed(server, auth_client, monkeypatch):
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'validated'"
    )
    name = _model_with_version(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).set_model_version_tag(name, "1", "notes", "looks fine")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_model_version(name, "1").tags["notes"] == "looks fine"


def test_deleting_a_restricted_version_tag_is_denied(server, auth_client, monkeypatch):
    # D12 on the version surface, which checks can_delete rather than can_update.
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'validated'", permission="MANAGE"
    )
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_model_version_tag(name, "1", "validated", "yes")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_model_version_tag(name, "1", "validated")


def test_a_version_resource_condition_reads_the_versions_own_tags(server, auth_client, monkeypatch):
    """The condition targets the version by its composed id, so it must see the version's
    own tags -- not the parent registry entry's, which are a different resource.
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.stage = 'candidate'"
    )
    # The parent carries the tag; the version does not. A condition on the version must
    # therefore deny, or it is reading the wrong resource's state.
    name = _model_with_version(server, monkeypatch, tags={"stage": "candidate"})

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_model_version_tag(name, "1", "notes", "x")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_model_version_tag(name, "1", "stage", "candidate")

    with User(username, password, monkeypatch):
        MlflowClient(server).set_model_version_tag(name, "1", "notes", "x")


def test_a_renamed_model_keeps_its_exact_scoped_condition(server, auth_client, monkeypatch):
    """A registry resource IS its name, so a rename moves the identity the row names.

    Grants were already migrated on rename; conditions were not, so the rename dropped
    every restriction on the resource and left the grants they narrowed fully intact.
    """
    old_name = f"m-{random_str()}"
    new_name = f"m-{random_str()}"
    username, password = _exact_name_conditioned_user(
        auth_client, monkeypatch, "registered_model", old_name, "tag_key != 'lifecycle'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).create_registered_model(old_name)

    # Denied before the rename...
    _assert_denied(
        lambda: _set_tag(server, username, password, monkeypatch, old_name, "lifecycle", "prod")
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).rename_registered_model(old_name, new_name)
    # ...and still denied after it, under the new name.
    _assert_denied(
        lambda: _set_tag(server, username, password, monkeypatch, new_name, "lifecycle", "prod")
    )


def test_a_renamed_model_keeps_the_conditions_scoped_to_its_versions(
    server, auth_client, monkeypatch
):
    """The other scope axis: a version row is wildcard-only and names the model as its
    container, so a rename has to move the container pattern too.
    """
    old_name = f"m-{random_str()}"
    new_name = f"m-{random_str()}"
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "registered_model", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_condition(
            role.id,
            "registered_model_version",
            container_resource_type="registered_model",
            container_resource_pattern=old_name,
            value_condition="tag_key != 'lifecycle'",
        )
        MlflowClient(server).create_registered_model(old_name)
        MlflowClient(server).create_model_version(old_name, source="s3://bucket/path")
        MlflowClient(server).rename_registered_model(old_name, new_name)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_model_version_tag(new_name, "1", "lifecycle", "prod")


def test_a_registered_model_condition_does_not_gate_a_version_tag(server, auth_client, monkeypatch):
    """D2 across the parent/child boundary: the two are distinct resource types, so a
    condition on the entry must not travel to its versions.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "registered_model", "*", "EDIT")
        auth_client.add_role_permission(role.id, "registered_model_version", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_condition(
            role.id, "registered_model", value_condition="tag_key != 'validated'"
        )
    name = _model_with_version(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).set_model_version_tag(name, "1", "validated", "yes")


# ---- Runs (RFC use case 1) --------------------------------------------------


def _run_conditioned_user(auth_client, monkeypatch, *, value_condition=None, target_condition=None):
    """A user who can mutate every run in the workspace, narrowed by a run condition."""
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "experiment", "*", "EDIT")
        auth_client.add_role_permission(role.id, "run", "*", "EDIT")
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_condition(
                role.id, "run", value_condition=value_condition, target_condition=target_condition
            )
    return username, password


def _a_run(server, monkeypatch, tags=None):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        experiment_id = client.create_experiment(f"exp-{random_str()}")
        run = client.create_run(experiment_id, tags=tags)
    return run.info.run_id


def test_a_restricted_run_tag_is_denied(server, auth_client, monkeypatch):
    # §7.1 case 2.
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_tag(run_id, "approved", "yes")


def test_an_unrestricted_run_tag_is_allowed(server, auth_client, monkeypatch):
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).set_tag(run_id, "notes", "fine")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_run(run_id).data.tags["notes"] == "fine"


def test_deleting_a_restricted_run_tag_is_denied(server, auth_client, monkeypatch):
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch, tags={"approved": "yes"})

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_tag(run_id, "approved")


def test_logging_a_param_is_not_denied_by_a_tag_condition(server, auth_client, monkeypatch):
    """`LogParam` shares its validator with the tag routes. A body carrying a param and no
    tag must not be denied -- the same shape as case 5b, on the route that shares a
    validator rather than having its own.
    """
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).log_param(run_id, "alpha", "0.1")


def test_a_metrics_only_log_batch_is_not_denied_by_a_tag_condition(
    server, auth_client, monkeypatch
):
    """§7.1 case 5b, the case the tracker flags as most likely to catch a real bug: a batch
    that writes metrics and no tags has nothing for a `tag_key` clause to judge.
    """
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).log_batch(run_id, metrics=[Metric("loss", 0.5, 0, 0)])

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_run(run_id).data.metrics["loss"] == 0.5


def test_a_log_batch_carrying_a_restricted_tag_is_denied(server, auth_client, monkeypatch):
    """§7.1 case 5a. The batch path must be gated on the tags it writes, or it is a way
    around the single-tag route.
    """
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).log_batch(run_id, tags=[RunTag("approved", "yes")])


def test_a_log_batch_is_denied_when_any_one_of_several_tags_fails(server, auth_client, monkeypatch):
    # A bulk body must not dilute a restriction that holds for one tag.
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).log_batch(
                run_id, tags=[RunTag("notes", "fine"), RunTag("approved", "yes")]
            )


def test_creating_a_run_with_a_restricted_tag_is_denied(server, auth_client, monkeypatch):
    # §7.1 case 4, at CREATE scope.
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).create_run(experiment_id, tags={"approved": "yes"})


def test_creating_a_run_without_tags_is_allowed(server, auth_client, monkeypatch):
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")

    with User(username, password, monkeypatch):
        MlflowClient(server).create_run(experiment_id)


def test_a_reserved_tag_key_cannot_be_named_by_a_request_condition(auth_client, monkeypatch):
    """D17 + D4, and the reason `run_name` needs no extraction.

    The store persists `run_name` as the reserved `mlflow.runName` tag, so if a request
    condition could name that key, `UpdateRun` would be a way to write a governed tag
    without going through a tag route. D4 closes it at the authoring end instead: a
    reserved key is refused on the request side, so no such clause can exist to be
    bypassed. Asserted here because the guarantee rests on the two rules agreeing -- a
    change to either would turn this into a live bypass.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        with pytest.raises(MlflowException, match=r"reserved tag keys"):
            auth_client.add_mutation_condition(
                role.id, "run", value_condition="tag_key != 'mlflow.runName'"
            )

        # The resource side rejects it too: the same keys are freely settable, so a target
        # condition reading one restricts nothing -- the holder renames the run and passes.
        with pytest.raises(MlflowException, match=r"reserved tag keys"):
            auth_client.add_mutation_condition(
                role.id, "run", target_condition="tags.`mlflow.runName` != 'secret'"
            )


def test_an_update_run_rename_is_not_denied_by_a_tag_condition(server, auth_client, monkeypatch):
    """The companion to the above: `UpdateRun` sets no tag the condition layer sees, so a
    rename passes under a `tag_key` clause (D20 vacuity).
    """
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'approved'"
    )
    run_id = _a_run(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).update_run(run_id, name="renamed")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert MlflowClient(server).get_run(run_id).info.run_name == "renamed"


def test_vacuity_holds_under_a_positive_clause(server, auth_client, monkeypatch):
    """The discriminating form of case 5b.

    A negative clause (`tag_key != 'x'`) passes whether the projection is vacuous or
    yields a placeholder, so it cannot tell the two apart -- a test written that way stays
    green even if vacuity breaks. A *positive* clause separates them: if a body carrying no
    tag were treated as carrying one, `tag_key = 'notes'` would judge that phantom value
    and deny.

    So this asserts, for each body that sets no tag, that a positive clause does not deny
    it -- which is only true if the clause genuinely does not apply.
    """
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key = 'notes'"
    )
    run_id = _a_run(server, monkeypatch)

    with User(username, password, monkeypatch):
        client = MlflowClient(server)
        # Metrics-only batch (5b), params-only batch, a param, and a rename: none sets a tag.
        client.log_batch(run_id, metrics=[Metric("loss", 0.5, 0, 0)])
        client.log_batch(run_id, params=[Param("beta", "2")])
        client.log_param(run_id, "alpha", "0.1")
        client.update_run(run_id, name="renamed")

    # The same clause still bites on a body that *does* set a non-matching tag, so the
    # condition is live rather than inert.
    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_tag(run_id, "other", "x")


def test_creating_a_run_without_tags_holds_under_a_positive_clause(
    server, auth_client, monkeypatch
):
    # The same discrimination at CREATE scope.
    username, password = _run_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key = 'notes'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")

    with User(username, password, monkeypatch):
        MlflowClient(server).create_run(experiment_id)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).create_run(experiment_id, tags={"other": "x"})


# ---- Traces, including the bulk delete's two modes ---------------------------


def _trace_conditioned_user(
    auth_client, monkeypatch, *, value_condition=None, target_condition=None, permission="EDIT"
):
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "experiment", "*", permission)
        auth_client.add_role_permission(role.id, "trace", "*", permission)
        auth_client.add_role_permission(role.id, "assessment", "*", permission)
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_condition(
                role.id, "trace", value_condition=value_condition, target_condition=target_condition
            )
    return username, password


def _experiment_scoped_trace_conditioned_user(
    auth_client, monkeypatch, experiment_id, *, target_condition, permission="MANAGE"
):
    """A user whose trace condition is scoped to ONE experiment, not the whole workspace.

    The scoped form is the one that exposed the loader contract: a context with no
    ``parent_resource_id`` makes the store match only UNSCOPED conditions, so a condition
    written like this was silently never loaded and never evaluated.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, "experiment", "*", permission)
        auth_client.add_role_permission(role.id, "trace", "*", permission)
        auth_client.add_role_permission(role.id, "assessment", "*", permission)
        auth_client.assign_role(username, role.id)
        auth_client.add_mutation_condition(
            role.id,
            "trace",
            container_resource_type="experiment",
            container_resource_pattern=experiment_id,
            target_condition=target_condition,
        )
    return username, password


def _a_trace(server, monkeypatch):
    """One finished trace, returning its experiment and trace ids."""
    import mlflow

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        mlflow.set_tracking_uri(server)
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_span(name="s"):
            pass
        traces = MlflowClient(server).search_traces([experiment_id])
    return experiment_id, traces[0].info.trace_id


def test_a_restricted_trace_tag_is_denied(server, auth_client, monkeypatch):
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'reviewed'"
    )
    _, trace_id = _a_trace(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_trace_tag(trace_id, "reviewed", "yes")


def test_an_unrestricted_trace_tag_is_allowed(server, auth_client, monkeypatch):
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'reviewed'"
    )
    _, trace_id = _a_trace(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).set_trace_tag(trace_id, "notes", "fine")


def test_deleting_traces_by_id_is_gated_on_each_trace(server, auth_client, monkeypatch):
    # §7.1 case 9a. The named traces are conditioned on their own current state.
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.reviewed = 'yes'", permission="MANAGE"
    )
    experiment_id, trace_id = _a_trace(server, monkeypatch)

    # The trace does not carry the required tag, so the delete is refused.
    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_traces(experiment_id, trace_ids=[trace_id])

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_trace_tag(trace_id, "reviewed", "yes")

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, trace_ids=[trace_id])


def _two_traces(server, monkeypatch):
    """Two finished traces in ONE experiment, the second strictly later than the first."""
    import mlflow

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        mlflow.set_tracking_uri(server)
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_span(name="older"):
            pass
        # The window bound is a millisecond timestamp, so the two traces have to land in
        # different milliseconds for the bound to separate them at all.
        time.sleep(0.05)
        with mlflow.start_span(name="newer"):
            pass
        traces = sorted(
            MlflowClient(server).search_traces([experiment_id]),
            key=lambda trace: trace.info.request_time,
        )
    assert traces[0].info.request_time < traces[1].info.request_time
    return experiment_id, traces[0].info, traces[1].info


def test_deleting_traces_by_timestamp_ignores_a_failing_trace_outside_the_window(
    server, auth_client, monkeypatch
):
    """The precision the parent pushdown is for: judge the traces the delete can reach.

    Asking the parent "does this experiment hold ANY trace that fails?" refuses a delete
    whose window contains nothing objectionable, purely because something objectionable
    exists elsewhere in the experiment. The delete's own predicate bounds the population,
    so the probe carries it: here the newer trace fails the condition but sits outside the
    window, and the delete proceeds.
    """
    experiment_id, older, newer = _two_traces(server, monkeypatch)
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.reviewed = 'yes'", permission="MANAGE"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        # The older trace satisfies the condition; the newer one never gets the tag.
        MlflowClient(server).set_trace_tag(older.trace_id, "reviewed", "yes")

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, max_timestamp_millis=older.request_time)


def test_deleting_traces_by_timestamp_refuses_a_failing_trace_inside_the_window(
    server, auth_client, monkeypatch
):
    # The other half: widening the window to cover the failing trace refuses the delete.
    experiment_id, older, newer = _two_traces(server, monkeypatch)
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.reviewed = 'yes'", permission="MANAGE"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_trace_tag(older.trace_id, "reviewed", "yes")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_traces(
                experiment_id, max_timestamp_millis=newer.request_time
            )


def test_deleting_traces_by_timestamp_is_refused_when_a_trace_fails_the_condition(
    server, auth_client, monkeypatch
):
    """§7.1 case 9b and D21.

    Timestamp mode does not name the traces it will delete, so the condition cannot be
    evaluated against an enumerated set. It is answered by the parent instead -- "does
    this experiment hold any trace that fails?" in one pushdown -- which is what keeps
    the mode from passing vacuously: an empty id list would otherwise satisfy every
    clause and make timestamp mode a way around any resource condition on traces.

    Here the experiment holds a trace without the required tag, so the delete is refused.
    The refusal is conservative by construction: that trace may sit outside the timestamp
    range, but the gate cannot know which traces the range covers, so it declines.
    """
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.reviewed = 'yes'", permission="MANAGE"
    )
    experiment_id, _ = _a_trace(server, monkeypatch)

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_traces(experiment_id, max_timestamp_millis=2**62)


def test_deleting_traces_by_timestamp_is_allowed_when_every_trace_passes(
    server, auth_client, monkeypatch
):
    """The other side of the parent pushdown, and why it beats a blanket refusal.

    Every trace in the experiment satisfies the condition, so no trace the delete could
    possibly reach violates it and there is nothing for the gate to protect. This used to
    be refused outright, because the context carried no parent and the mode had no way to
    answer the question.
    """
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.reviewed = 'yes'", permission="MANAGE"
    )
    experiment_id, trace_id = _a_trace(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_trace_tag(trace_id, "reviewed", "yes")

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, max_timestamp_millis=2**62)


def test_an_experiment_scoped_trace_condition_gates_a_named_delete(
    server, auth_client, monkeypatch
):
    """The scoped form of the condition must actually be enforced.

    The trace context carried no ``parent_resource_id``, and the loader skips such a
    context when deciding which parents are in play -- so the store matched only UNSCOPED
    conditions and this experiment-scoped one was never loaded, never evaluated, and the
    delete ran unconditioned. Workspace-wide conditions still applied, which is exactly
    what hid the gap.
    """
    experiment_id, trace_id = _a_trace(server, monkeypatch)
    username, password = _experiment_scoped_trace_conditioned_user(
        auth_client, monkeypatch, experiment_id, target_condition="tags.reviewed = 'yes'"
    )

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).delete_traces(experiment_id, trace_ids=[trace_id])

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        MlflowClient(server).set_trace_tag(trace_id, "reviewed", "yes")

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, trace_ids=[trace_id])


def test_a_trace_condition_scoped_to_another_experiment_does_not_gate_this_one(
    server, auth_client, monkeypatch
):
    """Scoping has to cut both ways, or "scoped" would just mean "slower workspace-wide".

    The condition names a different experiment, so it must not restrict a delete in this
    one -- the delete proceeds even though the trace lacks the tag the condition asks for.
    """
    other_experiment_id, _ = _a_trace(server, monkeypatch)
    experiment_id, trace_id = _a_trace(server, monkeypatch)
    username, password = _experiment_scoped_trace_conditioned_user(
        auth_client, monkeypatch, other_experiment_id, target_condition="tags.reviewed = 'yes'"
    )

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, trace_ids=[trace_id])


def test_deleting_traces_by_timestamp_is_unaffected_without_a_target_condition(
    server, auth_client, monkeypatch
):
    """The other half of D21, and what keeps the refusal proportionate: the mode is only
    refused when a resource condition exists to be evaded. A request condition does not
    trigger it, and neither does an empty table.
    """
    username, password = _trace_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_key != 'reviewed'", permission="MANAGE"
    )
    experiment_id, _ = _a_trace(server, monkeypatch)

    with User(username, password, monkeypatch):
        MlflowClient(server).delete_traces(experiment_id, max_timestamp_millis=2**62)


# ---- Experiments and logged models ------------------------------------------


def _typed_conditioned_user(
    auth_client,
    monkeypatch,
    resource_type,
    *,
    value_condition=None,
    target_condition=None,
    permission="EDIT",
    extra=(),
):
    """A user granted `permission` on `resource_type` (plus any `extra` grants), with a
    condition attached to that type alone.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        auth_client.add_role_permission(role.id, resource_type, "*", permission)
        for extra_type, extra_permission in extra:
            auth_client.add_role_permission(role.id, extra_type, "*", extra_permission)
        auth_client.assign_role(username, role.id)
        if value_condition is not None or target_condition is not None:
            auth_client.add_mutation_condition(
                role.id,
                resource_type,
                value_condition=value_condition,
                target_condition=target_condition,
            )
    return username, password


def test_a_restricted_experiment_tag_is_denied(server, auth_client, monkeypatch):
    """The legacy experiment surface resolves a Permission directly rather than through a
    Requirement list, so its conditions are called beside the grant check instead of
    through `authorize`. This asserts that path is actually wired.
    """
    username, password = _typed_conditioned_user(
        auth_client, monkeypatch, "experiment", value_condition="tag_key != 'locked'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_experiment_tag(experiment_id, "locked", "yes")


def test_an_unrestricted_experiment_tag_is_allowed(server, auth_client, monkeypatch):
    username, password = _typed_conditioned_user(
        auth_client, monkeypatch, "experiment", value_condition="tag_key != 'locked'"
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        experiment_id = MlflowClient(server).create_experiment(f"exp-{random_str()}")

    with User(username, password, monkeypatch):
        MlflowClient(server).set_experiment_tag(experiment_id, "team", "analytics")


def test_creating_an_experiment_with_a_restricted_tag_is_denied(server, auth_client, monkeypatch):
    username, password = _typed_conditioned_user(
        auth_client, monkeypatch, "experiment", value_condition="tag_key != 'locked'"
    )

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).create_experiment(f"exp-{random_str()}", tags={"locked": "yes"})


def test_a_restricted_logged_model_tag_is_denied(server, auth_client, monkeypatch):
    username, password = _typed_conditioned_user(
        auth_client,
        monkeypatch,
        "logged_model",
        value_condition="tag_key != 'certified'",
        extra=(("experiment", "EDIT"),),
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        experiment_id = client.create_experiment(f"exp-{random_str()}")
        model = client.create_logged_model(experiment_id, name=f"m-{random_str()}")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_logged_model_tags(model.model_id, {"certified": "yes"})


def test_a_logged_model_tag_batch_is_denied_when_any_tag_fails(server, auth_client, monkeypatch):
    """`SetLoggedModelTags` takes a repeated field, so the any-fails-denies rule applies
    here as it does to LogBatch.
    """
    username, password = _typed_conditioned_user(
        auth_client,
        monkeypatch,
        "logged_model",
        value_condition="tag_key != 'certified'",
        extra=(("experiment", "EDIT"),),
    )
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        experiment_id = client.create_experiment(f"exp-{random_str()}")
        model = client.create_logged_model(experiment_id, name=f"m-{random_str()}")

    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_logged_model_tags(
                model.model_id, {"notes": "fine", "certified": "yes"}
            )


def test_reads_are_never_gated_across_every_wired_type(server, auth_client, monkeypatch):
    """§7.1 case 11, generalized past the one route the RFC names.

    Conditions gate mutations only. A condition that would deny every write must leave
    every read untouched, on each type now wired -- otherwise wiring a route has silently
    restricted the read path beside it.
    """
    username = random_str()
    password = random_str(12)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        auth_client.create_user(username, password)
        role = auth_client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")
        for resource_type in ("experiment", "run", "registered_model", "registered_model_version"):
            auth_client.add_role_permission(role.id, resource_type, "*", "EDIT")
            # A condition no tag can satisfy, so any leak into a read path denies it.
            auth_client.add_mutation_condition(
                role.id, resource_type, value_condition="tag_key = 'impossible'"
            )
        auth_client.assign_role(username, role.id)

    # `_model_with_version` manages its own admin credentials, so it is called outside the
    # block below rather than nested inside it.
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        admin = MlflowClient(server)
        experiment_id = admin.create_experiment(f"exp-{random_str()}")
        run_id = admin.create_run(experiment_id, tags={"any": "thing"}).info.run_id
        admin.set_registered_model_tag(name, "lifecycle", "prod")

    with User(username, password, monkeypatch):
        client = MlflowClient(server)
        assert client.get_experiment(experiment_id).experiment_id == experiment_id
        assert client.get_run(run_id).info.run_id == run_id
        assert client.get_registered_model(name).name == name
        assert client.get_model_version(name, "1").version == "1"
        assert client.search_runs([experiment_id])


def _version_tagged(server, monkeypatch, tags):
    """A model with one version carrying ``tags`` ON THE VERSION.

    ``_model_with_version``'s ``tags`` land on the registered model; a target condition on
    ``registered_model_version`` reads the version's own tags (D18), so they have to be set
    here instead.
    """
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        for key, value in tags.items():
            client.set_model_version_tag(name, "1", key, value)
    return name


@pytest.mark.parametrize(
    ("version_tags", "allowed"),
    [
        ({"lifecycle": "dev"}, True),
        ({"lifecycle": "prod"}, False),
        ({}, False),  # D20: absence fails on the resource side
    ],
)
def test_a_target_condition_gates_a_stage_transition(
    server, auth_client, monkeypatch, version_tags, allowed
):
    """The stages API is unconditioned only on the VALUE half.

    `validate_can_update_model_or_prompt_version` used to be documented as unconditioned
    outright (D15/D16), which reads as "a condition cannot block a stage transition". Only
    half of that is true: the body sets no tag, so every request clause is vacuous and a
    value condition can never refuse it -- but the shared helper declares a full context
    with the version's resource id, so a TARGET condition is evaluated against the
    version's current tags like any other mutation.

    Exempting it would make the stages API a way around a restriction that holds for every
    other write to the same version, so this test exists to keep the exemption from being
    "tidied up" into the stage route later.
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = _version_tagged(server, monkeypatch, version_tags)

    def transition():
        with User(username, password, monkeypatch):
            MlflowClient(server).transition_model_version_stage(name, "1", "Staging")

    if allowed:
        transition()
    else:
        with pytest.raises(MlflowException, match=r"Permission denied"):
            transition()


def _model_with_two_versions(server, monkeypatch, *, v1_tags, v2_tags, v1_stage=None):
    """A model with versions 1 and 2, each carrying its own tags.

    ``v1_stage`` parks version 1 in a stage as ADMIN, so a later transition of version 2
    into that stage is what triggers the archive cascade over version 1.
    """
    name = _model_with_version(server, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client = MlflowClient(server)
        for key, value in v1_tags.items():
            client.set_model_version_tag(name, "1", key, value)
        client.create_model_version(name=name, source="s3://bucket/path")
        for key, value in v2_tags.items():
            client.set_model_version_tag(name, "2", key, value)
        if v1_stage is not None:
            client.transition_model_version_stage(name, "1", v1_stage)
    return name


def test_archiving_siblings_is_refused_when_a_sibling_fails_its_condition(
    server, auth_client, monkeypatch
):
    """F-0040. ``archive_existing_versions`` mutates versions the request never names.

    The store moves every OTHER version of the model already in the target stage to
    ``Archived``. Version 2 passes the condition and version 1 does not, so transitioning
    2 into Staging while archiving 1 must be refused -- otherwise the stages API archives a
    version the role is forbidden to touch.
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = _model_with_two_versions(
        server,
        monkeypatch,
        v1_tags={"lifecycle": "prod"},
        v2_tags={"lifecycle": "dev"},
        v1_stage="Staging",
    )
    with User(username, password, monkeypatch):
        with pytest.raises(MlflowException, match=r"Permission denied"):
            MlflowClient(server).transition_model_version_stage(
                name, "2", "Staging", archive_existing_versions=True
            )


def test_archiving_siblings_is_permitted_when_every_sibling_passes(
    server, auth_client, monkeypatch
):
    # The same request with a sibling the condition admits must still go through.
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = _model_with_two_versions(
        server,
        monkeypatch,
        v1_tags={"lifecycle": "dev"},
        v2_tags={"lifecycle": "dev"},
        v1_stage="Staging",
    )
    with User(username, password, monkeypatch):
        MlflowClient(server).transition_model_version_stage(
            name, "2", "Staging", archive_existing_versions=True
        )


def test_a_failing_sibling_outside_the_target_stage_does_not_refuse(
    server, auth_client, monkeypatch
):
    """The cascade is narrowed to the stage the archive actually reaches.

    Version 1 fails the condition but sits in NO stage, so no archive can touch it.
    Judging the transition against every version of the model would refuse a request that
    mutates nothing objectionable -- the same over-refusal the timestamp window exists to
    prevent for ``DeleteTraces``.
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = _model_with_two_versions(
        server,
        monkeypatch,
        v1_tags={"lifecycle": "prod"},
        v2_tags={"lifecycle": "dev"},
        v1_stage=None,
    )
    with User(username, password, monkeypatch):
        MlflowClient(server).transition_model_version_stage(
            name, "2", "Staging", archive_existing_versions=True
        )


def test_a_transition_without_the_archive_flag_ignores_siblings(server, auth_client, monkeypatch):
    """No archive, no cascade. A plain transition mutates only the version it names, so a
    failing sibling in the target stage is irrelevant to it.
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, target_condition="tags.lifecycle = 'dev'"
    )
    name = _model_with_two_versions(
        server,
        monkeypatch,
        v1_tags={"lifecycle": "prod"},
        v2_tags={"lifecycle": "dev"},
        v1_stage="Staging",
    )
    with User(username, password, monkeypatch):
        MlflowClient(server).transition_model_version_stage(
            name, "2", "Staging", archive_existing_versions=False
        )


def test_a_value_condition_cannot_refuse_a_stage_transition(server, auth_client, monkeypatch):
    """The other half of the same split, and the part D15/D16 actually decided.

    The condition would deny any tag write at all, yet the transition goes through: a stage
    is not in the clause vocabulary, so a request carrying no tag has nothing to test (D13).
    """
    username, password = _version_conditioned_user(
        auth_client, monkeypatch, value_condition="tag_value = 'no-such-value'"
    )
    name = _version_tagged(server, monkeypatch, {"lifecycle": "dev"})

    with User(username, password, monkeypatch):
        MlflowClient(server).transition_model_version_stage(name, "1", "Staging")

    # The same condition really would have refused a tag write on that version.
    with pytest.raises(MlflowException, match=r"Permission denied"):
        with User(username, password, monkeypatch):
            MlflowClient(server).set_model_version_tag(name, "1", "notes", "anything")
