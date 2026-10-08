# Route and client tests for the mutation-conditions admin API.
#
# Phase 1's reviewable claim is that conditions can be authored and read back while
# nothing enforces them -- so these tests cover the API surface and its authorization,
# not evaluation (see ``test_conditions.py`` for that).

from contextlib import contextmanager

import pytest

from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.protos.databricks_pb2 import (
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
    ErrorCode,
)
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

_WORKSPACE = "ws1"


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
        yield AuthServiceClient(url)


@contextmanager
def assert_unauthenticated():
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as ctx:
        yield
    assert ctx.value.error_code == ErrorCode.Name(UNAUTHENTICATED)


@contextmanager
def assert_unauthorized():
    with pytest.raises(MlflowException, match=r"Permission denied") as ctx:
        yield
    assert ctx.value.error_code == ErrorCode.Name(PERMISSION_DENIED)


@pytest.fixture
def role(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        return client.create_role(workspace=_WORKSPACE, name=f"dev-{random_str()}")


def _non_admin(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
    return username, password


# ---- Round trip ------------------------------------------------------------


def test_add_get_list_remove_round_trip(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(
            role.id,
            "registered_model",
            value_condition="tag_key != 'lifecycle'",
            target_condition="tags.lifecycle = 'dev'",
        )
        assert created.resource_type == "registered_model"
        assert created.value_condition == "tag_key != 'lifecycle'"
        assert created.target_condition == "tags.lifecycle = 'dev'"

        fetched = client.get_mutation_condition(created.id)
        assert fetched.to_json() == created.to_json()

        listed = client.list_mutation_conditions(role.id)
        assert [c.resource_type for c in listed] == ["registered_model"]

        client.remove_mutation_condition(created.id)
        assert client.list_mutation_conditions(role.id) == []


def test_add_with_only_a_value_condition(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    assert created.value_condition == "tag_key != 'a'"
    assert created.target_condition is None


def test_add_with_only_a_target_condition(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(role.id, "run", target_condition="tags.a = '1'")
    assert created.value_condition is None
    assert created.target_condition == "tags.a = '1'"


def test_prompt_parity_round_trip(client, monkeypatch, role):
    # D2: prompt and prompt_version are first-class, at parity with the model types.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_condition(role.id, "prompt", value_condition="alias LIKE 'dev-%'")
        client.add_mutation_condition(role.id, "prompt_version", value_condition="tag_key != 'x'")
        listed = {c.resource_type for c in client.list_mutation_conditions(role.id)}
    assert listed == {"prompt", "prompt_version"}


# ---- Partial update over the wire ------------------------------------------


def test_update_omitted_field_is_left_alone(client, monkeypatch, role):
    """Omission and explicit null are different operations, and the client must keep
    them different -- otherwise updating one condition silently clears the other.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(
            role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
        )
        updated = client.update_mutation_condition(created.id, value_condition="tag_key != 'z'")
    assert updated.value_condition == "tag_key != 'z'"
    assert updated.target_condition == "tags.b = '1'"


def test_update_explicit_none_clears(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(
            role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
        )
        updated = client.update_mutation_condition(created.id, target_condition=None)
    assert updated.value_condition == "tag_key != 'a'"
    assert updated.target_condition is None


# ---- Errors ----------------------------------------------------------------


def test_several_conditions_for_one_type_are_accepted(client, monkeypatch, role):
    """No longer a duplicate: a role may hold several conditions per type, which is
    what makes per-parent scoping expressible. They AND, so adding one can only
    narrow what the role may do.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        first = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
        second = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'b'")
        assert {first.condition_slot, second.condition_slot} == {1, 2}
        assert len(client.list_mutation_conditions(role.id)) == 2


def test_add_with_neither_filter_is_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="at least one"):
            client.add_mutation_condition(role.id, "run")


def test_get_missing_returns_resource_does_not_exist(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found") as exc:
            client.get_mutation_condition(99999)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_unsupported_resource_type_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not supported for resource type"):
            client.add_mutation_condition(role.id, "scorer", value_condition="tag_key != 'a'")


def test_malformed_condition_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="OR is not supported"):
            client.add_mutation_condition(
                role.id, "run", value_condition="tag_key = 'a' OR tag_key = 'b'"
            )


def test_reserved_tag_key_rejected_in_value_condition(client, monkeypatch, role):
    # D4: an admin must not be able to block MLflow's own tag writes.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="reserved tag keys"):
            client.add_mutation_condition(
                role.id, "run", value_condition="tag_key = 'mlflow.runName'"
            )


def test_reserved_tag_key_rejected_in_target_condition_too(client, monkeypatch, role):
    # D4 is symmetric: a reserved key is refused in either namespace.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="reserved tag keys"):
            client.add_mutation_condition(
                role.id,
                "registered_model",
                target_condition="tags.`mlflow.prompt.is_prompt` = 'true'",
            )


# ---- Authorization ---------------------------------------------------------


def test_a_non_admin_can_read_their_own_conditions(client, monkeypatch, role):
    """The self path is reachable by an ordinary user; the role-keyed one is not for them.

    This is the whole point of the endpoint. A user whose write was refused needs to see
    what restricts them, and they cannot get that from ``roles/mutation-conditions/list``
    -- they do not know which of their roles carries the condition, and asking about a
    role is asking about a policy that is not theirs.

    Driven over raw HTTP because the client has no method for it and because what is
    being tested is the route wiring and the open gate, not a store call.
    """
    import requests

    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'bob'")
        client.assign_role(username, role.id)

    url = f"{client.tracking_uri}/api/3.0/mlflow/users/current/mutation-conditions"

    anonymous = requests.get(url, timeout=30)
    assert anonymous.status_code == 401, (
        f"authentication is still required -- the gate is open, not absent: {anonymous.text}"
    )

    response = requests.get(url, auth=(username, password), timeout=30)
    assert response.status_code == 200, response.text
    rows = response.json()["mutation_conditions"]
    assert [r["value_condition"] for r in rows] == ["tag_key != 'bob'"]
    assert rows[0]["role_name"] == role.name, (
        "the row must name the role it came from, since that is the only way a user can "
        "tell which of their roles is restricting them"
    )
    assert rows[0]["resource_type"] == "run"


def test_the_self_path_shows_nothing_of_a_role_the_user_does_not_hold(client, monkeypatch, role):
    """Scoping is to roles HELD, not to roles that exist.

    The fail-open direction here is disclosure rather than access: a flat listing of
    every role's conditions would hand any authenticated user the whole policy, which is
    precisely the over-broad read the role-keyed endpoint already allows and this one
    exists to avoid.
    """
    import requests

    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'carol'")
        # deliberately NOT assigned to the user

    response = requests.get(
        f"{client.tracking_uri}/api/3.0/mlflow/users/current/mutation-conditions",
        auth=(username, password),
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.json()["mutation_conditions"] == [], (
        "a user holding no roles must see no conditions, however many exist"
    )


def test_writes_require_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)

    with assert_unauthenticated():
        client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")


def test_update_requires_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.update_mutation_condition(created.id, value_condition="tag_key != 'z'")


def test_remove_requires_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.remove_mutation_condition(created.id)

    # And the condition survived the denied attempt.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert client.get_mutation_condition(created.id)


def test_workspace_admin_can_manage_a_condition_by_id(client, monkeypatch, role):
    """A condition is addressed by its own id, so the workspace whose admin may manage
    it has to be resolved *through* the condition to its owning role.

    Regression test: while the routes were keyed on ``role_id`` the workspace fell out
    of the request directly. Moving to id addressing removed it, and the
    role-management check then could not run at all -- a workspace admin was refused
    with a malformed-request error rather than being authorized. The failure was
    fail-closed, but it made the route unusable for exactly the non-super-admin it
    exists to serve.
    """
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
        admin_role = client.create_role(workspace=_WORKSPACE, name=f"wsadmin-{random_str()}")
        client.add_role_permission(admin_role.id, "workspace", "*", "MANAGE")
        client.assign_role(username, admin_role.id)
        created = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    # The workspace admin never names the role, only the condition.
    with User(username, password, monkeypatch):
        assert client.get_mutation_condition(created.id).id == created.id
        updated = client.update_mutation_condition(created.id, value_condition="tag_key != 'z'")
        assert updated.value_condition == "tag_key != 'z'"
        client.remove_mutation_condition(created.id)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert client.list_mutation_conditions(role.id) == []


def test_reads_require_role_visibility(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")

    with assert_unauthenticated():
        client.get_mutation_condition(created.id)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.get_mutation_condition(created.id)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.list_mutation_conditions(role.id)


def test_cascade_on_role_delete(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
        client.delete_role(role.id)
        with pytest.raises(MlflowException, match="not found"):
            client.list_mutation_conditions(role.id)


# ---- The user-addressed add ------------------------------------------------


def test_user_addressed_add_creates_the_per_user_role(client, monkeypatch):
    """A direct condition does not require a direct grant to exist first.

    The route is the condition analogue of ``grant_user_permission``: the caller names a
    user, and the per-user role the condition actually lands on is resolved -- and
    created -- server-side.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        username = f"u-{random_str()}"
        client.create_user(username, "password1234")
        assert client.list_user_roles(username) == []

        created = client.add_user_mutation_condition(
            username, "run", target_condition="tags.lifecycle != 'prod'"
        )
        assert created.target_condition == "tags.lifecycle != 'prod'"

        roles = client.list_user_roles(username)
        assert len(roles) == 1, "the per-user role must be created on demand"
        assert created.role_id == roles[0].id
        # Addressable by its own id afterwards, like any other condition.
        assert client.get_mutation_condition(created.id).id == created.id


def test_user_addressed_add_reuses_the_role_a_direct_grant_made(client, monkeypatch):
    # It must land on the same role the direct grants use, not a second one.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        username = f"u-{random_str()}"
        client.create_user(username, "password1234")
        client.grant_user_permission(username, "experiment", "*", "EDIT")
        roles = client.list_user_roles(username)
        assert len(roles) == 1

        created = client.add_user_mutation_condition(
            username, "run", target_condition="tags.x = 'y'"
        )
        assert created.role_id == roles[0].id
        assert len(client.list_user_roles(username)) == 1, "no second role"


def test_user_addressed_add_refuses_a_filterless_object(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        username = f"u-{random_str()}"
        client.create_user(username, "password1234")
        with pytest.raises(MlflowException, match="at least one of"):
            client.add_user_mutation_condition(username, "run")


def test_user_addressed_add_is_refused_for_a_non_admin(client, monkeypatch):
    """Same gate as the role-addressed add: only an admin or workspace admin may set one.

    Worth its own case because this route resolves the workspace differently -- it names
    no role, so the shared role-workspace resolver cannot serve it.
    """
    username, password = _non_admin(client, monkeypatch)
    target = f"u-{random_str()}"
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(target, "password1234")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", username)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", password)
    with pytest.raises(MlflowException, match="PERMISSION_DENIED|Permission denied"):
        client.add_user_mutation_condition(target, "run", target_condition="tags.x = 'y'")
