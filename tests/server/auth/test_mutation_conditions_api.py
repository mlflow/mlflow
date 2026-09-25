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
    RESOURCE_ALREADY_EXISTS,
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
    with pytest.raises(MlflowException, match=r"Permission denied.") as ctx:
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
        created = client.add_mutation_conditions(
            role.id,
            "registered_model",
            value_condition="tag_key != 'lifecycle'",
            target_condition="tags.lifecycle = 'dev'",
        )
        assert created.resource_type == "registered_model"
        assert created.value_condition == "tag_key != 'lifecycle'"
        assert created.target_condition == "tags.lifecycle = 'dev'"

        fetched = client.get_mutation_conditions(role.id, "registered_model")
        assert fetched.to_json() == created.to_json()

        listed = client.list_mutation_conditions(role.id)
        assert [c.resource_type for c in listed] == ["registered_model"]

        client.remove_mutation_conditions(role.id, "registered_model")
        assert client.list_mutation_conditions(role.id) == []


def test_add_with_only_a_value_condition(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")
    assert created.value_condition == "tag_key != 'a'"
    assert created.target_condition is None


def test_add_with_only_a_target_condition(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_conditions(role.id, "run", target_condition="tags.a = '1'")
    assert created.value_condition is None
    assert created.target_condition == "tags.a = '1'"


def test_prompt_parity_round_trip(client, monkeypatch, role):
    # D2: prompt and prompt_version are first-class, at parity with the model types.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "prompt", "alias LIKE 'dev-%'")
        client.add_mutation_conditions(role.id, "prompt_version", "tag_key != 'x'")
        listed = {c.resource_type for c in client.list_mutation_conditions(role.id)}
    assert listed == {"prompt", "prompt_version"}


# ---- Partial update over the wire ------------------------------------------


def test_update_omitted_field_is_left_alone(client, monkeypatch, role):
    """Omission and explicit null are different operations, and the client must keep
    them different -- otherwise updating one condition silently clears the other.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'", "tags.b = '1'")
        updated = client.update_mutation_conditions(
            role.id, "run", value_condition="tag_key != 'z'"
        )
    assert updated.value_condition == "tag_key != 'z'"
    assert updated.target_condition == "tags.b = '1'"


def test_update_explicit_none_clears(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'", "tags.b = '1'")
        updated = client.update_mutation_conditions(role.id, "run", target_condition=None)
    assert updated.value_condition == "tag_key != 'a'"
    assert updated.target_condition is None


# ---- Errors ----------------------------------------------------------------


def test_duplicate_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")
        with pytest.raises(MlflowException, match="already exist") as exc:
            client.add_mutation_conditions(role.id, "run", "tag_key != 'b'")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_get_missing_returns_resource_does_not_exist(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found") as exc:
            client.get_mutation_conditions(role.id, "run")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_unsupported_resource_type_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not supported for resource type"):
            client.add_mutation_conditions(role.id, "scorer", "tag_key != 'a'")


def test_malformed_condition_rejected(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="OR is not supported"):
            client.add_mutation_conditions(role.id, "run", "tag_key = 'a' OR tag_key = 'b'")


def test_reserved_tag_key_rejected_in_value_condition(client, monkeypatch, role):
    # D4: an admin must not be able to block MLflow's own tag writes.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="reserved tag keys"):
            client.add_mutation_conditions(role.id, "run", "tag_key = 'mlflow.runName'")


def test_reserved_tag_key_permitted_in_target_condition(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.add_mutation_conditions(
            role.id,
            "registered_model",
            target_condition="tags.`mlflow.prompt.is_prompt` = 'true'",
        )
    assert created.target_condition == "tags.`mlflow.prompt.is_prompt` = 'true'"


# ---- Authorization ---------------------------------------------------------


def test_writes_require_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)

    with assert_unauthenticated():
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")


def test_update_requires_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.update_mutation_conditions(role.id, "run", value_condition="tag_key != 'z'")


def test_remove_requires_role_management(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.remove_mutation_conditions(role.id, "run")

    # And the condition survived the denied attempt.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        assert client.get_mutation_conditions(role.id, "run")


def test_reads_require_role_visibility(client, monkeypatch, role):
    username, password = _non_admin(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")

    with assert_unauthenticated():
        client.get_mutation_conditions(role.id, "run")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.get_mutation_conditions(role.id, "run")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.list_mutation_conditions(role.id)


def test_cascade_on_role_delete(client, monkeypatch, role):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.add_mutation_conditions(role.id, "run", "tag_key != 'a'")
        client.delete_role(role.id)
        with pytest.raises(MlflowException, match="not found"):
            client.list_mutation_conditions(role.id)
