import pytest
from unittest import mock

from mlflow.server.auth.client import AuthServiceClient
from mlflow.server.auth.routes import (
    CREATE_EXPERIMENT_PERMISSION,
    GET_EXPERIMENT_PERMISSION,
    UPDATE_EXPERIMENT_PERMISSION,
    DELETE_EXPERIMENT_PERMISSION,
    CREATE_REGISTERED_MODEL_PERMISSION,
    GET_REGISTERED_MODEL_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
)
from tests.helper_functions import LOCALHOST, get_safe_port


@pytest.fixture
def client():
    server_port = get_safe_port()
    url = f"http://{LOCALHOST}:{server_port}"
    yield AuthServiceClient(url)


@pytest.fixture
def mock_session():
    with mock.patch("requests.Session") as mock_session:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = "{}"
        mock_session.return_value.request.return_value = mock_response
        yield mock_session.return_value


def test_client_create_experiment_permission(client, mock_session):
    experiment_id = mock.Mock()
    username = mock.Mock()
    permission = mock.Mock()
    client.create_experiment_permission(experiment_id, username, permission)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "POST"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{CREATE_EXPERIMENT_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "experiment_id": experiment_id,
        "username": username,
        "permission": permission,
    }


def test_client_get_experiment_permission(client, mock_session):
    experiment_id = mock.Mock()
    username = mock.Mock()
    client.get_experiment_permission(experiment_id, username)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "GET"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{GET_EXPERIMENT_PERMISSION}"
    )
    assert call_args.kwargs["params"] == {
        "experiment_id": experiment_id,
        "username": username,
    }


def test_client_update_experiment_permission(client, mock_session):
    experiment_id = mock.Mock()
    username = mock.Mock()
    permission = mock.Mock()
    client.update_experiment_permission(experiment_id, username, permission)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "PATCH"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{UPDATE_EXPERIMENT_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "experiment_id": experiment_id,
        "username": username,
        "permission": permission,
    }


def test_client_delete_experiment_permission(client, mock_session):
    experiment_id = mock.Mock()
    username = mock.Mock()
    client.get_experiment_permission(experiment_id, username)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "DELETE"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{DELETE_EXPERIMENT_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "experiment_id": experiment_id,
        "username": username,
    }


def test_client_create_registered_model_permission(client, mock_session):
    name = mock.Mock()
    username = mock.Mock()
    permission = mock.Mock()
    client.create_registered_model_permission(name, username, permission)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "POST"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{CREATE_REGISTERED_MODEL_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "name": name,
        "username": username,
        "permission": permission,
    }


def test_client_get_registered_model_permission(client, mock_session):
    name = mock.Mock()
    username = mock.Mock()
    client.get_registered_model_permission(name, username)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "GET"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{GET_REGISTERED_MODEL_PERMISSION}"
    )
    assert call_args.kwargs["params"] == {
        "name": name,
        "username": username,
    }


def test_client_update_registered_model_permission(client, mock_session):
    name = mock.Mock()
    username = mock.Mock()
    permission = mock.Mock()
    client.update_registered_model_permission(name, username, permission)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "PATCH"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{UPDATE_REGISTERED_MODEL_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "name": name,
        "username": username,
        "permission": permission,
    }


def test_client_delete_registered_model_permission(client, mock_session):
    name = mock.Mock()
    username = mock.Mock()
    client.get_registered_model_permission(name, username)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "DELETE"
    assert (
        call_args.args[1] == f"{client.tracking_uri}{DELETE_REGISTERED_MODEL_PERMISSION}"
    )
    assert call_args.kwargs["json"] == {
        "name": name,
        "username": username,
    }
