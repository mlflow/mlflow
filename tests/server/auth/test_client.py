import pytest
from unittest import mock

from mlflow.server.auth.client import AuthServiceClient
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


def test_client_create_run(client, mock_session):
    experiment_id = mock.Mock()
    username = mock.Mock()
    permission = mock.Mock()
    client.create_experiment_permission(experiment_id, username, permission)

    call_args = mock_session.request.call_args
    assert call_args.args[0] == "POST"
    assert (
        call_args.args[1] == f"{client.tracking_uri}/api/2.0/mlflow/experiments/permissions/create"
    )
    assert call_args.kwargs["json"] == {
        "experiment_id": experiment_id,
        "username": username,
        "permission": permission,
    }
