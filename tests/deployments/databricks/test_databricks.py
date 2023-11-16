from unittest import mock

from mlflow.deployments import get_deploy_client


def test_get_deploy_client():
    get_deploy_client("databricks")


def test_create_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.create_endpoint(
            name="test",
            config={
                "served_entities": [
                    {
                        "name": "test",
                        "external_model": {
                            "name": "gpt-4",
                            "provider": "openai",
                            "openai_config": {
                                "openai_api_key": "secret",
                            },
                        },
                    }
                ],
                "task": "llm/v1/chat",
            },
        )
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_get_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.get_endpoint(endpoint="test")
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_list_endpoints():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"endpoints": [{"name": "test"}]}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.list_endpoints()
        mock_request.assert_called_once()
        assert resp == {"endpoints": [{"name": "test"}]}


def test_update_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint(
            endpoint="test",
            config={
                "served_entities": [
                    {
                        "name": "test",
                        "external_model": {
                            "name": "gpt-4",
                            "provider": "openai",
                            "openai_config": {
                                "openai_api_key": "secret",
                            },
                        },
                    }
                ],
                "task": "llm/v1/chat",
            },
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_delete_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.delete_endpoint(endpoint="test")
        mock_request.assert_called_once()
        assert resp == {}


def test_predict():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.predict(endpoint="test", inputs={})
        mock_request.assert_called_once()
        assert resp == {"foo": "bar"}
