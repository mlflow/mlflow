from typing import Any
from unittest.mock import MagicMock, patch

from mlflow.assistant.providers.presets import list_ollama_tags


def _mock_response(body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def test_list_ollama_tags():
    body = {"models": [{"model": "llama3"}, {"model": "mistral"}, {"other": "x"}]}
    with patch(
        "mlflow.assistant.providers.presets.requests.get",
        return_value=_mock_response(body),
    ) as mock_get:
        names = list_ollama_tags("http://localhost:11434/")
    assert names == ["llama3", "mistral"]
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", headers={}, timeout=10)


def test_list_ollama_tags_forwards_api_key_as_bearer():
    body = {"models": [{"model": "llama3"}]}
    with patch(
        "mlflow.assistant.providers.presets.requests.get",
        return_value=_mock_response(body),
    ) as mock_get:
        list_ollama_tags("http://localhost:11434/", api_key="secret")
    mock_get.assert_called_once_with(
        "http://localhost:11434/api/tags",
        headers={"Authorization": "Bearer secret"},
        timeout=10,
    )
