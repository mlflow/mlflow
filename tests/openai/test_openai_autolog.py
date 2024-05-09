import json
from unittest import mock

import openai
import pytest

import mlflow
from mlflow import MlflowClient

from tests.openai.conftest import is_v1


@pytest.fixture
def client(mock_openai):
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog(client):
    mlflow.openai.autolog(log_models=True)
    with mock.patch("mlflow.openai.log_model") as log_model_mock, mock.patch(
        "mlflow.tracking.MlflowClient.log_text"
    ) as log_text_mock:
        output = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            temperature=0,
        )
        # ensure openai is mocked
        assert output.id == "chatcmpl-123"
        log_model_mock.assert_called_once()
        assert log_text_mock.call_count == 2
        assert log_text_mock.call_args_list[0].artifact_path.endswith("input.json")
        assert log_text_mock.call_args_list[1].artifact_path.endswith("output.json")


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_loaded_chat_completions_autolog(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run() as run:
        client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
        )

    loaded_model = mlflow.openai.load_model(f"runs:/{run.info.run_id}/model")
    assert loaded_model == {
        "model": "gpt-3.5-turbo",
        "task": "chat.completions",
    }

    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": client.base_url,
        }
    )
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    # expected output from mock_openai
    assert pyfunc_model.predict("test") == [json.dumps(messages)]


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_autolog_with_registered_model_name(client):
    registered_model_name = "test_model"
    mlflow.openai.autolog(log_models=True, registered_model_name=registered_model_name)
    client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    registered_model = MlflowClient().get_registered_model(registered_model_name)
    assert registered_model.name == registered_model_name
